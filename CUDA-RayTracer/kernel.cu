#include "kernel.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Ray.cuh"
#include "BVH.cuh"
#include "Camera.cuh"
#include "ViewPlane.cuh"
#include "Model.h"

#include <stb_image_write.h>
#include "Random.cuh"
#include "Emissive.cuh"
#include "AreaLight.cuh"
#include "Rectangle.cuh"

typedef unsigned char uchar;

const int SCR_WIDTH = 1000;
const int SCR_HEIGHT = 1000;

bool buffer_reset = false;

float3  cam_pos             = make_float3(0, 200, 1000);
float3  cam_lookat          = make_float3(0, 70, 0);
float3  cam_dir;
float3  cam_right;
float3  cam_up;
float3  cam_worldUp         = make_float3(0,1,0);
float   cam_zoom            = 30;
float   cam_lens_radius     = 0.1;
float   cam_f               = 900.f;
float   cam_d               = 100;
float   cam_yaw             = 4.7;
float   cam_pitch           = -3.2;

float cam_movement_spd = 10;

// image buffer storing accumulated pixel samples
float4* accumulatebuffer;
// final output buffer storing averaged pixel samples
float3* finaloutputbuffer;

//BVHNode* nodes = nullptr;
//Triangle* triangles = nullptr;

Paths* paths;
Queues* queues;

const uint32_t pathCount = SCR_WIDTH * SCR_HEIGHT;
const uint32_t maxPathLength = 6;

void save_image(const char* filename, const void* data) {
    int stride = 3 * SCR_WIDTH;
    stbi_write_png(filename, SCR_WIDTH, SCR_HEIGHT, 3, data, stride);
}

__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

__device__ float3 trace_ray(Ray ray, LinearBVHNode* nodes) {

    float3 L = make_float3(0, 0, 0);
    float3 beta = make_float3(1, 1, 1);
    int maxDepth = 2;

    for (int bounces = 0;; ++bounces) {
        
        ShadeRec sr;
        sr.bvh = nodes;
        Intersect(ray, sr, nodes);
        
        sr.depth = bounces;
        
        if (!sr.hit_an_obj || bounces >= maxDepth)
            break;
        
        bool in_shadow = false;

        //float3 lightdir = normalize(make_float3(0, 300, 300) - sr.local_hit_point);
        float3 lightdir, sample_point;
        g_lights[0]->get_direction(sr, lightdir, sample_point);

        //if (g_lights[0]->casts_shadows()) {
            Ray shadow_ray(sr.local_hit_point, lightdir);
            in_shadow = g_lights[0]->in_shadow(shadow_ray, sr);
        //}

        if (!in_shadow) {
            float n_dot_wi = fmaxf(dot(sr.normal, lightdir), 0.f);
            float3 f = make_float3(1) * M_1_PI * n_dot_wi * sr.material_ptr->get_cd();
            float3 Li = g_lights[0]->L(sr, lightdir, sample_point);

            L += f * Li * beta;
        }
        
        // Sample BRDF to get new path direction
        float3 N = sr.normal;
        float3 wo = -sr.ray.d;

        float e0 = random();
        float e1 = random();

        float sinTheta = sqrtf(1 - e0 * e0);
        float phi = 2 * M_PI * e1;
        float x = sinTheta * cosf(phi);
        float z = sinTheta * sinf(phi);
        float3 sp = make_float3(x, e0, z);

        float3 T = normalize(cross(N, get_orthogonal_vec(N)));
        float3 B = normalize(cross(N, T));

        float3 wi = T * sp.x + N * sp.y + B * sp.z;

        float pdf = abs(dot(N, wi)) * M_1_PI;
        float3 f = M_1_PI * sr.material_ptr->get_cd();

        if (f == make_float3(0, 0, 0) || pdf == 0.f)
            break;

        float n_dot_wi = fmaxf(0.0, dot(sr.normal, wi));

        beta *= f * n_dot_wi / pdf;

        ray = Ray(sr.local_hit_point, wi);

        if (bounces > 3) {
            float q = fmaxf((float).05, 1 - beta.y);
            if (random() < q)
                break;
            beta /= 1 - q;
        }
        
        /*
        // Terminate path if ray escaped or maxDepth is reached
        if (!sr.hit_an_obj || bounces >= maxDepth)
            break;

        // Possibly add emitted light at intersection
        if (bounces == 0 && sr.material_ptr->is_emissive()) {
            L += sr.material_ptr->shade(sr);
        }
        else if (!sr.material_ptr->is_emissive()) {
            // Sample illumination from lights to find path contribution
            L += beta * sr.material_ptr->shade(sr);   
        }
        else break;
        
        // Sample BRDF to get new path direction
        float3 wo = -sr.ray.d, wi;
        float pdf;
        float3 f = fmaxf(make_float3(0,0,0), sr.material_ptr->sample_f(sr, wo, wi, pdf));

        if (f == make_float3(0,0,0) || pdf == 0.f)
            break;

        float n_dot_wi = fmaxf(0.0, dot(sr.normal, wi));

        beta *= f * n_dot_wi / pdf;

        ray = Ray(sr.local_hit_point, wi);

        if (bounces > 3) {
            float q = fmaxf((float).05, 1 - beta.y);
            if (random() < q)
                break;
            beta /= 1 - q;
        }
        */
    }
    return (L);
};

__global__ void render(uchar4* fb, float4* fb_accum, LinearBVHNode* nodes, unsigned int framenumber) {
    
    float3		pixel_color = make_float3(0, 0, 0);
    Ray			ray;
    float2		sp;				// sample point in [0, 1] x [0, 1]
    float2		pp;				// sample point on a pixel
    float2		dp;				// sample point on unit disk
    float2		lp;				// sample point on lens

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= g_viewplane_ptr->hres) || (j >= g_viewplane_ptr->vres)) return;

    int pixel_index = j * g_viewplane_ptr->hres + i;

    float lens_radius = g_camera_ptr->lens_radius;

    //for (int n = 0; n < g_viewplane_ptr->num_samples; n++) {
        sp = UniformSampleSquare();

        pp.x = g_viewplane_ptr->s * (i - 0.5 * g_viewplane_ptr->hres + sp.x);
        pp.y = g_viewplane_ptr->s * (j - 0.5 * g_viewplane_ptr->vres + sp.y);

        dp = ConcentricSampleDisk();
        lp = dp * lens_radius;

        ray.o = g_camera_ptr->position + lp.x * g_camera_ptr->right + lp.y * g_camera_ptr->up;
        ray.d = g_camera_ptr->ray_direction(pp, lp);

        pixel_color += trace_ray(ray, nodes);
        
    //}
    //pixel_color *= g_camera_ptr->exposure_time;
    //pixel_color /= framenumber;//(float)g_viewplane_ptr->num_samples;

    //pixel_color /= (pixel_color + 1.0f); // Hard coded Reinhard tone mapping

    //if (vp.gamma != 1.f)
        //    pixel_color = pow(pixel_color, vp.inv_gamma);

    fb_accum[pixel_index] += make_float4(pixel_color, 0);

    float4 final_col = fb_accum[pixel_index] / framenumber;
    final_col /= (final_col + 1.0f);

    fb[pixel_index] = to_uchar4(final_col * 255.0);
}

__global__ void init_render()
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        /*g_camera_ptr->position = make_float3(0, 200, 1000);
        g_camera_ptr->lookat = make_float3(0, 70, 0);
        g_camera_ptr->world_up = make_float3(0, 1, 0);
        g_camera_ptr->exposure_time = 1.f;
        g_camera_ptr->d = 100;
        g_camera_ptr->zoom = 30;
        g_camera_ptr->lens_radius = 120.f;
        g_camera_ptr->f = 900.f;
        g_camera_ptr->update_camera_vectors();

        printf("Camera Properties\n");
        printf("\t%-20s%-12.2f\n", "exposure time:", g_camera_ptr->exposure_time);
        printf("\t%-20s%-12.2f\n", "view distance:", g_camera_ptr->d);
        printf("\t%-20s%-12.2f\n", "zoom:", g_camera_ptr->zoom);
        printf("\t%-20s%-12.2f\n", "lens radius:", g_camera_ptr->lens_radius);
        printf("\t%-20s%-12.2f\n", "focal distance:", g_camera_ptr->f);

        printf("\t%-20s(%.2f, %.2f, %.2f)\n", "position:", g_camera_ptr->position.x, g_camera_ptr->position.y, g_camera_ptr->position.z);
        printf("\t%-20s(%.2f, %.2f, %.2f)\n", "lookat:", g_camera_ptr->lookat.x, g_camera_ptr->lookat.y, g_camera_ptr->lookat.z);
        printf("\t%-20s(%.2f, %.2f, %.2f)\n", "direction:", g_camera_ptr->direction.x, g_camera_ptr->direction.y, g_camera_ptr->direction.z);
        printf("\t%-20s(%.2f, %.2f, %.2f)\n", "right:", g_camera_ptr->right.x, g_camera_ptr->right.y, g_camera_ptr->right.z);
        printf("\t%-20s(%.2f, %.2f, %.2f)\n", "direction:", g_camera_ptr->up.x, g_camera_ptr->up.y, g_camera_ptr->up.z);
        */
        g_viewplane_ptr->gamma = 1.f;
        g_viewplane_ptr->hres = SCR_WIDTH;
        g_viewplane_ptr->vres = SCR_HEIGHT;
        g_viewplane_ptr->num_samples = 1;
        g_viewplane_ptr->s = 1.f / g_camera_ptr->zoom;

        printf("View Plane Properties\n");
        printf("\t%-20s%-12d\n", "hres:", g_viewplane_ptr->hres);
        printf("\t%-20s%-12d\n", "vres:", g_viewplane_ptr->vres);
        printf("\t%-20s%-12d\n", "samples:", g_viewplane_ptr->num_samples);
        printf("\t%-20s%-12.2f\n", "gamma:", g_viewplane_ptr->gamma);
        printf("\t%-20s%-12.2f\n", "pixel size:", g_viewplane_ptr->s);
        
        Emissive* emissive_ptr = new Emissive;
        emissive_ptr->scale_radiance(120);
        //emissive_ptr->scale_radiance(250);
        //emissive_ptr->set_ce(1, 1, 1);
        emissive_ptr->set_ce(0.96470, 0.80392, 0.54509);

        _Rectangle* rect_ptr = new _Rectangle(make_float3(-37.5, 386.74999, 300), make_float3(75, 0, 0), make_float3(0, 0, 75), make_float3(0, -1, 0));
        rect_ptr->set_material(emissive_ptr);
        rect_ptr->enable_shadows(false);

        AreaLight* area_light_ptr = new AreaLight;
        area_light_ptr->set_object(rect_ptr);
        area_light_ptr->set_material(emissive_ptr);
        area_light_ptr->enable_shadows(true);

        g_lights[0] = area_light_ptr;
//        g_num_lights = 1;
        
    }
}

__global__ void update_camera(
    float3  pos, 
    float   zoom, 
    float   lens_radius, 
    float   f, 
    float   d, 
    float   yaw, 
    float   pitch) 
{
    g_camera_ptr->position = pos;
    g_camera_ptr->world_up = make_float3(0,1,0);
    g_camera_ptr->d = d;
    g_camera_ptr->zoom = zoom;
    g_camera_ptr->lens_radius = lens_radius;
    g_camera_ptr->f = f;
    g_camera_ptr->yaw = yaw;
    g_camera_ptr->pitch = pitch;
    g_camera_ptr->update_camera_vectors();

    printf("PITCH:%f\tYAW:%f\n", pitch, yaw);
}

__global__ void wf_init(
    Paths* __restrict paths, 
    uint32_t pathCount) 
{
    uint32_t id = get_thread_id();

    if (id >= pathCount)
        return;

    paths->result[id] = make_float3(0.0f, 0.0f, 0.0f);
    paths->length[id] = 0;
    paths->extensionIntersection[id] = Isect();
}

__global__ void clearPathsKernel(Paths* paths, uint32_t pathCount)
{
    uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= pathCount)
        return;

    paths->length[id] = 0;
    paths->extensionIntersection[id] = Isect();
}

__global__ void wf_logic(
    Paths* __restrict paths, 
    Queues* __restrict queues,
    uchar4* fb,
    float4* fb_accum,
    uint32_t pathCount,
    uint32_t maxPathLength,
    unsigned int framenumber) {

    const uint32_t id = get_thread_id();

    if (id > pathCount)
        return;

    bool terminate = false;

    float3 beta = paths->throughput[id];
    uint32_t pathLength = paths->length[id];

    int x = int(paths->filmSamplePosition[id].x);
    int y = int(paths->filmSamplePosition[id].y);

    int pixel_index = y * g_viewplane_ptr->hres + x;
    
    if (pathLength >= maxPathLength || !paths->extensionIntersection[id].wasFound) {
        terminate = true;
        goto TERMINATE;
    }

    if (pathLength > 1) {

        if (!paths->lightRayBlocked[id]) {
            fb_accum[pixel_index] += make_float4(paths->lightBrdf[id] * beta, 0);
        }

        float pdf = paths->extensionBrdfPdf[id];
        float3 f = paths->extensionBrdf[id];
        float n_dot_wi = paths->extensionCosine[id];

        if (f == make_float3(0, 0, 0) || pdf == 0.f) {
            terminate = true;
            goto TERMINATE;
        }

        beta *= f * n_dot_wi / pdf;

        if (pathLength > 3) {
            float q = fmaxf((float).05, 1 - beta.y);
            if (random() < q)
                terminate = true;
            beta /= 1 - q;
        }
    }

    TERMINATE:
     
    if (terminate)
    {
        if (pathLength > 0) {
            fb_accum[pixel_index] += make_float4(paths->result[id], 0);

            float4 final_col = fb_accum[pixel_index] / framenumber;
            final_col /= (final_col + 1.0f);

            fb[pixel_index] = to_uchar4(final_col * 255.0);
        }

        // add path to newPath queue
        uint32_t queueIndex = atomicAggInc(&queues->newPathQueueLength);
        queues->newPathQueue[queueIndex] = id;
    }
    else // path continues
    {
        //float3 lightdir = normalize(make_float3(0, 300, 300) - paths->extensionIntersection[id].position);

        float3 lightdir, sample_point;
        g_lights[0]->get_direction(paths->extensionIntersection[id], lightdir, sample_point);

        Ray shadow_ray(paths->extensionIntersection[id].position, lightdir);
        paths->lightRay[id] = shadow_ray;
        paths->lightSamplePoint[id] = sample_point;

        // add path to diffuse material queue
        uint32_t queueIndex = atomicAggInc(&queues->diffuseMaterialQueueLength);
        queues->diffuseMaterialQueue[queueIndex] = id;

        paths->throughput[id] = beta;
    }
}

__global__ void wf_generate(
    Paths* __restrict paths,
    Queues* __restrict queues,
    uint32_t pathCount,
    uint32_t maxPathLength)
{
    uint32_t id = get_thread_id();

    if (id >= queues->newPathQueueLength)
        return;

    id = queues->newPathQueue[id];

    Ray			ray;
    float2		sp;				// sample point in [0, 1] x [0, 1]
    float2		pp;				// sample point on a pixel
    float2		dp;				// sample point on unit disk
    float2		lp;				// sample point on lens

    int x = id % g_viewplane_ptr->hres;
    int y = id / g_viewplane_ptr->hres;

    sp = UniformSampleSquare();

    pp.x = g_viewplane_ptr->s * (x - 0.5 * g_viewplane_ptr->hres + sp.x);
    pp.y = g_viewplane_ptr->s * (y - 0.5 * g_viewplane_ptr->vres + sp.y);

    dp = ConcentricSampleDisk();
    lp = dp * g_camera_ptr->lens_radius;

    ray.o = g_camera_ptr->position + lp.x * g_camera_ptr->right + lp.y * g_camera_ptr->up;
    ray.d = g_camera_ptr->ray_direction(pp, lp);

    paths->filmSamplePosition[id] = make_float2(x,y);
    paths->throughput[id] = make_float3(1.0f);
    paths->result[id] = make_float3(0.0f);
    paths->extensionRay[id] = ray;
    paths->length[id] = 0;

    uint32_t queueIndex = atomicAggInc(&queues->extensionRayQueueLength);
    queues->extensionRayQueue[queueIndex] = id;
}

__global__ void wf_extend(
    Paths* __restrict paths,
    Queues* __restrict queues, 
    LinearBVHNode* __restrict nodes) 
{
    uint32_t id = get_thread_id();

    if (id >= queues->extensionRayQueueLength)
        return;

    id = queues->extensionRayQueue[id];

    Ray ray = paths->extensionRay[id];
    Isect intersection;
    Intersect(ray, intersection, nodes);

    // make normal always to face the ray
    if (intersection.wasFound && dot(intersection.normal, ray.d) > 0.0f)
        intersection.normal = -intersection.normal;

    paths->length[id]++;
    paths->extensionIntersection[id] = intersection;
}

__global__ void wf_shadow(
    Paths* __restrict paths,
    Queues* __restrict queues,
    LinearBVHNode* __restrict nodes)
{
    uint32_t id = get_thread_id();

    if (id >= queues->lightRayQueueLength)
        return;

    id = queues->lightRayQueue[id];

    Ray ray = paths->lightRay[id];
    paths->lightRayBlocked[id] = g_lights[0]->in_shadow(ray, nodes);
}

__global__ void wf_shade(
    Paths* __restrict paths,
    Queues* __restrict queues) 
{
    uint32_t id = get_thread_id();

    if (id >= queues->diffuseMaterialQueueLength)
        return;

    id = queues->diffuseMaterialQueue[id];

    const Isect extensionIntersection = paths->extensionIntersection[id];
    //const Material extensionMaterial = materials[extensionIntersection.materialIndex];

    //////////////////////////////////////////////////////

    float e0 = random();
    float e1 = random();

    float sinTheta = sqrtf(1 - e0 * e0);
    float phi = 2 * M_PI * e1;
    float x = sinTheta * cosf(phi);
    float z = sinTheta * sinf(phi);
    float3 sp = make_float3(x, e0, z);

    float3 N = extensionIntersection.normal;
    float3 T = normalize(cross(N, get_orthogonal_vec(N)));
    float3 B = normalize(cross(N, T));

    float3 extensionDir = T * sp.x + N * sp.y + B * sp.z;

    //////////////////////////////////////////////////////

    paths->extensionBrdf[id] = make_float3(1) * M_1_PI;//material.cd * M_1_PI;
    paths->extensionBrdfPdf[id] = abs(dot(extensionIntersection.normal, extensionDir)) * M_1_PI;
    paths->extensionCosine[id] = fmaxf(0.0, dot(extensionIntersection.normal, extensionDir));

    float n_dot_wi = fmaxf(dot(paths->extensionIntersection[id].normal, paths->lightRay[id].d), 0.0);
    float3 f = make_float3(1) * M_1_PI * n_dot_wi;
    float3 Li = g_lights[0]->L(paths->extensionIntersection[id], paths->lightRay[id].d, paths->lightSamplePoint[id]);
    paths->lightBrdf[id] = f * Li;
    //paths->lightBrdfPdf[id] = getDiffusePdf(extensionIntersection.normal, paths->lightRay[id].direction);

    Ray extensionRay;
    extensionRay.o = extensionIntersection.position;
    extensionRay.d = extensionDir;

    paths->extensionRay[id] = extensionRay;

    uint32_t queueIndex = atomicAggInc(&queues->extensionRayQueueLength);
    queues->extensionRayQueue[queueIndex] = id;

    queueIndex = atomicAggInc(&queues->lightRayQueueLength);
    queues->lightRayQueue[queueIndex] = id;
}

__global__ void wf_drawbuff(
    Paths* __restrict paths, 
    uchar4* fb, 
    float4* fb_accum,
    uint32_t pixelCount,
    unsigned int framenumber)
{
    uint32_t id = get_thread_id();

    if (id >= pixelCount)
        return;

    int x = int(paths->filmSamplePosition[id].x);
    int y = int(paths->filmSamplePosition[id].y);

    int pixel_index = y * g_viewplane_ptr->hres + x;

    float4 pixel_col = fb_accum[pixel_index] / framenumber;
    pixel_col /= (pixel_col + 1.0f);
    fb[pixel_index] = to_uchar4(pixel_col * 255.0);
}

void display(void)
{
    if (buffer_reset) { 
        update_camera <<< 1, 1 >>> (cam_pos, cam_zoom, cam_lens_radius, cam_f, cam_d, cam_yaw, cam_pitch);
        cudaMemset(accumulatebuffer, 1, SCR_WIDTH * SCR_HEIGHT * sizeof(float4)); 
        framenumber = 0; 
    }

    buffer_reset = false;
    framenumber++;

    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &num_bytes, cuda_pbo_resource));

    int nx = SCR_WIDTH;
    int ny = SCR_HEIGHT;
    int tx = 16;
    int ty = 16;

    dim3 blocks(nx / tx, ny / ty, 1);
    dim3 threads(tx, ty, 1);

    if (g_bFirstTime) {
        g_bFirstTime = false;

        Camera* d_camera_ptr;
        checkCudaErrors(cudaMalloc((void**)&d_camera_ptr, sizeof(Camera)));
        checkCudaErrors(cudaMemcpyToSymbol(g_camera_ptr, &d_camera_ptr, sizeof(Camera*)));

        ViewPlane* d_viewplane_ptr;
        checkCudaErrors(cudaMalloc((void**)&d_viewplane_ptr, sizeof(ViewPlane)));
        checkCudaErrors(cudaMemcpyToSymbol(g_viewplane_ptr, &d_viewplane_ptr, sizeof(ViewPlane*)));

        Light** d_light_ptr;
        checkCudaErrors(cudaMalloc((void**)&d_light_ptr, sizeof(Light*)));
        checkCudaErrors(cudaMemcpyToSymbol(g_lights, &d_light_ptr, sizeof(Light**)));

        std::vector<Model*> models;
        models.push_back(new Model("E:/repos/CUDA-RayTracer/models/venere_test.obj"));

        int nmb_triangles;
        std::vector<BVHPrimitiveInfo> triangle_info;
        Triangle* d_triangles = loadModels(models, triangle_info, nmb_triangles);

        std::cerr << "Building BVH with " << nmb_triangles << " primitives. ";
        clock_t start, stop;
        start = clock();

        bvh = new BVHAccel(triangle_info, d_triangles, SplitMethod::SAH, 8);

        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n\n";

        update_camera << < 1, 1 >> > (cam_pos, cam_zoom, cam_lens_radius, cam_f, cam_d, cam_yaw, cam_pitch);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        init_render << < 1, 1 >> > ();
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    render <<< blocks, threads >>> (d_vbo, accumulatebuffer, bvh->d_nodes, framenumber);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    cudaThreadSynchronize();

    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glDrawPixels(SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();
}

void wf_display(void)
{
    int nx = SCR_WIDTH;
    int ny = SCR_HEIGHT;
    int tx = 16;
    int ty = 16;

    dim3 blocks(nx / tx, ny / ty, 1);
    dim3 threads(tx, ty, 1);

    int blockSize = 16;
    int gridSize = (pathCount + blockSize - 1) / blockSize;

    if (buffer_reset) {
        update_camera <<< 1, 1 >>> (cam_pos, cam_zoom, cam_lens_radius, cam_f, cam_d, cam_yaw, cam_pitch);
        //clearPathsKernel <<< gridSize, blockSize >>> (paths, pathCount);
        cudaMemset(accumulatebuffer, 1, SCR_WIDTH * SCR_HEIGHT * sizeof(float4));
        framenumber = 0;
    }

    buffer_reset = false;
    framenumber++;

    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &num_bytes, cuda_pbo_resource));

    if (g_bFirstTime) {
        g_bFirstTime = false;

        ViewPlane* d_viewplane_ptr;
        checkCudaErrors(cudaMallocManaged(&d_viewplane_ptr, sizeof(ViewPlane)), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMemcpyToSymbol(g_viewplane_ptr, &d_viewplane_ptr, sizeof(ViewPlane*)));

        Camera* d_camera_ptr;
        checkCudaErrors(cudaMallocManaged(&d_camera_ptr, sizeof(Camera)), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMemcpyToSymbol(g_camera_ptr, &d_camera_ptr, sizeof(Camera*)));

        Light** d_light_ptr;
        checkCudaErrors(cudaMalloc((void**)&d_light_ptr, sizeof(Light*)));
        checkCudaErrors(cudaMemcpyToSymbol(g_lights, &d_light_ptr, sizeof(Light**)));

        checkCudaErrors(cudaMallocManaged(&paths, sizeof(Paths)), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->filmSamplePosition, sizeof(float2) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->throughput, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->result, sizeof(float3) * pathCount),      "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->length, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->extensionRay, sizeof(Ray) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->extensionIntersection, sizeof(Isect) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->extensionBrdf, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->extensionBrdfPdf, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->extensionCosine, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->lightRay, sizeof(Ray) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->lightEmittance, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->lightBrdf, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->lightSamplePoint, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->lightBrdfPdf, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->lightPdf, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->lightCosine, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&paths->lightRayBlocked, sizeof(bool) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&queues, sizeof(Queues)), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&queues->newPathQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&queues->diffuseMaterialQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&queues->extensionRayQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
        checkCudaErrors(cudaMallocManaged(&queues->lightRayQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");

        std::vector<Model*> models;
        models.push_back(new Model("E:/repos/CUDA-RayTracer/models/venere_test.obj"));

        int nmb_triangles;
        std::vector<BVHPrimitiveInfo> triangle_info;
        Triangle* d_triangles = loadModels(models, triangle_info, nmb_triangles);

        std::cerr << "Building BVH with " << nmb_triangles << " primitives. ";
        clock_t start, stop;
        start = clock();

        bvh = new BVHAccel(triangle_info, d_triangles, SplitMethod::SAH, 8);

        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n\n";
        
        update_camera <<< 1, 1 >>> (cam_pos, cam_zoom, cam_lens_radius, cam_f, cam_d, cam_yaw, cam_pitch);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        init_render <<< 1, 1 >>> ();
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        wf_init <<< gridSize, blockSize >>> (paths, pathCount);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    wf_logic <<< gridSize, blockSize >>> (paths, queues, d_vbo, accumulatebuffer, pathCount, maxPathLength, framenumber);
    wf_generate <<< gridSize, blockSize >>> (paths, queues, pathCount, maxPathLength);
    wf_shade << < gridSize, blockSize >> > (paths, queues);
    wf_extend << < gridSize, blockSize >> > (paths, queues, bvh->d_nodes);
    wf_shadow <<< gridSize, blockSize >>> (paths, queues, bvh->d_nodes);

    //wf_drawbuff <<< gridSize, blockSize >>> (paths, d_vbo, accumulatebuffer, SCR_WIDTH * SCR_HEIGHT, framenumber);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    cudaThreadSynchronize();

    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glDrawPixels(SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();

    queues->newPathQueueLength = 0;
    queues->diffuseMaterialQueueLength = 0;
    queues->extensionRayQueueLength = 0;
    queues->lightRayQueueLength = 0;
}

void idle()
{
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

int lastX = 0, lastY = 0;
int theButtonState = 0;
int theModifierState = 0;

// camera mouse controls in X and Y direction
void motion(int x, int y)
{
    int deltaX = lastX - x;
    int deltaY = lastY - y;

    if (deltaX != 0 || deltaY != 0) {

        if (theButtonState == GLUT_LEFT_BUTTON)  // Rotate
        {
            //interactiveCamera->changeYaw(deltaX * 0.01);
            //interactiveCamera->changePitch(-deltaY * 0.01);
            cam_yaw += deltaX * 0.001;
            cam_yaw = fmod(cam_yaw, 2.f * M_PI);

            cam_pitch += deltaY * 0.001;
            //float padding = 0.05;
            //cam_pitch = clamp2(cam_pitch, -M_2_PI + padding, M_2_PI - padding)

        }
        else if (theButtonState == GLUT_MIDDLE_BUTTON) // Zoom
        {
            //interactiveCamera->changeAltitude(-deltaY * 0.01);
        }

        if (theButtonState == GLUT_RIGHT_BUTTON) // camera move
        {
            //interactiveCamera->changeRadius(-deltaY * 0.01);
        }

        lastX = x;
        lastY = y;
        buffer_reset = true;
        glutPostRedisplay();
    }
}

void mouse(int button, int state, int x, int y)
{
    theButtonState = button;
    theModifierState = glutGetModifiers();
    lastX = x;
    lastY = y;

    motion(x, y);
}

void keyboard(unsigned char key, int x, int y)
{
    float3 front;
    front.x = cos(cam_yaw) * cos(cam_pitch);
    front.y = sin(cam_pitch);
    front.z = sin(cam_yaw) * cos(cam_pitch);
    cam_dir = normalize(front);

    //direction = normalize(position - lookat);
    cam_right = normalize(cross(cam_dir, cam_worldUp));
    cam_up = normalize(cross(cam_right, cam_dir));
    //lookat = position + direction;

    switch (key)
    {
    case 27:
#if defined (__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif
        break;

    case 'w':
        cam_pos += cam_dir * cam_movement_spd;
        break;
    case 's':
        cam_pos -= cam_dir * cam_movement_spd;
        break;
    case 'a':
        cam_pos -= cam_right * cam_movement_spd;
        break;
    case 'd':
        cam_pos += cam_right * cam_movement_spd;
        break;
    case ' ':
        cam_pos += cam_up * cam_movement_spd;
        break;
    case 16: // shift
        cam_pos -= cam_up * cam_movement_spd;
        break;
    case 'q':
        cam_lens_radius -= 0.1;
        break;
    case 'e':
        cam_lens_radius += 0.1;
        break;
    case 'z':
        cam_f -= 1;
        break;
    case 'c':
        cam_f += 1;
        break;
    default:
        break;
    }
    buffer_reset = true;
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, 4096));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 5000 * 100000 * sizeof(Triangle*)));
    /*
    int nx = SCR_WIDTH;
    int ny = SCR_HEIGHT;
    int tx = 16;
    int ty = 16;

    dim3 blocks(nx / tx, ny / ty, 1);
    dim3 threads(tx, ty, 1);

    Camera* d_camera_ptr;
    checkCudaErrors(cudaMalloc((void**)&d_camera_ptr, sizeof(Camera)));
    checkCudaErrors(cudaMemcpyToSymbol(g_camera_ptr, &d_camera_ptr, sizeof(Camera*)));

    ViewPlane* d_viewplane_ptr;
    checkCudaErrors(cudaMalloc((void**)&d_viewplane_ptr, sizeof(ViewPlane)));
    checkCudaErrors(cudaMemcpyToSymbol(g_viewplane_ptr, &d_viewplane_ptr, sizeof(ViewPlane*)));

    Light** d_light_ptr;
    checkCudaErrors(cudaMalloc((void**)&d_light_ptr, sizeof(Light*)));
    checkCudaErrors(cudaMemcpyToSymbol(g_lights, &d_light_ptr, sizeof(Light**)));

    std::vector<Model*> models;
    models.push_back(new Model("E:/repos/CUDA-RayTracer/models/sphere_test.obj"));

    int nmb_triangles;
    std::vector<BVHPrimitiveInfo> triangle_info;
    Triangle* d_triangles = loadModels(models, triangle_info, nmb_triangles);

    std::cerr << "Building BVH with " << nmb_triangles << " primitives. ";
    clock_t start, stop;
    start = clock();

    BVHAccel* bvh = new BVHAccel(triangle_info, d_triangles, SplitMethod::SAH, 8);
    
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n\n";

    init_render <<< 1, 1 >>> ();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    */

    cudaMalloc(&accumulatebuffer, SCR_WIDTH * SCR_HEIGHT * sizeof(float4));

    // OPENGL stuff //

    ////////////////////////////////////////////////////////////////////////////////
    // 
    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(SCR_WIDTH, SCR_HEIGHT);
    glutCreateWindow("Path Tracer");
    //glutDisplayFunc(display);
    glutDisplayFunc(wf_display);

    glewInit();
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    //if (!isGLVersionSupported(2,0) ||
    //    !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    //{
    //    fprintf(stderr, "Required OpenGL extensions are missing.");
     //   exit(EXIT_FAILURE);
    //}
    
    // create pixel buffer object
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, SCR_WIDTH * SCR_HEIGHT * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

    glutMainLoop();

    ////////////////////////////////////////////////////////////////////////////////
    /*
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(float3);

    // allocate FB
    float3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks. ";
    start = clock();
    // Render our buffer
    render <<< blocks, threads >>> (fb, bvh->d_nodes);
    //checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    uint8_t* data = new uint8_t[nx * ny * 3];

    // Output FB as Image
    int index = 0;

    // Output FB as Image
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].x);
            int ig = int(255.99 * fb[pixel_index].y);
            int ib = int(255.99 * fb[pixel_index].z);

            data[index++] = ir;
            data[index++] = ig;
            data[index++] = ib;
        }
    }
    checkCudaErrors(cudaFree(fb));
    save_image("E:/repos/Firefly-master/images/out.png", data);
    */
    return 0;
}