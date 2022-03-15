#include "kernel.cuh"
#include "dTriangle.cuh"
#include "CudaHelpers.h"
#include "dFilm.cuh"
#include "dCamera.cuh"
#include "BVH.h"
#include "dDirectionalLight.cuh"
#include "dAreaLight.cuh"

#include "Rectangle.cuh"

#include <surface_functions.h>
#include "dScene.h"
#include "Wavefront.cuh"
#include "dRandom.cuh"

surface<void, cudaSurfaceType2D> surf;

__constant__ float SHADOW_OFFSET = 0.01;

union pxl_rgbx_24
{
    uint1       b32;

    struct {
        unsigned  r : 8;
        unsigned  g : 8;
        unsigned  b : 8;
        unsigned  na : 8;
    };
};

__device__ uint32_t get_thread_id() {return threadIdx.x + blockIdx.x * blockDim.x; }

__device__ inline uint32_t atomicAggInc(uint32_t* ctr)
{
    uint32_t mask = __ballot(1);
    uint32_t leader = __ffs(mask) - 1;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t res;

    if (laneid == leader)
        res = atomicAdd(ctr, __popc(mask));

    res = __shfl(res, leader);
    return res + __popc(mask & ((1 << laneid) - 1));
}

__global__
void d_add_directional_lights(float3 dir[], dMaterial* materials[], int size, dLight** lights)
{
    for (int i = 0; i < size; i ++) {
        lights[i] = new dDirectionalLight(dir[i]);
        lights[i]->material = materials[i];
    }
}

__global__
void d_add_area_lights(_Rectangle* objs[], dMaterial* materials[], int size, dLight** lights)
{
    for (int i = 0; i < size; i++) {
        dAreaLight* a_light = new dAreaLight();
        lights[i] = a_light;
        lights[i]->material = materials[i];
        _Rectangle* r = new _Rectangle(make_float3(-25.f, 100, -25.f), make_float3(50, 0.f, 0.f), make_float3(0.f, 0.f, 50), make_float3(0.f, -1.f, 0.f));
        a_light->set_object(r);
    }
}

__global__
void d_process_mesh(dVertex vertices[], unsigned int indicies[], dMaterial* materials[], int mat_index, int offset, int size, dTriangle* triangles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        int j = i * 3;

        unsigned int iv0 = indicies[j];
        unsigned int iv1 = indicies[j + 1];
        unsigned int iv2 = indicies[j + 2];

        dVertex v0 = vertices[iv0];
        dVertex v1 = vertices[iv1];
        dVertex v2 = vertices[iv2];

        new (&triangles[i + offset]) dTriangle(v0, v1, v2);
        triangles[i + offset].material = materials[mat_index];
    }
}

__global__ 
void d_reorder_primitives(dTriangle triangles[], dTriangle triangles_cpy[], int ordered_prims[], int size)
{
    for (int i = 0; i < size; i++) {
    	int index = ordered_prims[i];
        triangles[i] = triangles_cpy[index];
    }
}

__global__
void debug_kernel(
    dFilm* film,
    dCamera* camera,
    LinearBVHNode nodes[],
    dTriangle triangles[],
    dLight* lights[],
    int nmb_lights
)
{
    uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    int x = id % film->hres;
    int y = (id / film->hres) % film->vres;

    dRay ray = camera->gen_ray(film, make_float2(x, y));

    Isect isect;
    intersect(nodes, triangles, ray, isect);

    float3 color = make_float3(0.f);

    if (isect.wasFound) {
        for (int i = 0; i < nmb_lights; i++) {
            float3 lightdir, sample_point;
            lights[i]->get_direction(isect, lightdir, sample_point);

            dRay shadow_ray = dRay(isect.position + get_normal(isect) * SHADOW_OFFSET, lightdir);

            float diff = fmaxf(dot(get_normal(isect), lightdir), 0.f);

            if (diff >= 0.f) {
                if (!lights[i]->in_shadow(nodes, triangles, shadow_ray)) {
                    //color += (get_albedo(isect) * diff * 1.f) * lights[i]->L(isect);
                    //color = isect.normal * fmaxf(0.f, dot(isect.normal, lightdir));
                }
            }
        }
        color = get_normal(isect);
    }

    //color *= camera->exposure_time;
    //color /= (color + 1.0f);

    union pxl_rgbx_24 rgbx;

    rgbx.r = (int)255 * color.x;
    rgbx.g = (int)255 * color.y;
    rgbx.b = (int)255 * color.z;
    rgbx.na = 255;

    surf2Dwrite(
        rgbx.b32,
        surf,
        x * sizeof(rgbx),
        y,
        cudaBoundaryModeZero
    );
}

// WAVEFRONT //

__global__
void wf_init(
    Paths* __restrict paths,
    uint32_t PATHCOUNT)
{
    uint32_t id = get_thread_id();

    if (id >= PATHCOUNT)
        return;

    paths->length[id] = 0;
    paths->ext_isect[id] = Isect();
}


__global__
void wf_logic(
    Paths* __restrict paths,
    Queues* __restrict queues,
    float4* fb_accum,
    int* n_samples,
    uint32_t PATHCOUNT,
    uint32_t MAXPATHLENGTH,
    dCamera* camera,
    dFilm* film,
    dLight* lights[])
{

    const uint32_t id = get_thread_id();

    if (id > PATHCOUNT)
        return;

    float4 final_col = make_float4(0.f);
    bool terminate = false;

    float3 beta = paths->throughput[id];
    uint32_t pathLength = paths->length[id];

    dMaterial* material = paths->ext_isect[id].material;

    int x = int(paths->film_pos[id].x);
    int y = int(paths->film_pos[id].y);

    int pixel_index = y * film->vres + x;

    if (pathLength > MAXPATHLENGTH || !paths->ext_isect[id].wasFound) {
        terminate = true;
    }

    //if (!paths->ext_isect[id].wasFound) fb_accum[pixel_index] += make_float4(beta * make_float3(1.f), 0);

    if (pathLength == 1){
        for (int i = 0; i < 1; i++) {
            float t;
            Isect is;
            dRay r = paths->ext_ray[id];
            if (lights[i]->visible(r, t, is) && t < paths->ext_isect[id].distance) 
                fb_accum[pixel_index] += make_float4(beta * emissive_L(lights[i]->material), 0);
        }
    }

    if (pathLength > 1) {
        if (!paths->light_inshadow[id]) {
            fb_accum[pixel_index] += make_float4(paths->light_brdf[id] * beta, 0);
        }

        float pdf = paths->ext_pdf[id];
        float3 f = paths->ext_brdf[id];
        float n_dot_wi = paths->ext_cosine[id];

        if (f == make_float3(0, 0, 0) || pdf == 0.f) {
            terminate = true;
            goto TERMINATE;
        }

        beta *= f * n_dot_wi / pdf;

        if (pathLength > 3) {
            float q = fmaxf((float).05, 1 - beta.y);
            if (random() < q)
                terminate = true;
            goto TERMINATE;
            beta /= 1 - q;
        }
    }

TERMINATE:

    if (terminate)
    {
        if (pathLength > 0) n_samples[pixel_index]++;

        // add path to newPath queue
        uint32_t queueIndex = atomicAggInc(&queues->queue_newPath_length);
        queues->queue_newPath[queueIndex] = id;
    }
    else // path continues
    {
        float3 lightdir, sample_point;
        uint32_t queueIndex;
        dRay shadow_ray;

        //int light_id = 0;//rand_int(0, 0);
        paths->light_id[id] = 0;//light_id;

        /*
        switch (paths->ext_isect[id].material_ptr->materialIndex) {
        case MaterialIndex::Diffuse:
            g_lights[light_id]->get_direction(paths->ext_isect[id], lightdir, sample_point);

            shadow_ray = Ray(paths->ext_isect[id].position, lightdir);
            paths->light_ray[id] = shadow_ray;
            paths->light_samplePoint[id] = sample_point;

            queueIndex = atomicAggInc(&queues->queue_mat_diffuse_length);
            queues->queue_mat_diffuse[queueIndex] = id;
            break;
        case MaterialIndex::CookTor:
            g_lights[light_id]->get_direction(paths->ext_isect[id], lightdir, sample_point);

            shadow_ray = Ray(paths->ext_isect[id].position, lightdir);
            paths->light_ray[id] = shadow_ray;
            paths->light_samplePoint[id] = sample_point;

            queueIndex = atomicAggInc(&queues->queue_mat_cook_length);
            queues->queue_mat_cook[queueIndex] = id;
            break;
        case MaterialIndex::Mix:
            g_lights[light_id]->get_direction(paths->ext_isect[id], lightdir, sample_point);

            shadow_ray = Ray(paths->ext_isect[id].position, lightdir);
            paths->light_ray[id] = shadow_ray;
            paths->light_samplePoint[id] = sample_point;

            queueIndex = atomicAggInc(&queues->queue_mat_mix_length);
            queues->queue_mat_mix[queueIndex] = id;
            break;
        }
        */

        lights[0]->get_direction(paths->ext_isect[id], lightdir, sample_point);

        shadow_ray = dRay(paths->ext_isect[id].position + paths->ext_isect[id].normal * SHADOW_OFFSET, lightdir);
        paths->light_ray[id] = shadow_ray;
        paths->light_samplePoint[id] = sample_point;

        queueIndex = atomicAggInc(&queues->queue_mat_mix_length);
        queues->queue_mat_mix[queueIndex] = id;

        paths->throughput[id] = beta;
    }

    if (n_samples[pixel_index] >= 1) {
        final_col = fb_accum[pixel_index] / (float)n_samples[pixel_index];
        final_col *= camera->exposure_time;
        final_col /= (final_col + 1.0f);

        union pxl_rgbx_24 rgbx;

        rgbx.r = (int)255 * final_col.x;
        rgbx.g = (int)255 * final_col.y;
        rgbx.b = (int)255 * final_col.z;
        rgbx.na = 255;

        surf2Dwrite(
            rgbx.b32,
            surf,
            x * sizeof(rgbx),
            y,
            cudaBoundaryModeZero
        );
    }
}

__global__
void wf_generate(
    Paths* __restrict paths,
    Queues* __restrict queues,
    uint32_t PATHCOUNT,
    uint32_t MAXPATHLENGTH,
    dFilm* film,
    dCamera* camera)
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_newPath_length)
        return;

    id = queues->queue_newPath[id];

    int x = id % film->hres;
    int y = (id / film->hres) % film->vres;

    float2 film_pos = make_float2(x, y);

    paths->film_pos[id] = film_pos;
    paths->throughput[id] = make_float3(1.0f);
    paths->ext_ray[id] = camera->gen_ray(film, film_pos);
    paths->length[id] = 0;

    uint32_t queueIndex = atomicAggInc(&queues->queue_extension_length);
    queues->queue_extension[queueIndex] = id;
}

__global__
void wf_extend(
    Paths* __restrict paths,
    Queues* __restrict queues,
    LinearBVHNode nodes[],
    dTriangle triangles[])
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_extension_length)
        return;

    id = queues->queue_extension[id];

    dRay ray = paths->ext_ray[id];
    Isect isect;
    intersect(nodes, triangles, ray, isect);

    Isect isect_light;
    float tmin_light;

    //int light_id = 0;//paths->light_id[id];

    //if (lights[light_id]->visible(ray, tmin_light, isect_light) && isect_light.distance < isect.distance) {
    //    isect_light.material = lights[light_id]->material;
    //    paths->length[id]++;
    //    paths->ext_isect[id] = isect_light;
    //}
    //else {
        // make normal always to face the ray
        //if (isect.wasFound && dot(isect.normal, ray.d) < 0.0f)
        //    isect.normal = -isect.normal;

    paths->length[id]++;
    paths->ext_isect[id] = isect;
    //}
}

__global__
void wf_shadow(
    Paths* __restrict paths,
    Queues* __restrict queues,
    dLight* lights[], 
    LinearBVHNode nodes[],
    dTriangle triangles[])
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_shadow_length)
        return;

    id = queues->queue_shadow[id];

    dRay ray = paths->light_ray[id];
    int light_id = paths->light_id[id];
    paths->light_inshadow[id] = lights[light_id]->in_shadow(nodes, triangles, ray);
}

__global__
void wf_mat_diffuse(
    Paths* __restrict paths,
    Queues* __restrict queues,
    dLight* lights[])
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_mat_diffuse_length)
        return;

    id = queues->queue_mat_diffuse[id];

    const Isect isect = paths->ext_isect[id];
    const float3 wi = paths->light_ray[id].d;
    const float3 wo = -paths->ext_ray[id].d;
    const float3 sample_point = paths->light_samplePoint[id];
    int light_id = paths->light_id[id];
    const float3 ext_dir = diff_sample(isect);
    dRay ext_ray = dRay(isect.position, ext_dir);

    paths->ext_ray[id] = ext_ray;
    paths->light_brdf[id] = diff_L(lights, isect, wi, wo, light_id, sample_point);
    paths->ext_brdf[id] = diff_f(isect, ext_dir, wo);
    paths->ext_pdf[id] = diff_get_pdf();
    paths->ext_cosine[id] = fmaxf(0.0, dot(get_normal(isect), ext_dir));

    uint32_t queueIndex = atomicAggInc(&queues->queue_extension_length);
    queues->queue_extension[queueIndex] = id;

    queueIndex = atomicAggInc(&queues->queue_shadow_length);
    queues->queue_shadow[queueIndex] = id;
}

__global__
void wf_mat_cook(
    Paths* __restrict paths,
    Queues* __restrict queues,
    dLight* lights[])
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_mat_cook_length)
        return;

    id = queues->queue_mat_cook[id];

    const Isect isect = paths->ext_isect[id];
    const dMaterial* material = isect.material;
    const float3 wi = paths->light_ray[id].d;
    const float3 wo = -paths->ext_ray[id].d;
    const float3 sample_point = paths->light_samplePoint[id];
    int light_id = paths->light_id[id];
    const float3 ext_dir = spec_sample(isect, wo);
    dRay ext_ray = dRay(isect.position, ext_dir);

    paths->ext_ray[id] = ext_ray;
    paths->light_brdf[id] = spec_L(lights, isect, wi, wo, light_id, sample_point, get_roughness(isect));
    paths->ext_brdf[id] = spec_f(isect, ext_dir, wo);
    paths->ext_pdf[id] = spec_get_pdf(get_normal(isect), ext_dir, wo, get_roughness(isect));
    paths->ext_cosine[id] = fmaxf(0.0, dot(get_normal(isect), ext_dir));

    paths->ext_brdf_type[id] = BSDF::specularBounce;

    uint32_t queueIndex = atomicAggInc(&queues->queue_extension_length);
    queues->queue_extension[queueIndex] = id;

    queueIndex = atomicAggInc(&queues->queue_shadow_length);
    queues->queue_shadow[queueIndex] = id;
}

__global__
void wf_mat_mix(
    Paths* __restrict paths,
    Queues* __restrict queues,
    dLight* lights[])
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_mat_mix_length)
        return;

    id = queues->queue_mat_mix[id];

    const Isect isect = paths->ext_isect[id];
    const dMaterial* material = isect.material;
    const float3 wi = paths->light_ray[id].d;
    const float3 wo = -paths->ext_ray[id].d;
    const float3 sample_point = paths->light_samplePoint[id];
    int light_id = paths->light_id[id];

    float3 ext_dir;

    paths->light_brdf[id] = BRDF_L(lights, isect, wi, wo, light_id, sample_point, ext_dir);
    paths->ext_brdf[id] = BRDF_f(isect, ext_dir, wo);
    paths->ext_pdf[id] = BRDF_pdf(isect, ext_dir, wo);
    paths->ext_ray[id] = dRay(isect.position, ext_dir);
    paths->ext_cosine[id] = fmaxf(0.0, dot(get_normal(isect), ext_dir));

    uint32_t queueIndex = atomicAggInc(&queues->queue_extension_length);
    queues->queue_extension[queueIndex] = id;

    queueIndex = atomicAggInc(&queues->queue_shadow_length);
    queues->queue_shadow[queueIndex] = id;
}

// NOTE: this code is smelly! data clumps/long parameter list -> some kind of struct to contain pathtracing crap?
void wavefront_pathtrace(Paths* paths, Queues* queues, dFilm* film, float4* accumulatebuffer, int* n_samples, int path_count, int max_path_length, dScene* scene, cudaArray_const_t array, cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t cuda_err = cudaBindSurfaceToArray(surf, array);

    int blockSize = 64;
    int numBlocks = (path_count + blockSize - 1) / blockSize;

    wf_logic <<< numBlocks, blockSize, 0, stream >>> (paths, queues, accumulatebuffer, n_samples, path_count, max_path_length, scene->get_camera(), film, scene->get_lights());
    wf_generate <<< numBlocks, blockSize, 0, stream >>> (paths, queues, path_count, max_path_length, film, scene->get_camera());

    wf_mat_diffuse <<< numBlocks, blockSize, 0, stream >>> (paths, queues, scene->get_lights());
    wf_mat_cook <<< numBlocks, blockSize >>> (paths, queues, scene->get_lights());
    wf_mat_mix <<< numBlocks, blockSize >>> (paths, queues, scene->get_lights());

    wf_extend <<< numBlocks, blockSize, 0, stream >>> (paths, queues, scene->get_nodes(), scene->get_triangles());
    wf_shadow <<< numBlocks, blockSize, 0, stream >>> (paths, queues, scene->get_lights(), scene->get_nodes(), scene->get_triangles());

    checkCudaErrors(cudaDeviceSynchronize());
}

void wavefront_init(Paths* paths, int path_count)
{
    int blockSize = 256;
    int numBlocks = (path_count + blockSize - 1) / blockSize;

    wf_init <<< numBlocks, blockSize >>> (paths, path_count);
    checkCudaErrors(cudaGetLastError(), "CUDA ERROR: failed to initialize paths.");
    checkCudaErrors(cudaDeviceSynchronize());
}

void debug_raytracer(dFilm* film, dScene* scene, cudaArray_const_t array, cudaEvent_t event, cudaStream_t stream) {

    cudaError_t cuda_err = cudaBindSurfaceToArray(surf, array);

    int blockSize = 256;
    int numBlocks = ((film->hres*film->vres) + blockSize - 1) / blockSize;

    debug_kernel <<< numBlocks, blockSize, 0, stream >>> (film, scene->get_camera(), scene->get_nodes(), scene->get_triangles(), scene->get_lights(), scene->get_nmb_lights());

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void reorder_primitives(dTriangle triangles[], int ordered_prims[], int size) 
{
    dTriangle* triangles_cpy;
    size_t triangles_size = sizeof(dTriangle) * size;
    checkCudaErrors(cudaMalloc((void**)&triangles_cpy, triangles_size));
    checkCudaErrors(cudaMemcpy(triangles_cpy, triangles, triangles_size, cudaMemcpyHostToHost));

    d_reorder_primitives <<< 1, 1 >>> (triangles, triangles_cpy, ordered_prims, size);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void process_mesh(dVertex vertices[], unsigned int indicies[], dMaterial* materials[], int mat_index, int offset, int size, dTriangle* triangles) 
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    d_process_mesh <<< numBlocks, blockSize >>> (vertices, indicies, materials, mat_index, offset, size, triangles);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void add_directional_lights(float3* directions, dMaterial** materials, int size, dLight** d_lights) {

    d_add_directional_lights <<< 1, 1 >>> (directions, materials, size, d_lights);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void add_area_lights(_Rectangle** objs, dMaterial** materials, int size, dLight** d_lights) {

    d_add_area_lights << < 1, 1 >> > (objs, materials, size, d_lights);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}