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

#include "Random.cuh"
#include "Emissive.cuh"
#include "AreaLight.cuh"
#include "Rectangle.cuh"
#include "Sphere.cuh"
#include "Material.cuh"

__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

__global__ void init_render()
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
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

        // LIGHT 1
        float3 kitchen_lightcol = make_float3(1.0000, 0.8392, 0.6666);
        float kitchen_lightpow = 125;

        Sphere* sphere_ptr = new Sphere(make_float3(-120, 60.5, -90), 4);

        Material* mat = new Material();
        mat->emissiveColor = kitchen_lightcol;
        mat->materialIndex = MaterialIndex::Emissive;
        mat->radiance = kitchen_lightpow;
        mat->emissive = true;

        AreaLight* area_light_ptr = new AreaLight;
        area_light_ptr->set_object(sphere_ptr);
        area_light_ptr->material_ptr = mat;
        
        g_lights[0] = area_light_ptr;

        // LIGHT 2
        sphere_ptr = new Sphere(make_float3(-60, 60.5, -90), 4);

        mat = new Material();
        mat->emissiveColor = kitchen_lightcol;
        mat->materialIndex = MaterialIndex::Emissive;
        mat->radiance = kitchen_lightpow;
        mat->emissive = true;

        area_light_ptr = new AreaLight;
        area_light_ptr->set_object(sphere_ptr);
        area_light_ptr->material_ptr = mat;

        g_lights[1] = area_light_ptr;

        // LIGHT 3
        sphere_ptr = new Sphere(make_float3(0, 60.5, -90), 4);

        mat = new Material();
        mat->emissiveColor = kitchen_lightcol;
        mat->materialIndex = MaterialIndex::Emissive;
        mat->radiance = kitchen_lightpow;
        mat->emissive = true;

        area_light_ptr = new AreaLight;
        area_light_ptr->set_object(sphere_ptr);
        area_light_ptr->material_ptr = mat;

        g_lights[2] = area_light_ptr;

        //LIGHT 4
        sphere_ptr = new Sphere(make_float3(60, 60.5, -90), 4);

        mat = new Material();
        mat->emissiveColor = kitchen_lightcol;
        mat->materialIndex = MaterialIndex::Emissive;
        mat->radiance = kitchen_lightpow;
        mat->emissive = true;

        area_light_ptr = new AreaLight;
        area_light_ptr->set_object(sphere_ptr);
        area_light_ptr->material_ptr = mat;

        g_lights[3] = area_light_ptr;
    }
}

__global__ void update_camera(
    float3  pos,
    float   zoom,
    float   lens_radius,
    float   f,
    float   d,
    float   exposure_time,
    float   yaw,
    float   pitch)
{
    g_camera_ptr->position = pos;
    g_camera_ptr->world_up = make_float3(0, 1, 0);
    g_camera_ptr->d = d;
    g_camera_ptr->zoom = zoom;
    g_camera_ptr->lens_radius = lens_radius;
    g_camera_ptr->f = f;
    g_camera_ptr->yaw = yaw;
    g_camera_ptr->pitch = pitch;
    g_camera_ptr->exposure_time = exposure_time;
    g_camera_ptr->update_camera_vectors();

    g_viewplane_ptr->s = 1.f / g_camera_ptr->zoom;

    printf("PITCH:%f\tYAW:%f\n", pitch, yaw);
    printf("POS:\t%f,%f,%f\n", pos.x, pos.y, pos.z);
}

__global__ void wf_init(
    Paths* __restrict paths,
    uint32_t PATHCOUNT)
{
    uint32_t id = get_thread_id();

    if (id >= PATHCOUNT)
        return;

    paths->length[id] = 0;
    paths->ext_isect[id] = Isect();
}

__global__ void wf_logic(
    Paths* __restrict paths,
    Queues* __restrict queues,
    uchar4* fb,
    float4* fb_accum,
    int* n_samples,
    uint32_t PATHCOUNT,
    uint32_t MAXPATHLENGTH,
    unsigned int framenumber) 
{
    const uint32_t id = get_thread_id();

    if (id > PATHCOUNT)
        return;

    bool terminate = false;

    float3 beta = paths->throughput[id];
    uint32_t pathLength = paths->length[id];

    Material* material_ptr = paths->ext_isect[id].material_ptr;

    int x = int(paths->film_pos[id].x);
    int y = int(paths->film_pos[id].y);

    int pixel_index = y * g_viewplane_ptr->hres + x;

    if (pathLength > MAXPATHLENGTH || !paths->ext_isect[id].wasFound) {
        terminate = true;
        goto TERMINATE;
    }

    if (paths->ext_isect[id].material_ptr->materialIndex == MaterialIndex::Emissive) {
        fb_accum[pixel_index] += make_float4(beta * emissive_L(material_ptr), 0);
        terminate = true;
        goto TERMINATE;
    }

    //if (pathLength == 1) {
    //    fb_accum[pixel_index] += make_float4(emissive_L(material_ptr), 0);
        //terminate = true;
        //goto TERMINATE;
    //}

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
        n_samples[pixel_index]++;
        if (pathLength > 0) {
            float4 final_col = fb_accum[pixel_index] / framenumber;//(float)n_samples[pixel_index];
            final_col *= g_camera_ptr->exposure_time;
            final_col /= (final_col + 1.0f);
            fb[pixel_index] = to_uchar4(final_col * 255.0);
        }

        // add path to newPath queue
        uint32_t queueIndex = atomicAggInc(&queues->queue_newPath_length);
        queues->queue_newPath[queueIndex] = id;
    }

    else // path continues
    {
        float3 lightdir, sample_point;
        uint32_t queueIndex;
        Ray shadow_ray;

        int light_id = rand_int(0, 3);
        paths->light_id[id] = light_id;

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
        paths->throughput[id] = beta;
    }
}

__global__ void wf_generate(
    Paths* __restrict paths,
    Queues* __restrict queues,
    uint32_t PATHCOUNT,
    uint32_t MAXPATHLENGTH)
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_newPath_length)
        return;

    id = queues->queue_newPath[id];

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

    paths->film_pos[id] = make_float2(x, y);
    paths->throughput[id] = make_float3(1.0f);
    paths->ext_ray[id] = ray;
    paths->length[id] = 0;

    uint32_t queueIndex = atomicAggInc(&queues->queue_extension_length);
    queues->queue_extension[queueIndex] = id;
}

__global__ void wf_extend(
    Paths* __restrict paths,
    Queues* __restrict queues)
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_extension_length)
        return;

    id = queues->queue_extension[id];

    Ray ray = paths->ext_ray[id];
    Isect isect;
    intersect(ray, isect, paths->ext_isect[id]);

    Isect isect_light;
    float tmin_light;

    int light_id = paths->light_id[id];

    if (g_lights[light_id]->visible(ray, tmin_light, isect_light) && isect_light.distance < isect.distance) {
        isect_light.material_ptr = g_lights[light_id]->material_ptr;
        paths->length[id]++;
        paths->ext_isect[id] = isect_light;
    }
    else {
        // make normal always to face the ray
        //if (isect.wasFound && dot(isect.normal, ray.d) > 0.0f)
            //isect.normal = -isect.normal;

        paths->length[id]++;
        paths->ext_isect[id] = isect;
    }
}

__global__ void wf_shadow(
    Paths* __restrict paths,
    Queues* __restrict queues)
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_shadow_length)
        return;

    id = queues->queue_shadow[id];

    Ray ray = paths->light_ray[id];
    int light_id = paths->light_id[id];
    paths->light_inshadow[id] = g_lights[light_id]->in_shadow(ray);
}

__global__ void wf_mat_diffuse(
    Paths* __restrict paths,
    Queues* __restrict queues)
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
    float3 ext_dir = diff_sample(isect);
    Ray ext_ray = Ray(isect.position, ext_dir);

    paths->ext_ray[id] = ext_ray;
    paths->light_brdf[id] = diff_L(isect, wi, wo, light_id, sample_point);
    paths->ext_brdf[id] = diff_f(isect, ext_dir, wo);
    paths->ext_pdf[id] = diff_get_pdf();
    paths->ext_cosine[id] = fmaxf(0.0, dot(isect.normal, ext_dir));

    uint32_t queueIndex = atomicAggInc(&queues->queue_extension_length);
    queues->queue_extension[queueIndex] = id;

    queueIndex = atomicAggInc(&queues->queue_shadow_length);
    queues->queue_shadow[queueIndex] = id;
}

__global__ void wf_mat_cook(
    Paths* __restrict paths,
    Queues* __restrict queues)
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_mat_cook_length)
        return;

    id = queues->queue_mat_cook[id];

    const Isect isect = paths->ext_isect[id];
    const Material* material = isect.material_ptr;
    const float3 wi = paths->light_ray[id].d;
    const float3 wo = -paths->ext_ray[id].d;
    const float3 sample_point = paths->light_samplePoint[id];
    int light_id = paths->light_id[id];
    const float3 ext_dir = ct_sample(isect, wo);
    Ray ext_ray = Ray(isect.position, ext_dir);

    paths->ext_ray[id] = ext_ray;
    paths->light_brdf[id] = ct_L(isect, wi, wo, light_id, sample_point, get_roughness(isect));
    paths->ext_brdf[id] = ct_f(isect, ext_dir, wo);
    paths->ext_pdf[id] = ct_get_pdf(isect.normal, ext_dir, wo, get_roughness(isect));
    paths->ext_cosine[id] = fmaxf(0.0, dot(isect.normal, ext_dir));

    uint32_t queueIndex = atomicAggInc(&queues->queue_extension_length);
    queues->queue_extension[queueIndex] = id;

    queueIndex = atomicAggInc(&queues->queue_shadow_length);
    queues->queue_shadow[queueIndex] = id;
}

__global__ void wf_mat_mix(
    Paths* __restrict paths,
    Queues* __restrict queues)
{
    uint32_t id = get_thread_id();

    if (id >= queues->queue_mat_mix_length)
        return;

    id = queues->queue_mat_mix[id];

    const Isect isect = paths->ext_isect[id];
    const Material* material = isect.material_ptr;
    const float3 wi = paths->light_ray[id].d;
    const float3 wo = -paths->ext_ray[id].d;
    const float3 sample_point = paths->light_samplePoint[id];
    int light_id = paths->light_id[id];

    float3 ext_dir;
    if (random() < 0.5) {
        ext_dir = ct_sample(isect, wo);
        Ray ext_ray = Ray(isect.position, ext_dir);

        paths->ext_ray[id] = ext_ray;
        paths->light_brdf[id] = ct_L(isect, wi, wo, light_id, sample_point, get_roughness(isect));
        paths->ext_pdf[id] = ct_get_pdf(isect.normal, ext_dir, wo, get_roughness(isect));
        paths->ext_brdf[id] = ct_f(isect, ext_dir, wo);
    }
    else {
        ext_dir = diff_sample(isect);
        Ray ext_ray = Ray(isect.position, ext_dir);

        paths->ext_ray[id] = ext_ray;
        paths->light_brdf[id] = diff_L(isect, wi, wo, light_id, sample_point);
        paths->ext_pdf[id] = diff_get_pdf();
        paths->ext_brdf[id] = diff_f(isect, ext_dir, wo);
    }

    paths->ext_cosine[id] = fmaxf(0.0, dot(isect.normal, ext_dir));
    //paths->ext_brdf[id] = mix_f(isect, ext_dir, wo);
    //paths->ext_pdf[id] = mix_get_pdf(isect.normal, ext_dir, wo, material);
    //paths->light_brdf[id] = mix_L(isect, wi, wo, sample_point, material->roughness);

    uint32_t queueIndex = atomicAggInc(&queues->queue_extension_length);
    queues->queue_extension[queueIndex] = id;

    queueIndex = atomicAggInc(&queues->queue_shadow_length);
    queues->queue_shadow[queueIndex] = id;
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

    int x = int(paths->film_pos[id].x);
    int y = int(paths->film_pos[id].y);

    int pixel_index = y * g_viewplane_ptr->hres + x;

    float4 pixel_col = fb_accum[pixel_index] / framenumber;
    pixel_col /= (pixel_col + 1.0f);
    fb[pixel_index] = to_uchar4(pixel_col * 255.0);
}

static void glfw_window_size_callback(GLFWwindow* window, int width, int height);
static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos);
void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

void setup_glfw() {
    if (!glfwInit()) exit(EXIT_FAILURE);
    if (atexit(glfwTerminate)) { glfwTerminate(); exit(EXIT_FAILURE); }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "gl-cuda-test", NULL, NULL);
    if (!window) exit(EXIT_FAILURE);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) exit(EXIT_FAILURE);

    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);
    glfwSetCursorPosCallback(window, glfw_mouse_callback);
    glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
}
void setup_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
}
void setup_cudainterop() {
    // CUDA with GL interop
    glGenBuffers(1, &pbo); // make & register PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLubyte) * SCR_WIDTH * SCR_HEIGHT, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
}

void setup_scene() 
{
    ViewPlane* d_viewplane_ptr;
    checkCudaErrors(cudaMallocManaged(&d_viewplane_ptr, sizeof(ViewPlane)), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMemcpyToSymbol(g_viewplane_ptr, &d_viewplane_ptr, sizeof(ViewPlane*)));

    Camera* d_camera_ptr;
    checkCudaErrors(cudaMallocManaged(&d_camera_ptr, sizeof(Camera)), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMemcpyToSymbol(g_camera_ptr, &d_camera_ptr, sizeof(Camera*)));

    Light** d_light_ptr;
    checkCudaErrors(cudaMalloc((void**)&d_light_ptr, sizeof(Light*) * 5));
    checkCudaErrors(cudaMemcpyToSymbol(g_lights, &d_light_ptr, sizeof(Light**)));

    std::vector<Model*> models;
    models.push_back(new Model("../kitchen_model/kitchen_small.obj"));

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

    update_camera <<< 1, 1 >>> (cam_pos, cam_zoom, cam_lens_radius, cam_f, cam_d, cam_exposure, cam_yaw, cam_pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    init_render <<< 1, 1 >>> ();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
void setup_paths() 
{
    checkCudaErrors(cudaMallocManaged(&paths, sizeof(Paths)), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->film_pos, sizeof(float2) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->throughput, sizeof(float3) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->length, sizeof(uint32_t) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->ext_ray, sizeof(Ray) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->ext_isect, sizeof(Isect) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->ext_brdf, sizeof(float3) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->ext_pdf, sizeof(float) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->ext_cosine, sizeof(float) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->ext_specular, sizeof(bool) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->light_ray, sizeof(Ray) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->light_id, sizeof(int) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->light_emittance, sizeof(float3) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->light_brdf, sizeof(float3) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->light_samplePoint, sizeof(float3) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->light_pdf, sizeof(float) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->light_cosine, sizeof(float) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&paths->light_inshadow, sizeof(bool) * PATHCOUNT), "Could not allocate CUDA device memory");

    wf_init <<< GRIDSIZE, BLOCKSIZE >>> (paths, PATHCOUNT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
void setup_queues() 
{
    checkCudaErrors(cudaMallocManaged(&queues, sizeof(Queues)), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&queues->queue_newPath, sizeof(uint32_t) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&queues->queue_mat_diffuse, sizeof(uint32_t) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&queues->queue_mat_cook, sizeof(uint32_t) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&queues->queue_mat_mix, sizeof(uint32_t) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&queues->queue_extension, sizeof(uint32_t) * PATHCOUNT), "Could not allocate CUDA device memory");
    checkCudaErrors(cudaMallocManaged(&queues->queue_shadow, sizeof(uint32_t) * PATHCOUNT), "Could not allocate CUDA device memory");
}

void reset_buffer()
{
    if (buffer_reset) {
        update_camera <<< 1, 1 >>> (cam_pos, cam_zoom, cam_lens_radius, cam_f, cam_d, cam_exposure, cam_yaw, cam_pitch);
        //wf_init << < GRIDSIZE, BLOCKSIZE >> > (paths, PATHCOUNT);
        //checkCudaErrors(cudaGetLastError());
        //checkCudaErrors(cudaDeviceSynchronize());
        cameraWasMoving = true;
    }

    if (cameraWasMoving) {
        cudaMemset(accumulatebuffer, 0, SCR_WIDTH * SCR_HEIGHT * sizeof(float4));
        cudaMemset(n_samples, 0, SCR_WIDTH * SCR_HEIGHT * sizeof(int));
        frame = 0;
        clear_counter--;
    }

    if (clear_counter == 0) {
        cameraWasMoving = false;
        clear_counter = 15;
    }

    buffer_reset = false;
}
void render_gui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    if (show_demo_window)
        ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
        static float f = 0.0f;
        static int counter = 0;
        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &show_another_window);
        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color
        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }

    // 3. Show another simple window.
    {
        ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
        if (ImGui::TreeNode("Camera Settings"))
        {
            ImGui::Text("Focal Distance");
            ImGui::DragFloat("##fd", &cam_f, 0.1f, 0.f, 1000.f, "%.2f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::Text("Apature Size");
            ImGui::DragFloat("##lr", &cam_lens_radius, 0.1f, 0.001f, 100.f, "%.4f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::Text("Zoom");
            ImGui::DragFloat("##zoom", &cam_zoom, 1.f, 1.f, 500.f, "%.2f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::Text("?");
            ImGui::DragFloat("##?", &cam_d, 1.f, 1.f, 500.f, "%.2f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::Text("Exposure");
            ImGui::DragFloat("##exposure", &cam_exposure, 0.01f, 0.1f, 5.f, "%.2f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::TreePop();
        }
        ImGui::End();
    }

    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
void render_kernel() 
{
    //clock_t start = clock();

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo, NULL));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, NULL, cuda_pbo));

    wf_logic <<< GRIDSIZE, BLOCKSIZE >>> (paths, queues, d_pbo, accumulatebuffer, n_samples, PATHCOUNT, MAXPATHLENGTH, frame);
    wf_generate <<< GRIDSIZE, BLOCKSIZE >>> (paths, queues, PATHCOUNT, MAXPATHLENGTH);
    wf_mat_diffuse <<< GRIDSIZE, BLOCKSIZE >>> (paths, queues);
    wf_mat_cook <<< GRIDSIZE, BLOCKSIZE >>> (paths, queues);
    wf_mat_mix <<< GRIDSIZE, BLOCKSIZE >>> (paths, queues);
    wf_extend <<< GRIDSIZE, BLOCKSIZE >>> (paths, queues);
    wf_shadow <<< GRIDSIZE, BLOCKSIZE >>> (paths, queues);
    //wf_extend << < GRIDSIZE, BLOCKSIZE >> > (paths, queues, triangles_SoA, nodes_SoA);
    //wf_shadow << < GRIDSIZE, BLOCKSIZE >> > (paths, queues, triangles_SoA, nodes_SoA);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo, NULL));

    //clock_t stop = clock();
    //double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    //std::cerr << "took " << timer_seconds << " seconds.\n";
}
void reset_queues() 
{
    queues->queue_newPath_length = 0;
    queues->queue_mat_diffuse_length = 0;
    queues->queue_mat_cook_length = 0;
    queues->queue_mat_mix_length = 0;
    queues->queue_extension_length = 0;
    queues->queue_shadow_length = 0;
}

void draw_buffer() 
{
    // bind the texture and PBO
    glBindTexture(GL_TEXTURE_2D, textureId);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // copy pixels from PBO to texture object
    // Use offset instead of ponter.
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, PIXEL_FORMAT, GL_UNSIGNED_BYTE, 0);

    // bind PBO to update pixel values
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[nextIndex]);

    // map the buffer object into client's memory
    // Note that glMapBuffer() causes sync issue.
    // If GPU is working with this buffer, glMapBuffer() will wait(stall)
    // for GPU to finish its job. To avoid waiting (stall), you can call
    // first glBufferData() with NULL pointer before glMapBuffer().
    // If you do that, the previous data in PBO will be discarded and
    // glMapBuffer() returns a new allocated pointer immediately
    // even if GPU is still working with the previous data.
    //glBufferData(GL_PIXEL_UNPACK_BUFFER, DATA_SIZE, 0, GL_STREAM_DRAW);
    //GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    //if (ptr)
    //{
        // update data directly on the mapped buffer
        //updatePixels(ptr, DATA_SIZE);
        //glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);  // release pointer to mapping buffer
    //}

    glDrawPixels(SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);   // THE MAGIC LINE #2

    // clear buffer
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    /*
    // save the initial ModelView matrix before modifying ModelView matrix
    glPushMatrix();

    // tramsform camera
    glTranslatef(0, 0, -2.f);
    glRotatef(0, 1, 0, 0);   // pitch
    glRotatef(0, 0, 1, 0);   // heading

    // draw a point with texture
    glBindTexture(GL_TEXTURE_2D, textureId);
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    glTexCoord2f(0.0f, 0.0f);   glVertex3f(-1.0f, -1.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);   glVertex3f(1.0f, -1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);   glVertex3f(1.0f, 1.0f, 0.0f);
    glTexCoord2f(0.0f, 1.0f);   glVertex3f(-1.0f, 1.0f, 0.0f);
    glEnd();

    // unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);
    glPopMatrix();
    */
}

static void glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
    // get context
    //struct pxl_interop* const interop = (pxl_interop*)glfwGetWindowUserPointer(window);

    //pxl_interop_size_set(interop, width, height);
}
static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    if (mouseDown && !(ImGui::GetIO().WantCaptureMouse)) {
        float MouseSensitivity = 0.001;

        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        cam_yaw -= xoffset;
        cam_pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        //if (constrainPitch)
        //{
        if (cam_pitch > 89.0f)
            cam_pitch = 89.0f;
        if (cam_pitch < -89.0f)
            cam_pitch = -89.0f;
        //}
        buffer_reset = true;

    }
}
void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouseDown = true;
            click = true;
        }
        else if (action == GLFW_RELEASE) {
            mouseDown = false;
        }
    }
    
}
static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    float3 front;
    front.x = cos(cam_yaw) * cos(cam_pitch);
    front.y = sin(cam_pitch);
    front.z = sin(cam_yaw) * cos(cam_pitch);
    
    cam_dir = normalize(front);
    cam_right = normalize(cross(cam_dir, cam_worldUp));
    cam_up = normalize(cross(cam_right, cam_dir));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos -= cam_dir * cam_movement_spd;
    if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos += cam_dir * cam_movement_spd;
    if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos -= cam_right * cam_movement_spd;
    if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos += cam_right * cam_movement_spd;
    if (key == GLFW_KEY_Q && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos += cam_up * cam_movement_spd;
    if (key == GLFW_KEY_E && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos -= cam_up * cam_movement_spd;
    buffer_reset = true;
}

int main(int argc, char* argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, 4096));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 5000 * 100000 * sizeof(Triangle*)));

    cudaMalloc(&accumulatebuffer, SCR_WIDTH * SCR_HEIGHT * sizeof(float4));
    cudaMalloc(&n_samples, SCR_WIDTH * SCR_HEIGHT * sizeof(int));

    /*
    // init 2 texture objects
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, IMAGE_WIDTH, IMAGE_HEIGHT, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, (GLvoid*)imageData);
    glBindTexture(GL_TEXTURE_2D, 0);
    */

    setup_glfw();
    setup_imgui();
    setup_cudainterop();

    setup_scene();
    setup_paths();
    setup_queues();

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        reset_buffer();
        frame++;

        render_kernel();
        draw_buffer();

        render_gui();
        reset_queues();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    cudaDeviceReset();

    // missing some clean up here
    exit(EXIT_SUCCESS);
}