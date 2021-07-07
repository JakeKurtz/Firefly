#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>

#define CUDA_VERSION 11030;
//#define GLM_FORCE_CUDA

#include "../middleware/glm/glm/glm.hpp"
#include "../middleware/stb/stb_image_write.h"

#include "Utilities/CudaHelpers.h"
#include "Utilities/Ray.h"
#include "Utilities/ShadeRec.h"
#include "Utilities/Random.h"

#include "Cameras/ThinLensCamera.h"
#include "Cameras/PinholeCamera.h"

#include "Materials/Material.h"
#include "Materials/CookTorrence.h"

#include "Lights/Light.h"
#include "Lights/AmbientLight.h"
#include "Lights/AmbientOccluder.h"
#include "Lights/PointLight.h"
#include "Lights/AreaLight.h"

#include "Scene/Scene.h"

#include "Tracers/Whitted.h"
#include "Tracers/PathTrace.h"
#include "Tracers/BranchPathTrace.h"

#include "GeometricObjects/GeometricObj.h"
#include "GeometricObjects/Instance.h"
#include "GeometricObjects/Sphere.h"
#include "GeometricObjects/Plane.h"
#include "GeometricObjects/Rectangle.h"
#include "GeometricObjects/Triangle.h"
#include "GeometricObjects/Ellipse.h"
#include "GeometricObjects/Torus.h"

#include "Acceleration/BVHAccel.h"

#include "GeometricObjects/Compound/Model.h"

using namespace glm;

const int SCR_WIDTH = 1024;
const int SCR_HEIGHT = 1024;

//using Rectangle;

__global__ void render_ThinLensCamera(float3* fb, Scene** scene_ptr, ThinLensCamera** camera_ptr) {
    float3		pixel_color = make_float3(0,0,0);
    ViewPlane   vp((*scene_ptr)->vp);
    Ray			ray;
    float2		sp;				// sample point in [0, 1] x [0, 1]
    float2		pp;				// sample point on a pixel
    float2		dp;				// sample point on unit disk
    float2		lp;				// sample point on lens
    
    vp.s = (*camera_ptr)->get_zoom();
    
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    //unsigned long long threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if ((i >= vp.hres) || (j >= vp.vres)) return;

    int pixel_index = j * vp.hres + i;
    
    float lens_radius = (*camera_ptr)->get_lens_radius();
    Sampler* sampler_ptr = (*camera_ptr)->get_sampler();

    
    for (int n = 0; n < 32; n++) {
        sp = UniformSampleSquare();
        //sp = vp.sampler_ptr->sample_unit_square();
     
        pp.x = vp.s * (i - 0.5 * vp.hres + sp.x);
        pp.y = vp.s * (j - 0.5 * vp.vres + sp.y);
        
        //dp = sampler_ptr->sample_unit_disk();
        dp = ConcentricSampleDisk();
        lp = dp * lens_radius;
   
        ray.o = (*camera_ptr)->position + lp.x * (*camera_ptr)->right + lp.y * (*camera_ptr)->up;
        ray.d = (*camera_ptr)->ray_direction(pp, lp);

        pixel_color += (*scene_ptr)->tracer_ptr->trace_ray(ray);
    }

    pixel_color /= 32;
    pixel_color *= (*camera_ptr)->exposure_time;

    pixel_color /= (pixel_color + 1.0f); // Hard coded Reinhard tone mapping

    if (vp.gamma != 1.f)
        pixel_color = pow(pixel_color, vp.inv_gamma);
    
    //unsigned int seed = (i * 1024 + j) * 100;
    
    //fb[pixel_index] = float3(wang_hash(seed) * (1.0 / 4294967296.0));//pixel_color;
    //fb[pixel_index] = float3(sampler_ptr->sample_unit_disk(), 0.f);
    fb[pixel_index] = pixel_color;
}

__global__ void create_ThinLensCamera(ThinLensCamera** camera_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*camera_ptr) = new ThinLensCamera(make_float3(0, 200, 1000), make_float3(0, 70, 0));
        //(*camera_ptr) = new ThinLensCamera(float3(150, 100, 370), float3(0, 20, 0));
        //(*camera_ptr) = new ThinLensCamera(float3(-300, 200, 370), float3(0, 15, 0));
        (*camera_ptr)->exposure_time = 1.f;
        (*camera_ptr)->set_view_distance(100);
        (*camera_ptr)->set_zoom(30);
        (*camera_ptr)->set_sampler(new MultiJittered(1));
        (*camera_ptr)->set_lens_radius(15.f);
        (*camera_ptr)->set_focal_distance(900.f);
        (*camera_ptr)->update_camera_vectors();
    }
}

__global__ void setup_random_states(curandState_t states[]) 
{
    int id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curand_init(clock(), id, 0, &states[id]);
}

__global__ void create_scene(Scene** scene_ptr, Triangle* triangles, int num_triangles, LinearBVHNode* __restrict__ nodes, int ordered_prims[], curandState_t states[])
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        (*scene_ptr) = new Scene;

        (*scene_ptr)->states = states;

        ViewPlane* vp = new ViewPlane;
        vp->set_hres(SCR_WIDTH);
        vp->set_vres(SCR_WIDTH);
        vp->set_samples(1);
        vp->set_pixel_size(1.f);
        vp->set_gamma(1.f);
        vp->set_max_depth(1);
        (*scene_ptr)->vp = (*vp);

        (*scene_ptr)->set_tracer(new PathTrace((*scene_ptr)));

        MultiJittered* sampler_ptr = new MultiJittered(1);
        sampler_ptr->generate_samples();
        sampler_ptr->map_to_hemisphere(1);
        
        AmbientLight* ambient_ptr = new AmbientLight();
        //ambient_ptr->set_sampler(sampler_ptr);
        (*scene_ptr)->ambient_ptr = ambient_ptr;
        
        int number_lights = 1;
        (*scene_ptr)->lights = CudaList<Light*>(number_lights);

        //PointLight* light_ptr = new PointLight();
        //light_ptr->set_position(100, 500, 500);
        //light_ptr->scale_radiance(10);
        //(*scene_ptr)->add_light(light_ptr);

        //int number_objects = 10;
        int number_objects = num_triangles;
        (*scene_ptr)->objects = CudaList<GeometricObj*>(number_objects);

        #pragma region Cornell Box

        // BOTTOM //
        
        CookTorrence* material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(1,1,1));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(0.f);
        material_ptr->set_ks(1.f);
        material_ptr->set_roughness(0.1);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        Plane* plane_ptr = new Plane(make_float3(0, -20, 0), make_float3(0.f, 1.f, 0.f));
        plane_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(plane_ptr);

        // RIGHT //

        material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(1.f, 0.2f, 0.2f));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(1.f);
        material_ptr->set_ks(0.f);
        material_ptr->set_roughness(100.f);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        Rectangle* rect_ptr = new Rectangle(make_float3(-125, -22, 125), make_float3(0, 0, -250), make_float3(0, 250, 0), make_float3(1, 0, 0));
        //rect_ptr->set_sampler(sampler_ptr);
        //rect_ptr->enable_shadows(false);
        rect_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(rect_ptr);
        /*
        Emissive* emissive_ptr = new Emissive;
        emissive_ptr->scale_radiance(20);
        emissive_ptr->set_ce(1, 0, 0);

        AreaLight* area_light_ptr = new AreaLight;
        area_light_ptr->set_object(rect_ptr);
        area_light_ptr->set_material(emissive_ptr);
        area_light_ptr->enable_shadows(true);

        (*scene_ptr)->add_light(area_light_ptr);
        */
        // LEFT //

        material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(0.2f, 0.2f, 1.f));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(1.f);
        material_ptr->set_ks(0.f);
        material_ptr->set_roughness(100.f);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        rect_ptr = new Rectangle(make_float3(125, -22, 125), make_float3(0, 0, -250), make_float3(0, 250, 0), make_float3(-1, 0, 0));
        //rect_ptr->set_sampler(sampler_ptr);
        //rect_ptr->enable_shadows(false);
        rect_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(rect_ptr);
        /*
        emissive_ptr = new Emissive;
        emissive_ptr->scale_radiance(20);
        emissive_ptr->set_ce(0,0,1);

        area_light_ptr = new AreaLight;
        area_light_ptr->set_object(rect_ptr);
        area_light_ptr->set_material(emissive_ptr);
        area_light_ptr->enable_shadows(true);

        (*scene_ptr)->add_light(area_light_ptr);
        */
        // TOP //

        material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(1,1,1));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(1.f);
        material_ptr->set_ks(0.f);
        material_ptr->set_roughness(100.f);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        rect_ptr = new Rectangle(make_float3(-125, 228, -125), make_float3(250, 0, 0), make_float3(0, 0, 250), make_float3(0, -1, 0));
        rect_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(rect_ptr);

        // BACK //
        
        material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(1,1,1));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(1.f);
        material_ptr->set_ks(0.f);
        material_ptr->set_roughness(100.f);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        rect_ptr = new Rectangle(make_float3(-125, -22, -125), make_float3(250, 0, 0), make_float3(0, 250, 0), make_float3(0, 0, 1));
        rect_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(rect_ptr);
        
        // Light //

        Emissive* emissive_ptr = new Emissive;
        emissive_ptr->scale_radiance(250);
        emissive_ptr->set_ce(1,1,1);
        //emissive_ptr->set_ce(0.96470, 0.80392, 0.54509);

        Sphere* blah = new Sphere(make_float3(0, 225, 0), 25);
        blah->set_material(emissive_ptr);
        blah->set_sampler(new MultiJittered(1));
        blah->enable_shadows(false);

        rect_ptr = new Rectangle(make_float3(-37.5, 386.74999, 0), make_float3(75, 0, 0), make_float3(0, 0, 75), make_float3(0, -1, 0));
        //rect_ptr = new Rectangle(dfloat3(-125, 227.9999, -125), dfloat3(250, 0, 0), dfloat3(0, 0, 250), dfloat3(0, -1, 0));
        rect_ptr->set_material(emissive_ptr);
        rect_ptr->set_sampler(new MultiJittered(1));
        rect_ptr->enable_shadows(false);

        AreaLight* area_light_ptr = new AreaLight;
        area_light_ptr->set_object(rect_ptr);
        area_light_ptr->set_material(emissive_ptr);
        area_light_ptr->enable_shadows(true);

        //(*scene_ptr)->add_obj(rect_ptr);
        (*scene_ptr)->add_light(area_light_ptr);

        material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(0,0,0));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(1.f);
        material_ptr->set_ks(1.f);
        material_ptr->set_roughness(0.1);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        Sphere* sphere_ptr = new Sphere(make_float3(0, 30, 0), 50.f);
        sphere_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(sphere_ptr);

        material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(1,0.2,0.2));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(1.f);
        material_ptr->set_ks(1.f);
        material_ptr->set_roughness(100.f);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        sphere_ptr = new Sphere(make_float3(50, 0, 100), 20.f);
        sphere_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(sphere_ptr);

        material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(0.2, 1, 0.2));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(1.f);
        material_ptr->set_ks(1.f);
        material_ptr->set_roughness(100.f);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        sphere_ptr = new Sphere(make_float3(0, 0, 100), 20.f);
        sphere_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(sphere_ptr);

        material_ptr = new CookTorrence();
        material_ptr->set_cd(make_float3(0.2,0.2,1));
        material_ptr->set_ka(0.05f);
        material_ptr->set_kd(1.f);
        material_ptr->set_ks(1.f);
        material_ptr->set_roughness(100.f);
        material_ptr->set_diffuse_sampler(sampler_ptr);

        sphere_ptr = new Sphere(make_float3(-50, 0, 100), 20.f);
        sphere_ptr->set_material(material_ptr);
        //(*scene_ptr)->add_obj(sphere_ptr);

        #pragma endregion

        for (int i = 0; i < num_triangles; i++) {
            int index = ordered_prims[i];
            (*scene_ptr)->objects[i] = &triangles[index];
        }

        (*scene_ptr)->bvh = nodes;
    }
}

/*__global__ void foobar()
{    
    curandState_t state;
    curand_init(clock(), 0, 0, &state);

    for (int i = 0; i < 32; i++) {
        randStates[i] = curand_uniform(&state) * UINT32_MAX;
    }
    __syncthreads();
}*/

void save_image(const char* filename, const void* data) {
    int stride = 3 * SCR_WIDTH;
    stbi_write_png(filename, SCR_WIDTH, SCR_HEIGHT, 3, data, stride);
}

int main()
{

    int nx = SCR_WIDTH;
    int ny = SCR_HEIGHT;
    int tx = 8;
    int ty = 8;

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    //size_t x = 0;
    //checkCudaErrors(cudaThreadGetLimit(&x, cudaLimitStackSize));
    checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, 20000));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 5000 * 100000 * sizeof(Triangle*)));

    // tuple< h_triangles, d_triangles, list_size >//
    //std::vector<tuple<GeometricObj**, GeometricObj**, int>> primitive_merge_list;

    //unsigned int h_randStates[32];
    //for (int i = 0; i < 32; i++) {
    //    h_randStates[i] = rand();
    //}

    //checkCudaErrors(cudaMalloc((void**)&h_randStates, sizeof(unsigned int) * 32));
    //checkCudaErrors(cudaMalloc((void**)&randStates, sizeof(unsigned int) * 32));

    //foobar <<< 1, 1, sizeof(unsigned int) * 32 >>> ();
    //checkCudaErrors(cudaGetLastError());
    //checkCudaErrors(cudaDeviceSynchronize());

    std::vector<Model*> models;

    //models.push_back(new Model("E:/repos/CUDA-RayTracer/models/plane.obj"));
    //models.push_back(new Model("E:/repos/CUDA-RayTracer/models/ico-sphere.obj"));
    models.push_back(new Model("E:/repos/CUDA-RayTracer/models/dragon_test.obj"));
    //models.push_back(new Model("E:/repos/CUDA-RayTracer/models/Nefertiti.obj"));

    int nmb_triangles;
    std::vector<BVHPrimitiveInfo> triangle_info;
    Triangle* d_triangles = loadModels(models, triangle_info, nmb_triangles);

    std::cerr << "Building BVH with " << nmb_triangles << " primitives. ";
    clock_t start, stop;
    start = clock();

    BVHAccel* bvh = new BVHAccel(triangle_info, SplitMethod::SAH, 8);

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n\n";
    
    ThinLensCamera** tlcam_ptr;
    checkCudaErrors(cudaMalloc((void**)&tlcam_ptr, sizeof(ThinLensCamera*)));

    create_ThinLensCamera <<< 1, 1 >>> (tlcam_ptr);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    curandState_t* states;
    checkCudaErrors(cudaMalloc((void**)&states, sizeof(curandState_t)));

    setup_random_states <<< blocks, threads >>> (states);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    Scene** scene;
    checkCudaErrors(cudaMalloc((void**)&scene, sizeof(Scene*)));
 
    create_scene <<< 1, 1 >>> (scene, d_triangles, nmb_triangles, bvh->d_nodes, bvh->d_orderedPrims, states);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks. ";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(float3);

    // allocate FB
    float3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Render our buffer

    start = clock();

    render_ThinLensCamera <<<blocks, threads>>> (fb, scene, tlcam_ptr);
    checkCudaErrors(cudaGetLastError());
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

    save_image("E:/repos/CUDA-RayTracer/images/out.png", data);
    
    return 0;
}
