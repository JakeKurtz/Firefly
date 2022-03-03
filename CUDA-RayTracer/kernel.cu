#include "kernel.cuh"
#include "dTriangle.cuh"
#include "CudaHelpers.h"
#include "dFilm.cuh"
#include "dCamera.cuh"
#include "BVH.h"
#include "dDirectionalLight.cuh"

#include <surface_functions.h>
#include "dScene.h"

surface<void, cudaSurfaceType2D> surf;

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

__global__
void d_add_directional_lights(float3 dir[], dMaterial* materials[], int size, dLight** lights)
{
    for (int i = 0; i < size; i ++) {
        lights[i] = new dDirectionalLight(dir[i]);

        //new (&lights[i]) dDirectionalLight(dir[i]);
        lights[i]->material = materials[i];

        //float3 dir = lights[i]->get_direction();
        //printf("dir:\t%f, %f, %f\n\n", dir.x, dir.y, dir.z);
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
    Isect isect_d;
    intersect(nodes, triangles, ray, isect);

    float3 color = make_float3(0.f);

    if (isect.wasFound) {
        for (int i = 0; i < nmb_lights; i++) {
            float3 lightdir, sample_point;
            lights[i]->get_direction(isect, lightdir, sample_point);

            dRay shadow_ray = dRay(isect.position + isect.normal * 0.001, lightdir);

            if (!lights[i]->in_shadow(nodes, triangles, shadow_ray)) {
                float diff = fmaxf(dot(isect.normal, lightdir), 0.f);
                color += (isect.material->baseColorFactor * diff * 1.f) * fmaxf(0.f, dot(isect.normal, lightdir)) * lights[i]->L(isect);
                //color = isect.normal * fmaxf(0.f, dot(isect.normal, lightdir));
            }
        } 
    }

    color *= camera->exposure_time;
    color /= (color + 1.0f);

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