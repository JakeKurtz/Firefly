#include "kernel.cuh"
#include "dTriangle.cuh"
#include "CudaHelpers.h"
#include "dFilm.cuh"
#include "dCamera.cuh"
#include "BVH.h"

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
    dTriangle triangles[]
    //dLight* lights[]
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
        float3 lightdir = make_float3(0.001, 0.98, 0.001);// , sample_point;
        //lights[0]->get_direction(isect, lightdir, sample_point);

        dRay shadow_ray = dRay(isect.position + isect.normal * 0.001, lightdir);

        //if (!lights[0]->in_shadow(shadow_ray)) {
            float diff = fmaxf(dot(isect.normal, lightdir), 0.f);
            color = (isect.material->baseColorFactor * diff * 1.f) * fmaxf(0.f, dot(isect.normal, lightdir)) * 50.f;// *lights[0]->material_ptr->radiance;
            //color = isect.normal * fmaxf(0.f, dot(isect.normal, lightdir));
        //}
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

__global__
void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t launchKernel(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

void debug_raytracer(dFilm* film, dScene* scene, cudaArray_const_t array, cudaEvent_t event, cudaStream_t stream) {

    cudaError_t cuda_err = cudaBindSurfaceToArray(surf, array);

    int blockSize = 256;
    int numBlocks = ((film->hres*film->vres) + blockSize - 1) / blockSize;

    debug_kernel <<< numBlocks, blockSize, 0, stream >>> (film, scene->get_camera(), scene->get_nodes(), scene->get_triangles());

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
