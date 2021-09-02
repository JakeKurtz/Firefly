#pragma once
#include "CudaHelpers.cuh"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image_write.h>

#include "kernel.cuh"

#define checkHost(condition)   _checkHost(condition, #condition,__FILE__,__LINE__,__FUNCTION__)
/*
__global__ void
d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0 / float(imageW);
    float py = 1.0 / float(imageH);


    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        float4 color =
            (tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)) +
            (tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py));


        color /= 4.0;
        color *= 255.0;
        color = fminf(color, make_float4(255.0));

        surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
    }
}
*/
uint getMipMapLevels(cudaExtent size)
{
    size_t sz = MAX(MAX(size.width, size.height), size.depth);

    uint levels = 0;

    while (sz)
    {
        sz /= 2;
        levels++;
    }

    return levels;
}
/*
void generateMipMaps(cudaMipmappedArray_t mipmapArray, cudaExtent size)
{
    size_t width = size.width;
    size_t height = size.height;

#ifdef SHOW_MIPMAPS
    cudaArray_t levelFirst;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFirst, mipmapArray, 0));
#endif

    uint level = 0;

    while (width != 1 || height != 1)
    {
        width /= 2;
        width = MAX((size_t)1, width);
        height /= 2;
        height = MAX((size_t)1, height);

        cudaArray_t levelFrom;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
        cudaArray_t levelTo;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

        cudaExtent  levelToSize;
        checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
        checkHost(levelToSize.width == width);
        checkHost(levelToSize.height == height);
        checkHost(levelToSize.depth == 0);

        // generate texture object for reading
        cudaTextureObject_t         texInput;
        cudaResourceDesc            texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = levelFrom;

        cudaTextureDesc             texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;

        texDescr.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

        // generate surface object for writing

        cudaSurfaceObject_t surfOutput;
        cudaResourceDesc    surfRes;
        memset(&surfRes, 0, sizeof(cudaResourceDesc));
        surfRes.resType = cudaResourceTypeArray;
        surfRes.res.array.array = levelTo;

        checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));

        // run mipmap kernel
        dim3 blockSize(16, 16, 1);
        dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

        d_mipmap << <gridSize, blockSize >> > (surfOutput, texInput, (uint)width, (uint)height);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaDestroySurfaceObject(surfOutput));

        checkCudaErrors(cudaDestroyTextureObject(texInput));

#ifdef SHOW_MIPMAPS
        // we blit the current mipmap back into first level
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.dstArray = levelFirst;
        copyParams.srcArray = levelTo;
        copyParams.extent = make_cudaExtent(width, height, 1);
        copyParams.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
#endif

        level++;
    }
}
*/
class cudaTexture {
public:
    const char*             path;
    cudaArray_t             dataArray;
    cudaMipmappedArray_t    mipmapArray;
    cudaTextureObject_t     textureObject;
    void*                   h_data;
    int                     width, height, bpp;

    cudaTexture(const char* path, const string& directory, bool gamma = false);

    void initTexture(int width, int height);
    void loadImageData(const char* path);
};

cudaTexture::cudaTexture(const char* _path, const string& directory, bool gamma)
{
    path = _path;
    string filename = string(path);
    filename = directory + '/' + filename;

    loadImageData(filename.c_str());
};

void cudaTexture::initTexture(int imageWidth, int imageHeight)
{

    auto size = make_cudaExtent(imageWidth, imageHeight, 0);
    uint levels = getMipMapLevels(size);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaMallocMipmappedArray(&mipmapArray, &desc, size, levels));

    cudaArray_t level0;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(h_data, size.width * sizeof(uchar4), size.width, size.height);
    copyParams.dstArray = level0;
    copyParams.extent = size;
    copyParams.extent.depth = 1;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    // compute rest of mipmaps based on level 0
    //generateMipMaps(mipmapArray, size);

    // generate bindless texture object

    cudaResourceDesc            resDescr;
    memset(&resDescr, 0, sizeof(cudaResourceDesc));

    resDescr.resType = cudaResourceTypeMipmappedArray;
    resDescr.res.mipmap.mipmap = mipmapArray;

    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = 1;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.mipmapFilterMode = cudaFilterModeLinear;

    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeWrap;

    texDescr.maxMipmapLevelClamp = float(levels - 1);

    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(cudaCreateTextureObject(&textureObject, &resDescr, &texDescr, NULL));
}

void cudaTexture::loadImageData(const char* path)
{
    // load texture from disk
    h_data = stbi_load(path, &width, &height, &bpp, 0);

    // initialize texture
    initTexture(width, height);
}