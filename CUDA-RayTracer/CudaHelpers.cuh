#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        //std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        //    file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ uint threadID(void)
{
    return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ uint get_thread_id(void)
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

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