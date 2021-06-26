#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

__device__ float random()
{
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	curandState_t state;
	curand_init((unsigned long long)clock() + threadId, 0, 0, &state);

	return (curand_uniform_double(&state));
};

__device__ float rand_float()
{
	return random();
};

__device__ float rand_float(int min, int max)
{
	return ((random() * (max - min)) + min);
};

__device__ int rand_int(int min, int max)
{
	return (int)((random() * (max - min)) + min);
};

__device__ int rand_int()
{
	return (int)(random() * INT_MAX);
};