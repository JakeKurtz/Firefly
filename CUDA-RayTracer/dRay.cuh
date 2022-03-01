#pragma once

#include <cuda_runtime.h>

struct dRay {
	float3 o = make_float3(0.f, 0.f, 0.f);
	float3 d = make_float3(0.f, 1.f, 0.f);

	__device__ dRay(void);
	__device__ dRay(const float3& origin, const float3& dir);
	__device__ dRay(const dRay& ray);
};