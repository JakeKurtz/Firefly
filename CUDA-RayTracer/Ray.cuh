#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda_runtime.h>

struct Ray {
	float3 o;
	float3 d;

	__device__ Ray(void)
	{
		o = make_float3(0, 0, 0);
		d = make_float3(0, 0, 0);
	};

	__device__ Ray(const float3& origin, const float3& dir) {
		o = origin;
		d = dir;
	};

	__device__ Ray(const Ray& ray)
		: o(ray.o), d(ray.d)
	{};
};