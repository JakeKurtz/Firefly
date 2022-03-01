#include "dRay.cuh"

__device__ dRay::dRay(void)
{
	o = make_float3(0, 0, 0);
	d = make_float3(0, 0, 0);
};

__device__ dRay::dRay(const float3& origin, const float3& dir) {
	o = origin;
	d = dir;
};

__device__ dRay::dRay(const dRay& ray)
	: o(ray.o), d(ray.d)
{};