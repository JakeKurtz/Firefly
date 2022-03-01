#pragma once

#include <cuda_runtime.h>
#include "dVertex.cuh"
#include "dRay.cuh"
#include "dMaterial.cuh"
#include "Isect.cuh"
#include "BVH.h"

struct dTriangle
{
public:
	__device__ dTriangle(void);
	__device__ dTriangle(const dVertex v0, const dVertex v1, const dVertex v2);
	__device__ void init();
	__device__ bool intersect(const dRay& ray, float& u, float& v, float& t) const;
	__device__ bool hit(const dRay& ray, float& tmin, Isect& isect) const;
	__device__ bool hit(const dRay& ray) const;
	__device__ bool shadow_hit(const dRay& ray, float& tmin) const;

	float inv_area;
	int material_index = 0;
	dMaterial* material;
	dVertex v0, v1, v2;
	float3 face_normal;
};

__device__ void intersect(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& __restrict ray, Isect& isect);