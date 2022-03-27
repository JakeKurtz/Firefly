#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

class dRay;
class Isect;
class dMaterial;
class LinearBVHNode;
class dTriangle;

class dLight {

public:
	__device__ virtual void get_direction(const Isect& isect, float3& wi, float3& sample_point);
	__device__ virtual float3 L(const Isect& isect, float3 wi, float3 sample_point);
	__device__ virtual float3 L(const Isect& isect);
	__device__ virtual bool visible(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const = 0;
	__device__ virtual float get_pdf(const Isect& isect, const float3& wi) const;
	__device__ virtual float get_pdf(const Isect& isect) const;
	__device__ void set_color(const float x, const float y, const float z);
	__device__ void set_color(const float s);
	__device__ void set_color(const float3 col);
	__device__ void scale_radiance(const float _ls);
	__device__ bool is_delta();

	float		ls;
	float3		color;
	dMaterial*	material;

private:
	bool shadows;
protected:
	bool delta;
};