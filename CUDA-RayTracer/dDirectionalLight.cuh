#pragma once

#include "dLight.cuh"

class GeometricObj;
class Isect;
class dRay;

class dDirectionalLight : public dLight
{
public:
	__device__ dDirectionalLight(void);
	__device__ dDirectionalLight(float3 dir);
	__device__ virtual void get_direction(const Isect& isect, float3& wi, float3& sample_point);
	__device__ virtual float3 L(const Isect& isect);
	__device__ virtual float3 L(const Isect& isect, float3 wi, float3 sample_point);
	__device__ virtual float G(const Isect& isect) const;
	__device__ virtual bool visible(const dRay& ray) const;
	__device__ virtual bool visible(const dRay& ray, float& tmin, Isect& isect) const;
	__device__ virtual float get_pdf(const Isect& isect) const;
	__device__ virtual float get_pdf(const Isect& isect, const dRay& ray) const;
	__device__ virtual bool in_shadow(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const;
	__device__ void set_direction(const float x, const float y, const float z);
	__device__ void set_direction(const float3 dir);
private:
	float3 dir;
};