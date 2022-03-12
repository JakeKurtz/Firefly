#pragma once 
#include <cuda_runtime.h>

#include "dMath.cuh"
#include "dRay.cuh"
#include "Isect.cuh"
#include "dMaterial.cuh"

class GeometricObj {

public:
	__device__ virtual GeometricObj* clone(void) const = 0;
	__device__ virtual bool hit(const dRay& ray, float& tmin, Isect& isect) const = 0;
	__device__ virtual bool hit(const dRay& ray) const = 0;
	__device__ virtual bool shadow_hit(const dRay& ray, float& tmin) const = 0;
	__device__ virtual float3 sample(void);
	__device__ virtual float3 get_normal(const float3 p);
	__device__ void set_color(float r, float g, float b);
	__device__ float3 get_color();
	__device__ dMaterial* get_material(void);
	__device__ void set_material(dMaterial* _material_ptr);
	__device__ void enable_shadows(bool b);
	__device__ virtual float pdf(const Isect& isect);
	__device__ void add_object(GeometricObj* object_ptr);

	dMaterial* material;

protected:
	float3				color;
	bool				shadows = true;
	bool				transformed = false;
	float				inv_area;
};