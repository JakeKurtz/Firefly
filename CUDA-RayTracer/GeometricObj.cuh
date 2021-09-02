#ifndef _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_
#define _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_

#include "ShadeRec.cuh"
#include "Math.h"

class GeometricObj {

public:
	__device__ virtual GeometricObj* clone(void) const = 0;

	__device__ virtual bool hit(const Ray& ray, float& tmin, Isect& isect) const = 0;

	__device__ virtual bool hit(const Ray& ray) const = 0;

	__device__ virtual bool shadow_hit(const Ray& ray, float& tmin) const = 0;					// intersects a shadow ray with the object

	__device__ virtual float3 sample(void)
	{
		return make_float3(0, 0, 0);
	};

	__device__ virtual float3 get_normal(const float3 p)
	{
		return make_float3(0, 0, 0);
	};

	__device__ void set_color(float r, float g, float b)
	{
		color = make_float3(r, g, b);
	};

	__device__ float3 get_color()
	{
		return color;
	};

	__device__ Material* get_material(void)
	{
		return material_ptr;
	};

	__device__ void set_material(Material* _material_ptr)
	{
		material_ptr = _material_ptr;
	};

	__device__ void enable_shadows(bool b)
	{
		shadows = b;
	};

	__device__ virtual float pdf(const Isect& isect)
	{
		return inv_area;
	};

	__device__ void add_object(GeometricObj* object_ptr) {}

	__device__ virtual ~GeometricObj() {};

	Material* material_ptr;
	MaterialIndex  materialIndex;

protected:
	float3				color;
	bool				shadows = true;
	bool				transformed = false;
	float				inv_area;
};


#endif // _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_