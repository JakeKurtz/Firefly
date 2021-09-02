#ifndef _RAYTRACER_LIGHT_LIGHT_H_
#define _RAYTRACER_LIGHT_LIGHT_H_

#include "ShadeRec.cuh"
#include "Ray.cuh"
#include "kernel.cuh"

class Light {

public:

	__device__ virtual void get_direction(const Isect& isect, float3& wi, float3& sample_point) {  };

	__device__ virtual float3 L(const Isect& isect, float3 wi, float3 sample_point) { return make_float3(1, 1, 1); };

	__device__ virtual float3 L(const Isect& isect) { return make_float3(1, 1, 1); };

	__device__ virtual bool in_shadow(const Ray& ray) const = 0;

	__device__ virtual bool visible(const Ray& ray, float& tmin, Isect& isect) const = 0;

	__device__ virtual bool visible(const Ray& ray) const = 0;

	__device__ virtual float G(const Isect& isect) const
	{
		return 1.0f;
	};

	__device__ virtual float get_pdf(const Isect& isect, const Ray& ray) const
	{
		return 1.0f;
	};

	__device__ virtual float get_pdf(const Isect& isect) const
	{
		return 1.0f;
	};

	__device__ void set_color(const float x, const float y, const float z)
	{
		color = make_float3(x, y, z);
	};

	__device__ void set_color(const float s)
	{
		color = make_float3(s);
	};

	__device__ void set_color(const float3 col)
	{
		color = col;
	};

	__device__ void scale_radiance(const float _ls)
	{
		ls = _ls;
	};

	__device__ bool casts_shadows(void)
	{
		return shadows;
	};

	__device__ void enable_shadows(bool b)
	{
		shadows = b;
	};

	__device__ virtual ~Light() {};

	float		ls;
	float3		color;
	Material* material_ptr;
	MaterialIndex	materialIndex = MaterialIndex::Emissive;

private:
	bool shadows;
};
#endif // _RAYTRACER_LIGHT_LIGHT_H_