#ifndef _RAYTRACER_LIGHT_LIGHT_H_
#define _RAYTRACER_LIGHT_LIGHT_H_

#include "../Utilities/ShadeRec.h"
#include "../Utilities/Ray.h"
#include "../Utilities/CudaList.h"

using namespace glm;

class Scene;

class Light {

public:

	__device__ virtual float3 get_direction(ShadeRec& sr) { return make_float3(0,0,0); };

	__device__ virtual void get_direction(ShadeRec& sr, float3& wi, float3& sample_point) {  };

	__device__ virtual float3 L(ShadeRec& sr, float3 wi, float3 sample_point) { return make_float3(1,1,1); };

	__device__ virtual float3 L(ShadeRec& sr) { return make_float3(1,1,1); };

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const = 0;

	__device__ virtual bool visible(const Ray& ray, float& tmin, ShadeRec& sr) const = 0;

	__device__ virtual bool visible(const Ray& ray) const = 0;

	__device__ virtual float G(const ShadeRec& sr) const 
	{
		return 1.0f;
	};

	__device__ virtual float get_pdf(const ShadeRec& sr, const Ray& ray) const
	{
		return 1.0f;
	};

	__device__ virtual float get_pdf(const ShadeRec& sr) const
	{
		return 1.0f;
	};

	__device__ void set_sampler(Sampler* s_ptr)
	{
		if (sampler_ptr) {
			delete sampler_ptr;
			sampler_ptr = nullptr;
		}

		sampler_ptr = s_ptr;

		sampler_ptr->generate_samples();
		sampler_ptr->map_to_hemisphere(1);
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

protected:
	float		ls;
	float3		color;
	Sampler*	sampler_ptr;

private:
	bool shadows;
};
#endif // _RAYTRACER_LIGHT_LIGHT_H_