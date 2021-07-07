#ifndef _RAYTRACER_LIGHTS_AMBIENTOCCLUDER_H_
#define _RAYTRACER_LIGHTS_AMBIENTOCCLUDER_H_

#include "Light.h"

class AmbientOccluder : public Light {
public:

	__device__ AmbientOccluder(void) :
		Light()
	{
		ls = 2.f;
		min_amount = make_float3(0,0,0);
		color = make_float3(1,1,1);
		enable_shadows(true);
	}

	__device__ virtual float3 get_direction(ShadeRec& sr)
	{
		float3 dir = sr.normal;
		float3 right = normalize(cross(dir, make_float3(0.0072, 1.f, 0.0034)));
		float3 up = normalize(cross(right, dir));

		//float3 sp = sampler_ptr->sample_hemisphere();
		float3 sp = CosineSampleHemisphere();
		return (sp.x * up + sp.y * right + sp.z * dir);
	};

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const
	{
		float t = K_HUGE;
		int num_objs = sr.s.objects.size();

		return shadow_hit(ray, t, sr.s.bvh, sr.s.objects);
	};

	__device__ virtual bool visible(const Ray& ray, float& tmin, ShadeRec& sr) const
	{
		return false;
	}

	__device__ virtual bool visible(const Ray& ray) const
	{
		return false;
	}

	__device__ virtual float3 L(ShadeRec& sr)
	{
		Ray shadow_ray;
		shadow_ray.o = sr.local_hit_point;
		shadow_ray.d = get_direction(sr);

		if (in_shadow(shadow_ray, sr))
			return (min_amount * ls * color);
		else
			return (ls * color);
	};

	__device__ void set_min_amount(const float _ls);

private:
	float3	world_up = make_float3(0.0f, 1.0f, 0.0f);
	float3	min_amount;
};

#endif // _RAYTRACER_LIGHTS_AMBIENTOCCLUDER_H_
