#ifndef _RAYTRACER_LIGHTS_AMBIENTOCCLUDER_H_
#define _RAYTRACER_LIGHTS_AMBIENTOCCLUDER_H_

#include "Light.h"

class AmbientOccluder : public Light {
public:

	__device__ AmbientOccluder(void) :
		Light(),
		min_amount(0.f)
	{
		ls = 2.f;
		color = glm::vec3(1.f);
		enable_shadows(true);
	}

	__device__ virtual vec3 get_direction(ShadeRec& sr)
	{

		glm::dvec3 dir = sr.normal;
		glm::dvec3 right = normalize(cross(dir, glm::dvec3(0.0072, 1.f, 0.0034)));
		glm::dvec3 up = normalize(cross(right, dir));

		glm::dvec3 sp = sampler_ptr->sample_hemisphere();
		return (sp.x * up + sp.y * right + sp.z * dir);
	};

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const
	{
		double t = K_HUGE;
		int num_objs = sr.s.objects.size();

		return shadow_hit(ray, t, sr.s.bvh, sr.s.objects);
	};

	__device__ virtual bool visible(const Ray& ray, double& tmin, ShadeRec& sr) const
	{
		return false;
	}

	__device__ virtual bool visible(const Ray& ray) const
	{
		return false;
	}

	__device__ virtual glm::vec3 L(ShadeRec& sr)
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
	glm::dvec3	world_up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3	min_amount;
};

#endif // _RAYTRACER_LIGHTS_AMBIENTOCCLUDER_H_
