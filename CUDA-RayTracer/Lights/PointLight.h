#ifndef _RAYTRACER_LIGHTS_POINTLIGHT_H_
#define _RAYTRACER_LIGHTS_POINTLIGHT_H_

#include "Light.h"

class PointLight : public Light
{
public:
	__device__ PointLight(void) :
		Light(),
		position(0.0)
	{
		ls = 1.f;
		color = glm::vec3(1.f);
		enable_shadows(true);
	};

	__device__ PointLight(glm::vec3 pos, glm::vec3 col, float _ls) :
		Light()
	{
		ls = _ls;
		color = col;
		position = pos;
		enable_shadows(true);
	};

	__device__ virtual glm::vec3 get_direction(ShadeRec& sr)
	{
		return (normalize(position - sr.local_hit_point));
	};

	__device__ virtual glm::vec3 L(ShadeRec& sr)
	{
		return (ls * color);
	};

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const
	{
		double d = distance(position, ray.o);
		return shadow_hit(ray, d, sr.s.bvh, sr.s.objects);
	};

	__device__ virtual bool visible(const Ray& ray, double& tmin, ShadeRec& sr) const
	{
		return false;
	}

	__device__ virtual bool visible(const Ray& ray) const 
	{
		return false;
	}

	__device__ void set_position(const float x, const float y, const float z)
	{
		position = glm::vec3(x, y, z);
	};

	__device__ void set_position(const glm::vec3 pos) 
	{
		position = pos;
	};

private:
	glm::dvec3		position;
};

#endif // _RAYTRACER_LIGHTS_POINTLIGHT_H_
