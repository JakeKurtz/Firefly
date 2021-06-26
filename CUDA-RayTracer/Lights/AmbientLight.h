#ifndef _RAYTRACE_LIGHTS_AMBIENTLIGHT_H_
#define _RAYTRACE_LIGHTS_AMBIENTLIGHT_H_

#include "Light.h"

class AmbientLight : public Light
{
public:

	__device__ AmbientLight(void) :
		Light()
	{
		ls = 1.f;
		color = glm::vec3(1.f);
	}

	__device__ virtual glm::vec3 get_direction(ShadeRec& sr)
	{
		return (glm::vec3(0.f));
	};

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const 
	{
		return false;
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
		return (ls * color);
	};
};

#endif // _RAYTRACE_LIGHTS_AMBIENTLIGHT_H_