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
		color = make_float3(1,1,1);
	}

	__device__ virtual float3 get_direction(ShadeRec& sr)
	{
		return (make_float3(0,0,0));
	};

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const 
	{
		return false;
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
		return (ls * color);
	};
};

#endif // _RAYTRACE_LIGHTS_AMBIENTLIGHT_H_