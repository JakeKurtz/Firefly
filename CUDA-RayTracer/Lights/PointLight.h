#ifndef _RAYTRACER_LIGHTS_POINTLIGHT_H_
#define _RAYTRACER_LIGHTS_POINTLIGHT_H_

#include "Light.h"

class PointLight : public Light
{
public:
	__device__ PointLight(void) :
		Light()
	{
		ls = 1.f;
		position = make_float3(0,0,0);
		color = make_float3(1,1,1);
		enable_shadows(true);
	};

	__device__ PointLight(float3 pos, float3 col, float _ls) :
		Light()
	{
		ls = _ls;
		color = col;
		position = pos;
		enable_shadows(true);
	};

	__device__ virtual float3 get_direction(ShadeRec& sr)
	{
		return (normalize(position - sr.local_hit_point));
	};

	__device__ virtual float3 L(ShadeRec& sr)
	{
		return (ls * color);
	};

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const
	{
		float d = distance(position, ray.o);
		return shadow_hit(ray, d, sr.s.bvh, sr.s.objects);
	};

	__device__ virtual bool visible(const Ray& ray, float& tmin, ShadeRec& sr) const
	{
		return false;
	}

	__device__ virtual bool visible(const Ray& ray) const 
	{
		return false;
	}

	__device__ void set_position(const float x, const float y, const float z)
	{
		position = make_float3(x, y, z);
	};

	__device__ void set_position(const float3 pos) 
	{
		position = pos;
	};

private:
	float3		position;
};

#endif // _RAYTRACER_LIGHTS_POINTLIGHT_H_
