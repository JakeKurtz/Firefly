#ifndef _RAYTRACE_LIGHTS_AREALIGHT_H_
#define _RAYTRACE_LIGHTS_AREALIGHT_H_

#include "Light.h"

#include "../Materials/Emissive.h"
#include "../GeometricObjects/GeometricObj.h"

class AreaLight : public Light
{
public:

	__device__ AreaLight(void) :
		Light(),
		object_ptr(nullptr),
		material_ptr(nullptr)
	{
		ls = 1.f;
		position = make_float3(0,0,0);
		color = make_float3(1,1,1);
		enable_shadows(true);
	}

	__device__ virtual void get_direction(ShadeRec& sr, float3& wi, float3& sample_point)
	{
		sample_point = object_ptr->sample();
		float3 light_normal = object_ptr->get_normal(sample_point);
		wi = normalize(sample_point - sr.local_hit_point);
	};

	__device__ virtual float3 L(ShadeRec& sr, float3 wi, float3 sample_point)
	{
		float3 light_normal = object_ptr->get_normal(sample_point);
		float n_dot_d = dot(-light_normal, wi);

		if (n_dot_d > 0.0)
			return (material_ptr->get_Le(sr));
		else
			return (make_float3(0,0,0));
	};

	__device__ virtual float G(const ShadeRec& sr) const
	{
		float3 sample_point = object_ptr->sample();
		float3 light_normal = object_ptr->get_normal(sample_point);
		float3 wi = normalize(sample_point - sr.local_hit_point);

		float n_dot_d = dot(-light_normal, wi);
		float d2 = pow(distance(sample_point, sr.local_hit_point), 2);

		return (n_dot_d / d2);
	};

	__device__ virtual bool visible(const Ray& ray) const 
	{
		return object_ptr->hit(ray);
	};

	__device__ virtual bool visible(const Ray& ray, float& tmin, ShadeRec& sr) const
	{
		return object_ptr->hit(ray, tmin, sr);
	}

	__device__ virtual float get_pdf(const ShadeRec& sr) const
	{
		float3 sample_point = object_ptr->sample();
		float3 light_normal = object_ptr->get_normal(sample_point);
		float3 wi = normalize(sample_point - sr.local_hit_point);

		float n_dot_d = abs(dot(light_normal, -wi));
		float d2 = pow(distance(sample_point, sr.local_hit_point), 2);

		return ((d2 / n_dot_d) * object_ptr->pdf(sr));
	};

	__device__ virtual float get_pdf(const ShadeRec& sr, const Ray& ray) const
	{
		float3 wi = normalize(ray.o - sr.local_hit_point);

		Ray visibility_ray(sr.local_hit_point, wi);
		ShadeRec foobar(sr.s);
		float tmin;
		if (!object_ptr->hit(visibility_ray, tmin, foobar)) return 0.f;

		float n_dot_d = abs(dot(sr.normal, -wi));
		float d2 = pow(distance(ray.o, sr.local_hit_point), 2);

		return ((d2 / n_dot_d) * object_ptr->pdf(sr));
	};

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const
	{
		float3 sample_point = object_ptr->sample();

		float t;
		float ts = dot((sample_point - ray.o), ray.d);
		
		return shadow_hit(ray, ts, sr.s.bvh, sr.s.objects);
	};

	__device__ void set_position(const float x, const float y, const float z)
	{
		position = make_float3(x, y, z);
	};

	__device__ void set_position(const float3 pos)
	{
		position = pos;
	};

	__device__ void set_object(GeometricObj* obj_ptr)
	{
		object_ptr = obj_ptr;
	};

	__device__ void set_material(Emissive* mat_ptr)
	{
		material_ptr = mat_ptr;
	};

private:
	float3			position;
	GeometricObj*		object_ptr;
	Emissive*			material_ptr;
	//float3 sample_point;
	//float3 light_normal;
	//float3 wi;
};

#endif // _RAYTRACE_LIGHTS_AREALIGHT_H_