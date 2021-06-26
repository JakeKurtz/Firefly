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
		position(0.f),
		object_ptr(nullptr),
		material_ptr(nullptr)
	{
		ls = 1.f;
		color = glm::vec3(1.f);
		enable_shadows(true);
	}

	__device__ virtual void get_direction(ShadeRec& sr, dvec3& wi, dvec3& sample_point)
	{
		sample_point = object_ptr->sample();
		glm::dvec3 light_normal = object_ptr->get_normal(sample_point);
		wi = normalize(sample_point - sr.local_hit_point);
	};

	__device__ virtual glm::vec3 L(ShadeRec& sr, dvec3 wi, dvec3 sample_point)
	{
		//glm::dvec3 sample_point = object_ptr->sample();
		glm::dvec3 light_normal = object_ptr->get_normal(sample_point);
		//glm::dvec3 wi = normalize(sample_point - sr.local_hit_point);

		float n_dot_d = dot(-light_normal, wi);

		if (n_dot_d > 0.0)
			return (material_ptr->get_Le(sr));
		else
			return (glm::vec3(0.f));
	};

	__device__ virtual float G(const ShadeRec& sr) const
	{
		glm::dvec3 sample_point = object_ptr->sample();
		glm::dvec3 light_normal = object_ptr->get_normal(sample_point);
		glm::dvec3 wi = normalize(sample_point - sr.local_hit_point);

		float n_dot_d = dot(-light_normal, wi);
		float d2 = pow(distance(sample_point, sr.local_hit_point), 2);

		return (n_dot_d / d2);
	};

	__device__ virtual bool visible(const Ray& ray) const 
	{
		return object_ptr->hit(ray);
	};

	__device__ virtual bool visible(const Ray& ray, double& tmin, ShadeRec& sr) const
	{
		return object_ptr->hit(ray, tmin, sr);
	}

	__device__ virtual float get_pdf(const ShadeRec& sr) const
	{
		glm::dvec3 sample_point = object_ptr->sample();
		glm::dvec3 light_normal = object_ptr->get_normal(sample_point);
		glm::dvec3 wi = normalize(sample_point - sr.local_hit_point);

		float n_dot_d = dot(-light_normal, wi);
		float d2 = pow(distance(sample_point, sr.local_hit_point), 2);

		return ((d2 / n_dot_d) * object_ptr->pdf(sr));
	};

	__device__ virtual float get_pdf(const ShadeRec& sr, const Ray& ray) const
	{
		glm::dvec3 wi = normalize(ray.o - sr.local_hit_point);

		float n_dot_d = dot(-sr.normal, wi);
		float d2 = pow(distance(ray.o, sr.local_hit_point), 2);

		return ((d2/n_dot_d) * object_ptr->pdf(sr));
	};

	__device__ virtual bool in_shadow(const Ray& ray, const ShadeRec& sr) const
	{
		glm::dvec3 sample_point = object_ptr->sample();

		double t;
		int num_objs = sr.s.objects.size();
		double ts = dot((sample_point - ray.o), ray.d);
		
		return shadow_hit(ray, ts, sr.s.bvh, sr.s.objects);
	};

	__device__ void set_position(const float x, const float y, const float z)
	{
		position = glm::vec3(x, y, z);
	};

	__device__ void set_position(const glm::vec3 pos)
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
	glm::dvec3			position;
	GeometricObj*		object_ptr;
	Emissive*			material_ptr;
	//glm::dvec3 sample_point;
	//glm::dvec3 light_normal;
	//glm::dvec3 wi;
};

#endif // _RAYTRACE_LIGHTS_AREALIGHT_H_