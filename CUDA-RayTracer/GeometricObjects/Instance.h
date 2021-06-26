#pragma once

#include "GeometricObj.h"
#include "../Utilities/Ray.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Instance : public GeometricObj
{
public:
	__device__ Instance(void) :
		obj_ptr(nullptr)
	{
		inv_matrix = glm::mat4(1.f);
		tran_matrix = glm::mat4(1.f);
	};

	__device__ Instance(GeometricObj* _obj_ptr) :
		obj_ptr(_obj_ptr)
	{
		inv_matrix = glm::mat4(1.f);
		tran_matrix = glm::mat4(1.f);
	};

	__device__ Instance* clone(void) const
	{
		return (new Instance(*this));
	}

	__device__ virtual bool hit(const Ray& ray, double& tmin, ShadeRec& sr) const
	{
		Ray inv_ray(ray);
		inv_ray.o = glm::dvec3(inv_matrix * glm::dvec4(inv_ray.o, 1.0));
		inv_ray.d = glm::dvec3(inv_matrix * glm::dvec4(inv_ray.d, 0.0));

		if (obj_ptr->hit(inv_ray, tmin, sr)) {
			sr.normal = normalize(glm::dvec3(transpose(inv_matrix) * glm::dvec4(sr.normal, 0.0)));

			if (obj_ptr->get_material())
				sr.local_hit_point = ray.o + tmin * ray.d;

			return (true);
		}

		return (false);
	};

	__device__ virtual bool hit(const Ray& ray) const
	{
		Ray inv_ray(ray);
		inv_ray.o = glm::dvec3(inv_matrix * glm::dvec4(inv_ray.o, 1.0));
		inv_ray.d = glm::dvec3(inv_matrix * glm::dvec4(inv_ray.d, 0.0));

		if (obj_ptr->hit(inv_ray)) {
			return (true);
		}

		return (false);
	};

	__device__ virtual bool shadow_hit(const Ray& ray, double& tmin) const
	{
		Ray inv_ray(ray);
		inv_ray.o = glm::dvec3(inv_matrix * glm::dvec4(inv_ray.o, 1.0));
		inv_ray.d = glm::dvec3(inv_matrix * glm::dvec4(inv_ray.d, 0.0));

		if (obj_ptr->shadow_hit(inv_ray, tmin)) {
			return (true);
		}

		return (false);
	};

	__device__ virtual void set_sampler(Sampler* sampler)
	{
		obj_ptr->set_sampler(sampler);
	};

	__device__ virtual glm::vec3 sample(void)
	{
		return obj_ptr->sample();
	};

	__device__ virtual float pdf(const ShadeRec& sr)
	{
		return obj_ptr->pdf(sr);
	};

	__device__ virtual glm::dvec3 get_normal(const glm::dvec3 p)
	{
		return obj_ptr->get_normal(p);
	};

	__device__ virtual Bounds3f get_bounding_box(void) 
	{
		return obj_ptr->get_bounding_box(tran_matrix);
	};

	__device__ void set_material(Material* _material_ptr)
	{
		obj_ptr->set_material(_material_ptr);
		material_ptr = _material_ptr;
	};

	__device__ void translate(const float x, const float y, const float z)
	{
		obj_translation = glm::vec3(-x, -y, -z);
	};

	__device__ void scale(const float x, const float y, const float z)
	{
		obj_scale = glm::vec3(1.f / x, 1.f / y, 1.f / z);
	};

	__device__ void scale(const float s)
	{
		obj_scale = glm::vec3(1.f / s);
	};

	__device__ void rotate(const float radians, const glm::vec3 axis)
	{
		obj_rot_radians = radians;
		obj_rot_axis = axis;
	};

	__device__ void update_inv_matrix(void)
	{
		inv_matrix = glm::mat4(1.f);
		inv_matrix = glm::scale((glm::mat4)inv_matrix, obj_scale);
		inv_matrix = glm::transpose(glm::rotate(inv_matrix, (double)obj_rot_radians, (glm::dvec3)obj_rot_axis));
		inv_matrix = glm::translate(inv_matrix, (glm::dvec3)obj_translation);

		tran_matrix = glm::inverse(inv_matrix);

		if (inv_matrix != glm::dmat4(1.f))
			transformed = true;
	};


private:
	GeometricObj*	obj_ptr;
	glm::dmat4		inv_matrix;
	glm::dmat4		tran_matrix;
	glm::vec3		obj_translation		= glm::vec3(0.f);
	glm::vec3		obj_scale			= glm::vec3(1.f);
	glm::vec3		obj_rot_axis		= glm::vec3(1.f);
	float			obj_rot_radians		= 0.f;
	bool			transform_texture;
};