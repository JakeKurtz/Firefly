#ifndef _RAYTRACER_GEOMETRICOBJECTS_PLANE_H_
#define _RAYTRACER_GEOMETRICOBJECTS_PLANE_H_

#include "GeometricObj.h"
#include "../Samplers/Sampler.h"

class Plane : public GeometricObj
{
public:
	__device__ Plane(void)
	{
		point = glm::dvec3(0.f);
		normal = glm::dvec3(0.f, 1.f, 0.f);
		inv_area = 1.f;
	};

	__device__ Plane(const glm::dvec3 p, const glm::dvec3& n)
	{
		point = p;
		normal = n;
		inv_area = 1.f;
	};

	__device__ void set_normal(const glm::dvec3 n)
	{
		normal = n;
	};

	__device__ void set_point(const glm::dvec3 p)
	{
		point = p;
	};

	__device__ glm::dvec3 get_point(void)
	{
		return point;
	};

	__device__ Plane* clone(void) const
	{
		return (new Plane(*this));
	}

	__device__ virtual bool hit(const Ray& ray, double& tmin, ShadeRec& sr) const
	{
		double t = dot((point - ray.o), normal) / dot(ray.d, normal);

		if (t > K_EPSILON) {
			tmin = t;
			sr.normal = normal;
			sr.local_hit_point = ray.o + t * ray.d;
			return (true);
		}
		else
			return (false);
	};

	__device__ virtual bool hit(const Ray& ray) const {
		double t = dot((point - ray.o), normal) / dot(ray.d, normal);

		if (t > K_EPSILON) {
			return (true);
		}
		else
			return (false);
	}

	__device__ virtual bool shadow_hit(const Ray& ray, double& tmin) const
	{
		if (!shadows)
			return (false);

		double t = dot((point - ray.o), normal) / dot(ray.d, normal);

		if (t > K_EPSILON) {
			tmin = t;
			return (true);
		}
		else
			return (false);
	};

	__device__ virtual void set_sampler(Sampler* sampler) {};

	__device__ virtual glm::vec3 sample(void)
	{
		return glm::vec3(1);
	};

	__device__ virtual glm::dvec3 get_normal(const glm::dvec3 p)
	{
		return normal;
	};

private:
	glm::dvec3 point;					// point through which the plane passes
	glm::dvec3 normal;					// normal to the plane
};

#endif // _RAYTRACER_GEOMETRICOBJECTS_PLANE_H_
