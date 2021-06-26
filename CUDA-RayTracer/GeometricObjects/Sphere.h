#ifndef _RAYTRACER_GEOMETRICOBJECTS_SPHERE_H_
#define _RAYTRACER_GEOMETRICOBJECTS_SPHERE_H_

#include "GeometricObj.h"
#include "../Samplers/Sampler.h"

class Sphere : public GeometricObj
{
public:
	__device__ Sphere(void) :
		center(0.f),
		radius(1.f)
	{
		inv_area = 1.f / (double)(4.f * M_PI * radius * radius);
	};

	__device__ Sphere(const glm::dvec3 c, const double& r) :
		center(c),
		radius(r)
	{
		inv_area = 1.f / (double)(4.f * M_PI * radius * radius);
	};

	__device__ Sphere* clone(void) const
	{
		return (new Sphere(*this));
	}

	__device__ void set_center(glm::dvec3 c)
	{
		center = c;
	};

	__device__ void set_center(double x, double y, double z)
	{
		center = glm::dvec3(x, y, z);
	};

	__device__ void set_center(double c)
	{
		center = glm::dvec3(c);
	};

	__device__ void set_radius(double r)
	{
		radius = r;
	};

	__device__ virtual bool hit(const Ray& ray, double& t, ShadeRec& s) const;

	__device__ virtual bool hit(const Ray& ray) const;

	__device__ virtual bool shadow_hit(const Ray& ray, double& tmin) const;

	__device__ virtual void set_sampler(Sampler* sampler);

	__device__ virtual glm::vec3 sample(void);

	__device__ virtual glm::dvec3 get_normal(const glm::dvec3 p);

private:
	glm::dvec3	center;					// sphere center
	double	radius;					// sphere radius
};

__device__ bool Sphere::hit(const Ray& ray, double& tmin, ShadeRec& sr) const
{
	double	t;
	glm::dvec3	temp = ray.o - center;
	double	a = dot(ray.d, ray.d);
	double	b = 2.f * dot(temp, ray.d);
	double	c = dot(temp, temp) - radius * radius;
	double	d = b * b - 4.f * a * c;

	if (d < 0.0)
		return (false);
	else {
		double e = sqrt(d);
		double denom = 2.f * a;

		t = (-b - e) / denom; // smaller root.
		if (t > K_EPSILON) {
			tmin = t;
			sr.normal = (temp + t * ray.d) / radius;
			sr.local_hit_point = ray.o + t * ray.d;
			return (true);
		}

		t = (-b + e) / denom; // larger root
		if (t > K_EPSILON) {
			tmin = t;
			sr.normal = (temp + t * ray.d) / radius;
			sr.local_hit_point = ray.o + t * ray.d;
			return (true);
		}
	}

	return (false);
}

__device__ bool Sphere::hit(const Ray& ray) const {
	double	t;
	glm::dvec3	temp = ray.o - center;
	double	a = dot(ray.d, ray.d);
	double	b = 2.f * dot(temp, ray.d);
	double	c = dot(temp, temp) - radius * radius;
	double	d = b * b - 4.f * a * c;

	if (d < 0.0)
		return (false);
	else {
		double e = sqrt(d);
		double denom = 2.f * a;

		t = (-b - e) / denom; // smaller root.
		if (t > K_EPSILON) {
			return (true);
		}

		t = (-b + e) / denom; // larger root
		if (t > K_EPSILON) {
			return (true);
		}
	}

	return (false);
}

__device__ bool Sphere::shadow_hit(const Ray& ray, double& tmin) const
{

	if (!shadows)
		return (false);

	double	t;
	glm::dvec3	temp = ray.o - center;
	double	a = dot(ray.d, ray.d);
	double	b = 2.f * dot(temp, ray.d);
	double	c = dot(temp, temp) - radius * radius;
	double	d = b * b - 4.f * a * c;

	if (d < 0.0)
		return (false);
	else {
		double e = sqrt(d);
		double denom = 2.f * a;

		t = (-b - e) / denom; // smaller root.
		if (t > K_EPSILON) {
			tmin = t;
			return (true);
		}

		t = (-b + e) / denom; // larger root
		if (t > K_EPSILON) {
			tmin = t;
			return (true);
		}
	}

	return (false);
}

__device__ void Sphere::set_sampler(Sampler* sp)
{
	if (sampler_ptr) {
		delete sampler_ptr;
		sampler_ptr = nullptr;
	}

	sampler_ptr = sp;
	sampler_ptr->generate_samples();
	sampler_ptr->map_to_sphere();
}

__device__ glm::vec3 Sphere::sample(void)
{
	glm::dvec3 p = sampler_ptr->sample_sphere();
	p = glm::dvec3(p.x + center.x, p.y + center.y, p.z + center.z);
	return p;
}

__device__ glm::dvec3 Sphere::get_normal(const glm::dvec3 p)
{
	return (normalize(center - p));
}

#endif // _RAYTRACER_GEOMETRICOBJECTS_SPHERE_H_
