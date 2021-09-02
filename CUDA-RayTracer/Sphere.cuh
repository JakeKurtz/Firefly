#ifndef _RAYTRACER_GEOMETRICOBJECTS_SPHERE_H_
#define _RAYTRACER_GEOMETRICOBJECTS_SPHERE_H_

#include "GeometricObj.cuh"
#include "Random.cuh"

class Sphere : public GeometricObj
{
public:
	__device__ Sphere(void)
	{
		center = make_float3(0, 0, 0);
		radius = 0.f;
		inv_area = 0.f;
	};

	__device__ Sphere(const float3 c, const float& r) :
		center(c),
		radius(r)
	{
		inv_area = 1.f / (float)(4.f * M_PI * radius * radius);
	};

	__device__ Sphere* clone(void) const
	{
		return (new Sphere(*this));
	}

	__device__ void set_center(float3 c)
	{
		center = c;
	};

	__device__ void set_center(float x, float y, float z)
	{
		center = make_float3(x, y, z);
	};

	__device__ void set_center(float c)
	{
		center = make_float3(c);
	};

	__device__ void set_radius(float r)
	{
		radius = r;
	};

	__device__ virtual bool hit(const Ray& ray, float& tmin, Isect& isect) const;

	__device__ virtual bool hit(const Ray& ray) const;

	__device__ virtual bool shadow_hit(const Ray& ray, float& tmin) const;

	__device__ virtual float3 sample(void);

	__device__ virtual float3 get_normal(const float3 p);

private:
	float3	center;					// sphere center
	float	radius;					// sphere radius
};

__device__ bool Sphere::hit(const Ray& ray, float& tmin, Isect& isect) const
{
	float	t;
	float3	temp = ray.o - center;
	float	a = dot(ray.d, ray.d);
	float	b = 2.f * dot(temp, ray.d);
	float	c = dot(temp, temp) - radius * radius;
	float	d = b * b - 4.f * a * c;

	if (d < 0.0)
		return (false);
	else {
		float e = sqrt(d);
		float denom = 2.f * a;

		t = (-b - e) / denom; // smaller root.
		if (t > K_EPSILON) {
			tmin = t;
			isect.distance = t;
			isect.normal = (temp + t * ray.d) / radius;
			isect.position = ray.o + t * ray.d;
			isect.wasFound = true;
			return (true);
		}

		t = (-b + e) / denom; // larger root
		if (t > K_EPSILON) {
			tmin = t;
			isect.distance = t;
			isect.normal = (temp + t * ray.d) / radius;
			isect.position = ray.o + t * ray.d;
			isect.wasFound = true;
			return (true);
		}
	}

	return (false);
}

__device__ bool Sphere::hit(const Ray& ray) const {
	float	t;
	float3	temp = ray.o - center;
	float	a = dot(ray.d, ray.d);
	float	b = 2.f * dot(temp, ray.d);
	float	c = dot(temp, temp) - radius * radius;
	float	d = b * b - 4.f * a * c;

	if (d < 0.0)
		return (false);
	else {
		float e = sqrt(d);
		float denom = 2.f * a;

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

__device__ bool Sphere::shadow_hit(const Ray& ray, float& tmin) const
{

	if (!shadows)
		return (false);

	float	t;
	float3	temp = ray.o - center;
	float	a = dot(ray.d, ray.d);
	float	b = 2.f * dot(temp, ray.d);
	float	c = dot(temp, temp) - radius * radius;
	float	d = b * b - 4.f * a * c;

	if (d < 0.0)
		return (false);
	else {
		float e = sqrt(d);
		float denom = 2.f * a;

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

__device__ float3 Sphere::sample(void)
{
	float3 p = UniformSampleSphere();
	p = make_float3(p.x + center.x, p.y + center.y, p.z + center.z);
	return p;
}

__device__ float3 Sphere::get_normal(const float3 p)
{
	return (normalize(center - p));
}

#endif // _RAYTRACER_GEOMETRICOBJECTS_SPHERE_H_
