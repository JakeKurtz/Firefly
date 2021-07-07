#ifndef _RAYTRACER_GEOMETRICOBJECTS_ELLIPSE_H_
#define _RAYTRACER_GEOMETRICOBJECTS_ELLIPSE_H_

#include "GeometricObj.h"
#include "../Samplers/Sampler.h"

class Ellipse : public GeometricObj
{
public:

	__device__ Ellipse(void)
	{
		c = make_float3(0, 0, 0);
		u = make_float3(0, 0, 1);
		v = make_float3(1, 0, 0);
		normal = make_float3(0, 1, 0);
		inv_area = 1.0 / (float)(M_PI * length(u) * length(v));
	};

	__device__ Ellipse(const float3 center, const float3 radii_1, const float3 radii_2) :
		c(center), u(radii_1), v(radii_2)
	{
		normal = cross(u, v);
		inv_area = 1.0 / (float)(M_PI * length(u) * length(v));
	};

	__device__ Ellipse(const float3 center, const float3 radius) :
		c(center), u(radius), v(radius)
	{
		normal = cross(u, v);
		inv_area = 1.0 / (float)(M_PI * length(u) * length(v));
	};

	__device__ Ellipse* clone(void) const
	{
		return (new Ellipse(*this));
	}

	__device__ virtual bool hit(const Ray& ray, float& tmin, ShadeRec& sr) const
	{
		float3 q = ray.o - c;
		float t = -dot(normal, q) / dot(ray.d, normal);

		if (t <= K_EPSILON)
			return (false);

		float r = dot(u, q + ray.d * t);
		float s = dot(v, q + ray.d * t);

		if (r * r + s * s > 1.0)
			return (false);

		tmin = t;
		sr.normal = normal;
		sr.local_hit_point = ray.o + t * ray.d;

		return (true);
	};

	__device__ virtual bool hit(const Ray& ray) const
	{
		float3 q = ray.o - c;
		float t = -dot(normal, q) / dot(ray.d, normal);

		if (t <= K_EPSILON)
			return (false);

		float r = dot(u, q + ray.d * t);
		float s = dot(v, q + ray.d * t);

		if (r * r + s * s > 1.0)
			return (false);

		return (true);
	};

	__device__ virtual bool shadow_hit(const Ray& ray, float& tmin) const
	{
		if (!shadows)
			return (false);

		float3 q = ray.o - c;
		float t = -dot(normal, q) / dot(ray.d, normal);

		if (t <= K_EPSILON)
			return (false);

		float r = dot(u, q + ray.d * t);
		float s = dot(v, q + ray.d * t);

		if (r * r + s * s > 1.0)
			return (false);

		tmin = t;

		return (true);
	};

	__device__ virtual void set_sampler(Sampler* sp)
	{
		if (sampler_ptr) {
			delete sampler_ptr;
			sampler_ptr = nullptr;
		}

		sampler_ptr = sp;
		sampler_ptr->generate_samples();
	};

	__device__ virtual float3 sample(void)
	{

	};

	__device__ virtual float3 get_normal(const float3 p)
	{
		return normal;
	};

private:
	float3		c, u, v;
	float3		normal;
	Sampler* sampler_ptr;
};
#endif // _RAYTRACER_GEOMETRICOBJECTS_ELLIPSE_H_