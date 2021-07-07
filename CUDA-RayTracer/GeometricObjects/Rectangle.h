#ifndef _RAYTRACER_GEOMETRICOBJECTS_RECTANGLE_H_
#define _RAYTRACER_GEOMETRICOBJECTS_RECTANGLE_H_

#include "GeometricObj.h"
#include "../Samplers/Sampler.h"

class Rectangle : public GeometricObj
{
public:

	__device__ Rectangle(void)
	{
		point = make_float3(0,0,0);
		a = make_float3(1, 0, 0);
		b = make_float3(0, 1, 0);
		normal = make_float3(0, 0, 1);
		a_len_squared = pow(length(a), 2);
		b_len_squared = pow(length(b), 2);
		inv_area = 1.f / (float)(length(a) * length(b));
	};

	__device__ Rectangle(float3 p0, float3 _a, float3 _b, float3 n)
	{
		point = p0;
		a = _a;
		b = _b;
		normal = n;
		a_len_squared = pow(length(a), 2);
		b_len_squared = pow(length(b), 2);
		inv_area = 1.f / (float)(length(a) * length(b));
	};

	__device__ Rectangle* clone(void) const
	{
		return (new Rectangle(*this));
	}

	__device__ virtual bool hit(const Ray& ray, float& tmin, ShadeRec& sr) const
	{
		float t = dot((point - ray.o), normal) / dot(ray.d, normal);

		float3 p = ray.o + t * ray.d;
		float3 p0p = p - point;

		if (t <= K_EPSILON)
			return (false);

		float p0p_dot_a = dot(p0p, a);

		if (p0p_dot_a < 0.0 || p0p_dot_a > a_len_squared)
			return (false);

		float p0p_dot_b = dot(p0p, b);

		if (p0p_dot_b < 0.0 || p0p_dot_b > b_len_squared)
			return (false);

		tmin = t;
		sr.normal = normal;
		sr.local_hit_point = p;

		return (true);
	};

	__device__ virtual bool hit(const Ray& ray) const
	{
		float t = dot((point - ray.o), normal) / dot(ray.d, normal);

		float3 p = ray.o + t * ray.d;
		float3 p0p = p - point;

		if (t <= K_EPSILON)
			return (false);

		float p0p_dot_a = dot(p0p, a);

		if (p0p_dot_a < 0.0 || p0p_dot_a > a_len_squared)
			return (false);

		float p0p_dot_b = dot(p0p, b);

		if (p0p_dot_b < 0.0 || p0p_dot_b > b_len_squared)
			return (false);

		return (true);
	};

	__device__ virtual bool shadow_hit(const Ray& ray, float& tmin) const
	{
		if (!shadows)
			return (false);

		float t = dot((point - ray.o), normal) / dot(ray.d, normal);

		float3 p = ray.o + t * ray.d;
		float3 p0p = p - point;

		if (t <= K_EPSILON)
			return (false);

		float p0p_dot_a = dot(p0p, a);

		if (p0p_dot_a < 0.0 || p0p_dot_a > a_len_squared)
			return (false);

		float p0p_dot_b = dot(p0p, b);

		if (p0p_dot_b < 0.0 || p0p_dot_b > b_len_squared)
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
		float2 sample_point = UniformSampleSquare();
		//float2 sample_point = sampler_ptr->sample_unit_square();
		return (point + sample_point.x * a + sample_point.y * b);
	};

	__device__ virtual float3 get_normal(const float3 p)
	{
		return normal;
	};

private:
	float3		point;
	float3		a;
	float3		b;
	float3		normal;
	Sampler*	sampler_ptr;
	float		a_len_squared;
	float		b_len_squared;
};
#endif // _RAYTRACER_GEOMETRICOBJECTS_RECTANGLE_H_