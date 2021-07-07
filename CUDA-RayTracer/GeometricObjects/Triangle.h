#ifndef _RAYTRACER_GEOMETRICOBJECTS_TRIANGLE_H_
#define _RAYTRACER_GEOMETRICOBJECTS_TRIANGLE_H_

#include "GeometricObj.h"
#include "../Samplers/Sampler.h"

using namespace glm;

class Triangle : public GeometricObj
{
public:
	__host__ __device__ Triangle(void)
	{
		v0.Position = make_float3(0,0,0);
		v1.Position = make_float3(0,0,1);
		v2.Position = make_float3(1,0,0);

		float3 v1v0 = v1.Position - v0.Position;
		float3 v2v0 = v2.Position - v0.Position;
		face_normal = normalize(cross(v1v0, v2v0));

		inv_area = 1.f / (float)(0.5 * (v0.Position.x * v1.Position.y + v1.Position.x * v2.Position.y + v2.Position.x * v0.Position.y - v1.Position.x * v0.Position.y - v2.Position.x * v1.Position.y - v0.Position.x * v2.Position.y));
		bounds = Union(Bounds3f(v0.Position, v1.Position), (float3)v2.Position);;
	};

	__host__ __device__ Triangle(const Vertex v0, const Vertex v1, const Vertex v2) :
		v0(v0), v1(v1), v2(v2)
	{
		float3 v1v0 = v1.Position - v0.Position;
		float3 v2v0 = v2.Position - v0.Position;
		face_normal = normalize(cross(v1v0, v2v0));
		inv_area = 1.f / (float)(0.5 * (v0.Position.x * v1.Position.y + v1.Position.x * v2.Position.y + v2.Position.x * v0.Position.y - v1.Position.x * v0.Position.y - v2.Position.x * v1.Position.y - v0.Position.x * v2.Position.y));
		bounds = Union(Bounds3f(v0.Position, v1.Position), (float3)v2.Position);
	};

	__device__ Triangle* clone(void) const
	{
		return (new Triangle(*this));
	}

	__device__ virtual bool hit(const Ray& ray, float& tmin, ShadeRec& sr) const
	{
		float3 v0v1 = v1.Position - v0.Position;
		float3 v0v2 = v2.Position - v0.Position;
		float3 pvec = cross(ray.d, v0v2);
		float det = dot(v0v1, pvec);

		// if the determinant is negative the triangle is backfacing
		// if the determinant is close to 0, the ray misses the triangle
		if (det < K_EPSILON) return false;

		// ray and triangle are parallel if det is close to 0
		if (fabs(det) < K_EPSILON) return false;

		float invDet = 1 / det;

		float3 tvec = ray.o - v0.Position;
		float u = dot(tvec,pvec) * invDet;
		if (u < 0 || u > 1) return false;

		float3 qvec = cross(tvec,v0v1);
		float v = dot(ray.d,qvec) * invDet;
		if (v < 0 || u + v > 1) return false;

		float t = dot(v0v2, qvec) * invDet;

		float3 normal = u* v0.Normal + v * v1.Normal + (1 - u - v) * v2.Normal;

		tmin = t;
		sr.normal = normal;
		sr.local_hit_point = ray.o + t * ray.d;

		return true;
	};

	__device__ virtual bool hit(const Ray& ray) const
	{
		float3 v0v1 = v1.Position - v0.Position;
		float3 v0v2 = v2.Position - v0.Position;
		float3 pvec = cross(ray.d, v0v2);
		float det = dot(v0v1, pvec);

		// if the determinant is negative the triangle is backfacing
		// if the determinant is close to 0, the ray misses the triangle
		if (det < K_EPSILON) return false;

		// ray and triangle are parallel if det is close to 0
		if (fabs(det) < K_EPSILON) return false;

		float invDet = 1 / det;

		float3 tvec = ray.o - v0.Position;
		float u = dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1) return false;

		float3 qvec = cross(tvec, v0v1);
		float v = dot(ray.d, qvec) * invDet;
		if (v < 0 || u + v > 1) return false;

		return true;
	}

	__device__ virtual bool shadow_hit(const Ray& ray, float& tmin) const
	{
		if (!shadows)
			return (false);

		float3 v0v1 = v1.Position - v0.Position;
		float3 v0v2 = v2.Position - v0.Position;
		float3 pvec = cross(ray.d, v0v2);
		float det = dot(v0v1, pvec);

		// if the determinant is negative the triangle is backfacing
		// if the determinant is close to 0, the ray misses the triangle
		if (det < K_EPSILON) return false;

		// ray and triangle are parallel if det is close to 0
		if (fabs(det) < K_EPSILON) return false;

		float invDet = 1 / det;

		float3 tvec = ray.o - v0.Position;
		float u = dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1) return false;

		float3 qvec = cross(tvec, v0v1);
		float v = dot(ray.d, qvec) * invDet;
		if (v < 0 || u + v > 1) return false;

		tmin = dot(v0v2, qvec) * invDet;

		return true;
	};

	__device__ virtual void set_sampler(Sampler* sampler)
	{

	};

	__device__ virtual float3 sample(void)
	{
		return make_float3(1,1,1);
	};

	__device__ virtual float3 get_normal(const float3 p)
	{
		return face_normal;
	};

	//__device__ virtual Bounds3f get_bounding_box(void)
	//{
	//	return bounds;
	//};

	//__device__ virtual Bounds3f get_bounding_box(mat4 mat)
	//{
		//float3 _v0 = float3(mat * vec4(v0.Position,1));
		//float3 _v1 = float3(mat * vec4(v1.Position,1));
		//float3 _v2 = float3(mat * vec4(v2.Position,1));

		//return Union(Bounds3f(_v0, _v1), (float3)_v2);
	//};

private:
	Vertex v0, v1, v2;
	float3 face_normal;
	Bounds3f bounds;
};

#endif // _RAYTRACER_GEOMETRICOBJECTS_TRIANGLE_H_
