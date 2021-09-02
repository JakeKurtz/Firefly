#include <cuda_runtime.h>
#include "Ray.cuh"
#include "Math.h"
#include "ShadeRec.cuh"
#include "BRDF.cuh"
#include "Material.cuh"

#ifndef _RAYTRACER_GEOMETRICOBJECTS_TRIANGLE_H_
#define _RAYTRACER_GEOMETRICOBJECTS_TRIANGLE_H_

struct Vertex {
	float3 Position;
	float3 Normal;
	float2 TexCoords;
	float3 Tangent;
	float3 Bitangent;
};

class Triangle
{
public:
	__device__ Triangle(void);

	__device__ Triangle(const Vertex v0, const Vertex v1, const Vertex v2);

	__device__ Triangle* clone(void) const;

	__device__ bool hit(const Ray& ray, float& tmin, Isect& isect) const;

	__device__ bool hit(const Ray& ray) const;

	__device__ bool shadow_hit(const Ray& ray, float& tmin) const;

	__device__ float3 get_normal(const float3 p);

	float inv_area;
	Material* material_ptr;
	Vertex v0, v1, v2;
	float3 face_normal;
};

__device__ Triangle::Triangle(void)
{
	v0.Position = make_float3(0, 0, 0);
	v1.Position = make_float3(0, 0, 1);
	v2.Position = make_float3(1, 0, 0);

	float3 v1v0 = v1.Position - v0.Position;
	float3 v2v0 = v2.Position - v0.Position;
	face_normal = normalize(cross(v1v0, v2v0));

	inv_area = 1.f / (float)(0.5 * (v0.Position.x * v1.Position.y + v1.Position.x * v2.Position.y + v2.Position.x * v0.Position.y - v1.Position.x * v0.Position.y - v2.Position.x * v1.Position.y - v0.Position.x * v2.Position.y));
};

__device__ Triangle::Triangle(const Vertex v0, const Vertex v1, const Vertex v2) :
	v0(v0), v1(v1), v2(v2)
{
	float3 v1v0 = v1.Position - v0.Position;
	float3 v2v0 = v2.Position - v0.Position;
	face_normal = normalize(cross(v1v0, v2v0));
	inv_area = 1.f / (float)(0.5 * (v0.Position.x * v1.Position.y + v1.Position.x * v2.Position.y + v2.Position.x * v0.Position.y - v1.Position.x * v0.Position.y - v2.Position.x * v1.Position.y - v0.Position.x * v2.Position.y));
};

__device__ Triangle* Triangle::clone(void) const
{
	return (new Triangle(*this));
}

__device__ bool Triangle::hit(const Ray& ray) const
{
	float3 v0v1 = v1.Position - v0.Position;
	float3 v0v2 = v2.Position - v0.Position;
	float3 pvec = cross(ray.d, v0v2);
	float det = dot(v0v1, pvec);

	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < K_EPSILON) return false;

	// ray and triangle are parallel if det is close to 0
	//if (fabs(det) < K_EPSILON) return false;

	float invDet = 1.f / (float)det;

	float3 tvec = ray.o - v0.Position;
	float u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	float3 qvec = cross(tvec, v0v1);
	float v = dot(ray.d, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	return true;
}

__device__ bool Triangle::hit(const Ray& ray, float& tmin, Isect& isect) const
{
	float3 v0v1 = v1.Position - v0.Position;
	float3 v0v2 = v2.Position - v0.Position;
	float3 pvec = cross(ray.d, v0v2);
	double det = dot(v0v1, pvec);

	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < K_EPSILON) return false;

	// ray and triangle are parallel if det is close to 0
	//if (fabs(det) < K_EPSILON) return false;

	double invDet = 1 / det;

	float3 tvec = ray.o - v0.Position;
	double u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	float3 qvec = cross(tvec, v0v1);
	double v = dot(ray.d, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	double t = dot(v0v2, qvec) * invDet;

	if (t < 0)
		return false;

	float3 normal = u * v1.Normal + v * v2.Normal + (1 - u - v) * v0.Normal;
	float2 texcoord = u * v1.TexCoords + v * v2.TexCoords + (1 - u - v) * v0.TexCoords;

	tmin = t;

	isect.normal = normal;
	isect.texcoord = texcoord;
	isect.position = ray.o + t * ray.d;

	return true;
};

__device__ bool Triangle::shadow_hit(const Ray& ray, float& tmin) const
{
	//if (!shadows)
	//	return (false);

	float3 v0v1 = v1.Position - v0.Position;
	float3 v0v2 = v2.Position - v0.Position;
	float3 pvec = cross(ray.d, v0v2);
	float det = dot(v0v1, pvec);

	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < K_EPSILON) return false;

	// ray and triangle are parallel if det is close to 0
	//if (fabs(det) < K_EPSILON) return false;

	float invDet = 1 / det;

	float3 tvec = ray.o - v0.Position;
	float u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	float3 qvec = cross(tvec, v0v1);
	float v = dot(ray.d, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	double t = dot(v0v2, qvec) * invDet;

	if (t < 0)
		return false;

	tmin = t;
	return true;
};

__device__ float3 Triangle::get_normal(const float3 p)
{
	return face_normal;
};

#endif // _RAYTRACER_GEOMETRICOBJECTS_TRIANGLE_H_
