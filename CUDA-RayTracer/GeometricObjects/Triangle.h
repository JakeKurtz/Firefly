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
		v0.Position = dvec3(0,0,0);
		v1.Position = dvec3(0,0,1);
		v2.Position = dvec3(1,0,0);

		glm::dvec3 v1v0 = v1.Position - v0.Position;
		glm::dvec3 v2v0 = v2.Position - v0.Position;
		face_normal = normalize(cross(v1v0, v2v0));

		inv_area = 1.f / (double)(0.5 * (v0.Position.x * v1.Position.y + v1.Position.x * v2.Position.y + v2.Position.x * v0.Position.y - v1.Position.x * v0.Position.y - v2.Position.x * v1.Position.y - v0.Position.x * v2.Position.y));
		bounds = Union(Bounds3f(v0.Position, v1.Position), (glm::vec3)v2.Position);;
	};

	__host__ __device__ Triangle(const Vertex v0, const Vertex v1, const Vertex v2) :
		v0(v0), v1(v1), v2(v2)
	{
		glm::dvec3 v1v0 = v1.Position - v0.Position;
		glm::dvec3 v2v0 = v2.Position - v0.Position;
		face_normal = normalize(cross(v1v0, v2v0));
		inv_area = 1.f / (double)(0.5 * (v0.Position.x * v1.Position.y + v1.Position.x * v2.Position.y + v2.Position.x * v0.Position.y - v1.Position.x * v0.Position.y - v2.Position.x * v1.Position.y - v0.Position.x * v2.Position.y));
		bounds = Union(Bounds3f(v0.Position, v1.Position), (glm::vec3)v2.Position);
	};

	__device__ Triangle* clone(void) const
	{
		return (new Triangle(*this));
	}

	__device__ virtual bool hit(const Ray& ray, double& tmin, ShadeRec& sr) const
	{
		glm::dvec3 v1v0 = v1.Position - v0.Position;
		glm::dvec3 v2v0 = v2.Position - v0.Position;
		glm::dvec3 rov0 = ray.o - v0.Position;

		glm::dvec3 n = cross(v1v0, v2v0);
		glm::dvec3 q = cross(rov0, ray.d);

		double d = 1.0 / dot(ray.d, n);
		double u = d * dot(-q, v2v0);

		if (u < 0.0)
			return (false);

		double v = d * dot(q, v1v0);

		if (v < 0.0 || u + v > 1.0)
			return (false);

		double t = d * dot(-n, rov0);

		if (t < K_EPSILON)
			return (false);

		vec3 normal = normalize((1.f - u - v) * v0.Normal + u * v1.Normal + v * v2.Normal);

		tmin = t;
		sr.normal = normal;
		sr.local_hit_point = ray.o + t * ray.d;

		return (true);
	};

	__device__ virtual bool hit(const Ray& ray) const
	{
		glm::dvec3 v1v0 = v1.Position - v0.Position;
		glm::dvec3 v2v0 = v2.Position - v0.Position;
		glm::dvec3 rov0 = ray.o - v0.Position;

		glm::dvec3 n = cross(v1v0, v2v0);
		glm::dvec3 q = cross(rov0, ray.d);

		double d = 1.0 / dot(ray.d, n);
		double u = d * dot(-q, v2v0);

		if (u < 0.0)
			return (false);

		double v = d * dot(q, v1v0);

		if (v < 0.0 || u + v > 1.0)
			return (false);

		double t = d * dot(-n, rov0);

		if (t < K_EPSILON)
			return (false);

		return (true);
	}

	__device__ virtual bool shadow_hit(const Ray& ray, double& tmin) const
	{
		if (!shadows)
			return (false);

		glm::dvec3 v1v0 = v1.Position - v0.Position;
		glm::dvec3 v2v0 = v2.Position - v0.Position;
		glm::dvec3 rov0 = ray.o - v0.Position;

		glm::dvec3 n = cross(v1v0, v2v0);
		glm::dvec3 q = cross(rov0, ray.d);

		double d = 1.0 / dot(ray.d, n);
		double u = d * dot(-q, v2v0);

		if (u < 0.0)
			return (false);

		double v = d * dot(q, v1v0);

		if (v < 0.0 || u + v > 1.0)
			return (false);

		double t = d * dot(-n, rov0);

		if (t < K_EPSILON)
			return (false);

		tmin = t;

		return (true);
	};

	__device__ virtual void set_sampler(Sampler* sampler)
	{

	};

	__device__ virtual glm::vec3 sample(void)
	{
		return glm::vec3(1);
	};

	__device__ virtual glm::dvec3 get_normal(const glm::dvec3 p)
	{
		return face_normal;
	};

	__device__ virtual Bounds3f get_bounding_box(void)
	{
		return bounds;
	};

	__device__ virtual Bounds3f get_bounding_box(mat4 mat)
	{
		vec3 _v0 = vec3(mat * vec4(v0.Position,1));
		vec3 _v1 = vec3(mat * vec4(v1.Position,1));
		vec3 _v2 = vec3(mat * vec4(v2.Position,1));

		return Union(Bounds3f(_v0, _v1), (glm::vec3)_v2);
	};
private:
	Vertex v0, v1, v2;
	glm::dvec3 face_normal;
	Bounds3f bounds;
};

#endif // _RAYTRACER_GEOMETRICOBJECTS_TRIANGLE_H_
