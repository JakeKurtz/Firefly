#ifndef _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_
#define _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_

#include "../Utilities/ShadeRec.h"
#include "../Acceleration/BBox.h"
#include "../Acceleration/Bounds3.h"

#include <glm/glm.hpp>

class Sampler;

class GeometricObj {

public:
	__device__ virtual GeometricObj* clone(void) const = 0;

	__device__ virtual bool hit(const Ray& ray, double& tmin, ShadeRec& sr) const = 0;

	__device__ virtual bool hit(const Ray& ray) const = 0;

	__device__ virtual bool shadow_hit(const Ray& ray, double& tmin) const = 0;					// intersects a shadow ray with the object

	__device__ virtual void set_sampler(Sampler* sampler) {};

	__device__ virtual glm::vec3 sample(void)
	{
		return glm::vec3(0.0);
	};

	__device__ virtual glm::dvec3 get_normal(const glm::dvec3 p)
	{
		return glm::dvec3(0.0);
	};

	__device__ virtual Bounds3f get_bounding_box(void)
	{
		return Bounds3f(glm::vec3(-K_HUGE), glm::vec3(K_HUGE));
	};

	__device__ virtual Bounds3f get_bounding_box(glm::mat4 mat)
	{
		return Bounds3f(glm::vec3(-K_HUGE), glm::vec3(K_HUGE));
	};

	__device__ void set_color(float r, float g, float b)
	{
		color = glm::vec3(r, g, b);
	};

	__device__ glm::vec3 get_color()
	{
		return color;
	};

	__device__ Material* get_material(void)
	{
		return material_ptr;
	};

	__device__ void set_material(Material* _material_ptr)
	{
		material_ptr = _material_ptr;
	};

	__device__ void enable_shadows(bool b)
	{
		shadows = b;
	};

	__device__ virtual float pdf(const ShadeRec& sr)
	{
		return inv_area;
	};

	__device__ void add_object(GeometricObj* object_ptr) {}

	__device__ virtual ~GeometricObj() {};

protected:
	glm::vec3			color;
	mutable Material*	material_ptr;
	bool				shadows = true;
	bool				transformed = false;
	Sampler*			sampler_ptr;
	float				inv_area;
};

// ------ Geometry Inline Functions ------ //

template <typename T> __host__ __device__ Bounds3<T> intersect(const Bounds3<T>& b1, const Bounds3<T>& b2) {
	return Bounds3<T>(
		glm::tvec3<T>(
			glm::max(b1.pMin.x, b2.pMin.x),
			glm::max(b1.pMin.y, b2.pMin.y),
			glm::max(b1.pMin.z, b2.pMin.z)),
		glm::tvec3<T>(
			glm::min(b1.pMax.x, b2.pMax.x),
			glm::min(b1.pMax.y, b2.pMax.y),
			glm::min(b1.pMax.z, b2.pMax.z)));
};

template <typename T> __host__ __device__ bool overlaps(const Bounds3<T>& b1, const Bounds3<T>& b2) {
	bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
	bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
	bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
	return (x && y && z);
};

template <typename T>__host__ __device__ bool inside(const glm::vec3& p, const Bounds3<T>& b) {
	return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
		p.y >= b.pMin.y && p.y <= b.pMax.y &&
		p.z >= b.pMin.z && p.z <= b.pMax.z);
};

template <typename T> __host__ __device__ bool inside_exclusive(const glm::tvec3<T>& p, const Bounds3<T>& b) {
	return (p.x >= b.pMin.x && p.x < b.pMax.x&&
		p.y >= b.pMin.y && p.y < b.pMax.y&&
		p.z >= b.pMin.z && p.z < b.pMax.z);
};

template <typename T> __host__ __device__ Bounds3 <T> Union(const Bounds3<T>& b, const glm::tvec3<T>& p) {
	return Bounds3<T>(
		glm::tvec3<T>(
			glm::min(b.pMin.x, p.x),
			glm::min(b.pMin.y, p.y),
			glm::min(b.pMin.z, p.z)),
		glm::tvec3<T>(
			glm::max(b.pMax.x, p.x),
			glm::max(b.pMax.y, p.y),
			glm::max(b.pMax.z, p.z)));
};

template <typename T> __host__ __device__ Bounds3<T> Union(const Bounds3<T>& b1, const Bounds3<T>& b2) {
	return Bounds3<T>(
		glm::tvec3<T>(
			glm::min(b1.pMin.x, b2.pMin.x),
			glm::min(b1.pMin.y, b2.pMin.y),
			glm::min(b1.pMin.z, b2.pMin.z)),
		glm::tvec3<T>(
			glm::max(b1.pMax.x, b2.pMax.x),
			glm::max(b1.pMax.y, b2.pMax.y),
			glm::max(b1.pMax.z, b2.pMax.z)));
};

#endif // _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_