#ifndef _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_
#define _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_

#include "../Utilities/ShadeRec.h"
#include "../Acceleration/BBox.h"
#include "../Acceleration/Bounds3.h"

class Sampler;

class GeometricObj {

public:
	__device__ virtual GeometricObj* clone(void) const = 0;

	__device__ virtual bool hit(const Ray& ray, float& tmin, ShadeRec& sr) const = 0;

	__device__ virtual bool hit(const Ray& ray) const = 0;

	__device__ virtual bool shadow_hit(const Ray& ray, float& tmin) const = 0;					// intersects a shadow ray with the object

	__device__ virtual void set_sampler(Sampler* sampler) {};

	__device__ virtual float3 sample(void)
	{
		return make_float3(0,0,0);
	};

	__device__ virtual float3 get_normal(const float3 p)
	{
		return make_float3(0,0,0);
	};

	__device__ virtual Bounds3f get_bounding_box(void)
	{
		return Bounds3f(make_float3(-K_HUGE), make_float3(K_HUGE));
	};

	//__device__ virtual Bounds3f get_bounding_box(glm::mat4 mat)
	//{
	//	return Bounds3f(make_float3(-K_HUGE), make_float3(K_HUGE));
	//};

	__device__ void set_color(float r, float g, float b)
	{
		color = make_float3(r, g, b);
	};

	__device__ float3 get_color()
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
	float3				color;
	mutable Material*	material_ptr;
	bool				shadows = true;
	bool				transformed = false;
	Sampler*			sampler_ptr;
	float				inv_area;
};

// ------ Geometry Inline Functions ------ //

__host__ __device__ Bounds3f intersect(const Bounds3f& b1, const Bounds3f& b2) {
	return Bounds3f(
		make_float3(
			fmaxf(b1.pMin.x, b2.pMin.x),
			fmaxf(b1.pMin.y, b2.pMin.y),
			fmaxf(b1.pMin.z, b2.pMin.z)),
		make_float3(
			fminf(b1.pMax.x, b2.pMax.x),
			fminf(b1.pMax.y, b2.pMax.y),
			fminf(b1.pMax.z, b2.pMax.z)));
};

__host__ __device__ bool overlaps(const Bounds3f& b1, const Bounds3f& b2) {
	bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
	bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
	bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
	return (x && y && z);
};

__host__ __device__ bool inside(float3& p, const Bounds3f& b) {
	return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
		p.y >= b.pMin.y && p.y <= b.pMax.y &&
		p.z >= b.pMin.z && p.z <= b.pMax.z);
};

__host__ __device__ bool inside_exclusive(const float3& p, const Bounds3f& b) {
	return (p.x >= b.pMin.x && p.x < b.pMax.x&&
		p.y >= b.pMin.y && p.y < b.pMax.y&&
		p.z >= b.pMin.z && p.z < b.pMax.z);
};

__host__ __device__ Bounds3f Union(const Bounds3f& b, const float3& p) {
	return Bounds3f(
		make_float3(
			fminf(b.pMin.x, p.x),
			fminf(b.pMin.y, p.y),
			fminf(b.pMin.z, p.z)),
		make_float3(
			fmaxf(b.pMax.x, p.x),
			fmaxf(b.pMax.y, p.y),
			fmaxf(b.pMax.z, p.z)));
};

__host__ __device__ Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2) {
	return Bounds3f(
		make_float3(
			fminf(b1.pMin.x, b2.pMin.x),
			fminf(b1.pMin.y, b2.pMin.y),
			fminf(b1.pMin.z, b2.pMin.z)),
		make_float3(
			fmaxf(b1.pMax.x, b2.pMax.x),
			fmaxf(b1.pMax.y, b2.pMax.y),
			fmaxf(b1.pMax.z, b2.pMax.z)));
};

#endif // _RAYTRACER_GEOMETRICOBJECTS_GEOMETRICOBJ_H_