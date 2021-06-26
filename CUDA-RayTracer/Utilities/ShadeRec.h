#pragma once
#include "Ray.h"

class Material;
class Scene;

class ShadeRec
{
public:

	bool			hit_an_obj;
	double			t;
	Material*		material_ptr;
	glm::dvec3		hit_point;			// world coord of hit point
	glm::dvec3		local_hit_point;	// for attaching textures to object
	glm::dvec3		normal;
	Ray				ray;				// for specular highlights
	glm::vec3		direction;			// for area lighs
	Scene&			s;
	int				depth;
	bool			specular_sample;

	__device__ ShadeRec(Scene& s) :
		hit_an_obj(false),
		material_ptr(NULL),
		hit_point(0), local_hit_point(0),
		normal(0), ray(), direction(0), s(s), depth(0), specular_sample(false)
	{};

	__device__ ShadeRec(const ShadeRec& sr) :
		hit_an_obj(sr.hit_an_obj),
		material_ptr(sr.material_ptr),
		hit_point(sr.hit_point), local_hit_point(sr.local_hit_point),
		normal(sr.normal), ray(sr.ray), direction(sr.direction), s(sr.s), depth(0), specular_sample(sr.specular_sample)
	{}

	ShadeRec& ShadeRec::operator= (const ShadeRec& rhs) { return *this; }
};

