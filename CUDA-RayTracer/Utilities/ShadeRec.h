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
	float3			hit_point;			// world coord of hit point
	float3			local_hit_point;	// for attaching textures to object
	float3			normal;
	Ray				ray;				// for specular highlights
	float3			direction;			// for area lighs
	Scene&			s;
	int				depth;
	bool			specular_sample;

	__device__ ShadeRec(Scene& s) :
		hit_an_obj(false),
		material_ptr(NULL),
		ray(), s(s), depth(0), specular_sample(false)
	{
		hit_point = make_float3(0,0,0);
		local_hit_point = make_float3(0,0,0);
		normal = make_float3(0,0,0);
		direction = make_float3(0,0,0);
	};

	__device__ ShadeRec(const ShadeRec& sr) :
		hit_an_obj(sr.hit_an_obj),
		material_ptr(sr.material_ptr),
		hit_point(sr.hit_point), local_hit_point(sr.local_hit_point),
		normal(sr.normal), ray(sr.ray), direction(sr.direction), s(sr.s), depth(0), specular_sample(sr.specular_sample)
	{}

	ShadeRec& ShadeRec::operator= (const ShadeRec& rhs) { return *this; }
};

