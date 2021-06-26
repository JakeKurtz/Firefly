#pragma once
#include "ViewPlane.h"

#include "../Cameras/PinholeCamera.h"
#include "../Cameras/Camera.h"

#include "../Utilities/ShadeRec.h"
#include "../Acceleration/BVHAccel.h"

#include "../Lights/Light.h"

#include "../GeometricObjects/GeometricObj.h"

#include <glm/glm.hpp>

using namespace std;

class GeometricObj;
class Tracer;

enum Comp {
	Y = 1,
	YA = 2,
	RGB = 3,
	RGBA = 4
};

class Scene
{
public:
	ViewPlane					vp;
	Camera*						camera_ptr;
	glm::vec3					background_color;
	Light*						ambient_ptr;
	CudaList<Light*>			lights;
	CudaList<GeometricObj*>		objects;
	LinearBVHNode*				bvh;
	Tracer*						tracer_ptr = nullptr;

	__device__ Scene(void);

	__device__ void add_obj(GeometricObj* obj_ptr);

	__device__ void add_light(Light* obj_ptr);

	__device__ ShadeRec hit_objs(const Ray& ray);

	__device__ void set_tracer(Tracer* _tracer_ptr) { tracer_ptr = _tracer_ptr; }
};

__device__ Scene::Scene(void) :
	background_color(glm::vec3(0.f))
{}

__device__ void Scene::add_obj(GeometricObj* obj_ptr) {
	objects.add(obj_ptr);
}

inline __device__ void Scene::add_light(Light* light_ptr)
{
	return lights.add(light_ptr);
}

inline __device__ ShadeRec Scene::hit_objs(const Ray& ray)
{
	ShadeRec	sr(*this);
	/*
	double		t;
	glm::dvec3	normal;
	glm::vec3	local_hit_point;
	double		tmin = K_HUGE;
	
	int	num_objs = objects.size();
	for (int i = 0; i < num_objs; i++) {
		if (objects[i]->hit(ray, t, sr) && (t < tmin)) {
			sr.hit_an_obj = true;
			tmin = t;
			sr.material_ptr = objects[i]->get_material();
			normal = sr.normal;
			local_hit_point = sr.local_hit_point;
		}
	}

	if (sr.hit_an_obj) {
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
	}
	*/
	Intersect(ray, sr, bvh, objects);

	return (sr);
}