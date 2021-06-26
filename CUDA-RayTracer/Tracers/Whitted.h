#ifndef __WHITTED__
#define __WHITTED__

#include "Tracer.h"
#include "../Scene/Scene.h"
#include "../Utilities/ShadeRec.h"
#include "../Materials/Material.h"

class Whitted : public Tracer {
public:

	__device__ Whitted(void)
		: Tracer()
	{};

	__device__ Whitted(Scene* _scene_ptr)
		: Tracer(_scene_ptr)
	{};

	__device__ virtual ~Whitted(void) {};

	__device__ virtual glm::vec3 trace_ray(Ray& ray) const {
		ShadeRec sr(scene_ptr->hit_objs(ray));

		if (sr.hit_an_obj) {
			sr.ray = ray;
			return (sr.material_ptr->shade(sr));
		}
		else
			return (scene_ptr->background_color);
	}
};

#endif