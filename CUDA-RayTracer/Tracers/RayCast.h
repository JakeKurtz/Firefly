#ifndef _RAYTRACER_TRACERS_RAYCATS_H_
#define _RAYTRACER_TRACERS_RAYCATS_H_

#include "Tracer.h"
//#include "Materials/Material.h"
//#include "../Scene/Scene.h"

class RayCast : public Tracer
{
public:

	Tracer::Tracer;

	__device__ virtual vec3 trace_ray(const Ray ray) const;
};

__device__ vec3 RayCast::trace_ray(const Ray ray) const
{
	//ShadeRec sr(scene_ptr->hit_objs(ray));

	/*if (sr.hit_an_obj) {
		sr.ray = ray;
		//return (sr.material_ptr->shade(sr));
	}
	else
	{
		return (scene_ptr->background_color);
	}
	return vec3(0.f);*/
}

#endif // _RAYTRACER_TRACERS_RAYCATS_H_
