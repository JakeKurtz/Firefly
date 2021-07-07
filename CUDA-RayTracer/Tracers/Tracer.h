#ifndef __TRACER__
#define __TRACER__

#include "../Utilities/Ray.h"

class Scene;

class Tracer {
public:

	__device__ Tracer(void) 
		: scene_ptr(NULL)
	{};

	__device__ Tracer(Scene* _scene_ptr)
		: scene_ptr(_scene_ptr)
	{};

	__device__ virtual ~Tracer(void) {
		if (scene_ptr)
			scene_ptr = NULL;
	};

	__device__ virtual float3 trace_ray(Ray& ray) const {
		return make_float3(0,0,0);
	};

protected:
	Scene* scene_ptr;
};

#endif

