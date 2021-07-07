#pragma once

#include "Tracer.h"
#include "../Scene/Scene.h"
#include "../Utilities/ShadeRec.h"
#include "../Materials/Material.h"

class PathTrace : public Tracer {
public:

	__device__ PathTrace(void)
		: Tracer()
	{};

	__device__ PathTrace(Scene* _scene_ptr)
		: Tracer(_scene_ptr)
	{};

	__device__ virtual ~PathTrace(void) {};

	__device__ virtual float3 trace_ray(Ray& ray) const {

		float3 L = make_float3(0,0,0);
		float3 beta = make_float3(1,1,1);
		int maxDepth = 1;

		for (int bounces = 0;; ++bounces) 
		{
			// Intersect ray with scene
			ShadeRec sr(scene_ptr->hit_objs(ray));
			sr.ray = ray;
			sr.depth = bounces;

			// Terminate path if ray escaped or maxDepth is reached
			if (!sr.hit_an_obj || bounces >= maxDepth)
				break;
			
			// Possibly add emitted light at intersection
			if (bounces == 0 && sr.material_ptr->is_emissive()) {
				L += sr.material_ptr->shade(sr);
			}
			else if (!sr.material_ptr->is_emissive()) {
				// Sample illumination from lights to find path contribution
				L += beta * sr.material_ptr->shade(sr);
			}
			else break;
			
			// Sample BRDF to get new path direction
			float3 wo = -sr.ray.d, wi;
			float pdf;

			float3 f = fmaxf(make_float3(0,0,0), sr.material_ptr->sample_f(sr, wo, wi, pdf));

			if (f == make_float3(0,0,0) || pdf == 0.f)
				break;

			float n_dot_wi = fmaxf(0.0, dot(sr.normal, wi));
			
			beta *= f * n_dot_wi / pdf;

			ray = Ray(sr.local_hit_point, wi);

			if (bounces > 3) {
				float q = fmaxf((float).05, 1 - beta.y);
				if (random() < q)
					break;
				beta /= 1 - q;
			}
		}
		return (L);
	};
};
