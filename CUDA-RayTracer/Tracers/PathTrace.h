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

	__device__ virtual glm::vec3 trace_ray(Ray& ray) const {

		vec3 L = vec3(0);
		vec3 beta = vec3(1.f);
		int maxDepth = 3;

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
			dvec3 wo = -sr.ray.d, wi;
			float pdf;

			vec3 f = glm::max(vec3(0.0), sr.material_ptr->sample_f(sr, wo, wi, pdf));

			if (f == vec3(0.f) || pdf == 0.f)
				break;

			float n_dot_wi = glm::max(0.0, glm::dot(sr.normal, wi));
			
			beta *= f * n_dot_wi / pdf;

			ray = Ray(sr.local_hit_point, wi);

			if (bounces > 3) {
				float q = glm::max((float).05, 1 - beta.y);
				if (random() < q)
					break;
				beta /= 1 - q;
			}
		}
		return (L);
	};
};
