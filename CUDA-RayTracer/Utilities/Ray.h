#pragma once

#include <glm/glm.hpp>
#include <crt/host_defines.h>

class Ray {
public:
	glm::dvec3 o;
	glm::dvec3 d;

	__device__ Ray(void) : 
		o(0.f),
		d(0.f)
	{};
	__device__ Ray(const glm::dvec3& origin, const glm::dvec3& dir) {
		o = origin;
		d = dir;
	};

	__device__ Ray(const Ray& ray)
		: o(ray.o),
		d(ray.d)
	{};
};
