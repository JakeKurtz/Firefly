#pragma once

#include <glm/glm.hpp>

#include "../Utilities/Math.h"
#include "../Utilities/Ray.h"

template <typename T> class Bounds3 {
public:
	__host__ __device__ Bounds3() {
		T minNum = std::numeric_limits<T>::lowest();
		T maxNum = std::numeric_limits<T>::max();
		pMin = glm::tvec3<T>(maxNum);
		pMax = glm::tvec3<T>(minNum);
	}

	__host__ __device__ Bounds3(const glm::tvec3<T>& p) : pMin(p), pMax(p) {};

	__host__ __device__ Bounds3(const glm::tvec3<T>& p1, const glm::tvec3<T>& p2) :
		pMin(glm::min(p1.x,p2.x), glm::min(p1.y, p2.y), glm::min(p1.z, p2.z)),
		pMax(glm::max(p1.x,p2.x), glm::max(p1.y, p2.y), glm::max(p1.z, p2.z))
	{};

	__host__ __device__ const glm::tvec3<T>& operator[](int i) const 
	{
		return (i == 0) ? pMin : pMax;
	};

	__host__ __device__ glm::tvec3<T>& operator[](int i) 
	{
		return (i == 0) ? pMin : pMax;
	};

	__host__ __device__ bool operator==(const Bounds3& b) const
	{
		return b.pMin == pMin && b.pMax == pMax;
	};

	__host__ __device__ bool operator!=(const Bounds3& b) const
	{
		return b.pMin != pMin || b.pMax != pMax;
	};

	__host__ __device__ glm::tvec3<T> corner(int corner) const {
		return glm::tvec3<T>(
			(*this)[corner & 1].x,
			(*this)[(corner & 2) ? 1 : 0].y,
			(*this)[(corner & 4) ? 1 : 0].z
		);
	};

	__host__ __device__ glm::tvec3<T> diagonal() const { return pMax - pMin; };

	__host__ __device__ double surface_area() const
	{
		glm::tvec3<T> d = diagonal();
		return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z);
	}

	__host__ __device__ double volume() const {
		glm::tvec3<T> d = diagonal();
		return d.x * d.y * d.z;
	}

	__host__ __device__ int maximum_extent() const {
		glm::tvec3<T> d = diagonal();
		if (d.x > d.y && d.x > d.z)
			return 0;
		else if (d.y > d.z)
			return 1;
		else
			return 2;
	}

	__host__ __device__ glm::tvec3<T> lerp(const glm::tvec3<T>& t) const {
		return glm::tvec3<T>(lerp(t.x, pMin.x, pMax.x),
					 lerp(t.y, pMin.y, pMax.y),
					 lerp(t.z, pMin.z, pMax.z));
	}

	__host__ __device__ glm::tvec3<T> offset(const glm::tvec3<T>& p) const {
		glm::tvec3<T> o = p - pMin;
		if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
		if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
		if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
		return o;
	}

	__host__ __device__ void bounding_sphere(glm::tvec3<T>* center, float* radius) const {
		*center = (pMin + pMax) / 2.0;
		*radius = Inside(*center, *this) ? distance(*center, pMax) : 0;
	}

	__host__ __device__ bool hit(const Ray& ray) const
	{
		float tmin = (pMin.x - ray.o.x) / ray.d.x;
		float tmax = (pMax.x - ray.o.x) / ray.d.x;

		if (tmin > tmax) {
			//swap(tmin, tmax);
			float temp = tmin;
			tmin = tmax;
			tmax = temp;
		}

		float tymin = (pMin.y - ray.o.y) / ray.d.y;
		float tymax = (pMax.y - ray.o.y) / ray.d.y;

		if (tymin > tymax) {
			//swap(tymin, tymax);
			float temp = tymin;
			tymin = tymax;
			tymax = temp;
		}

		if ((tmin > tymax) || (tymin > tmax))
			return false;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;

		float tzmin = (pMin.z - ray.o.z) / ray.d.z;
		float tzmax = (pMax.z - ray.o.z) / ray.d.z;

		if (tzmin > tzmax) {
			//swap(tzmin, tzmax);
			float temp = tzmin;
			tzmin = tzmax;
			tzmax = temp;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;

		return true;
	};

	__host__ __device__ inline bool hit(const Ray& ray, const vec3& invDir, const int dirIsNeg[3]) const 
	{
		const Bounds3<float>& bounds = *this;

		float tmin = (bounds[dirIsNeg[0]].x - ray.o.x) * invDir.x;
		float tmax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;

		if (tmin > tmax) {
			float temp = tmin;
			tmin = tmax;
			tmax = temp;
		}

		float tymin = (bounds[dirIsNeg[1]].y - ray.o.y) * invDir.y;
		float tymax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;

		if (tymin > tymax) {
			float temp = tymin;
			tymin = tymax;
			tymax = temp;
		}

		if ((tmin > tymax) || (tymin > tmax))
			return false;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;

		float tzmin = (bounds[dirIsNeg[2]].z - ray.o.z) * invDir.z;
		float tzmax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;

		if (tzmin > tzmax) {
			float temp = tzmin;
			tzmin = tzmax;
			tzmax = temp;
		}

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;

		return true;
	}

	glm::tvec3<T>   pMin, pMax;
};

typedef Bounds3<float> Bounds3f;
typedef Bounds3<int>   Bounds3i;