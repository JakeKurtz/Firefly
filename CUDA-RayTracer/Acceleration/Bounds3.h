#pragma once

#include "../Utilities/Math.h"
#include "../Utilities/Ray.h"
#include "../Utilities/cutil_math.h"

__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }

class Bounds3f {
public:
	__host__ __device__ Bounds3f() {
		float minNum = std::numeric_limits<float>::lowest();
		float maxNum = std::numeric_limits<float>::max();
		pMin = make_float3(maxNum, maxNum, maxNum);
		pMax = make_float3(minNum, minNum, minNum);
	}

	__host__ __device__ Bounds3f(const float3& p) : pMin(p), pMax(p) {};

	__host__ __device__ Bounds3f(const float3& p1, const float3& p2)
	{
		pMin = make_float3(fminf(p1.x, p2.x), fminf(p1.y, p2.y), fminf(p1.z, p2.z));
		pMax = make_float3(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y), fmaxf(p1.z, p2.z));
	};

	__host__ __device__ const float3& __restrict__ operator[](int i) const
	{
		return (i == 0) ? pMin : pMax;
	};

	__host__ __device__ float3& __restrict__ operator[](int i)
	{
		return (i == 0) ? pMin : pMax;
	};

	__host__ __device__ bool operator==(const Bounds3f& b) const
	{
		return b.pMin == pMin && b.pMax == pMax;
	};

	__host__ __device__ bool operator!=(const Bounds3f& b) const
	{
		return b.pMin != pMin || b.pMax != pMax;
	};

	__host__ __device__ float3 corner(int corner) const {
		return make_float3(
			(*this)[corner & 1].x,
			(*this)[(corner & 2) ? 1 : 0].y,
			(*this)[(corner & 4) ? 1 : 0].z
		);
	};

	__host__ __device__ float3 diagonal() const { return pMax - pMin; };

	__host__ __device__ double surface_area() const
	{
		float3 d = diagonal();
		return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z);
	}

	__host__ __device__ double volume() const {
		float3 d = diagonal();
		return d.x * d.y * d.z;
	}

	__host__ __device__ int maximum_extent() const {
		float3 d = diagonal();
		if (d.x > d.y && d.x > d.z)
			return 0;
		else if (d.y > d.z)
			return 1;
		else
			return 2;
	}

	__host__ __device__ float3 lerp_(const float3& t) const {
		return make_float3(lerp(t.x, pMin.x, pMax.x),
						   lerp(t.y, pMin.y, pMax.y),
						   lerp(t.z, pMin.z, pMax.z));
	}

	__host__ __device__ float3 offset(const float3& p) const {
		float3 o = p - pMin;
		if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
		if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
		if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
		return o;
	}

	__host__ __device__ void bounding_sphere(float3* center, float* radius) const {
		//*center = (pMin + pMax) / 2.0;
		//*radius = inside(*center, *this) ? distance(*center, pMax) : 0;
	}

	__device__ bool hit(const Ray& __restrict__ ray) const
	{
		/*
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
		*/
		float tmin, tmax, tymin, tymax, tzmin, tzmax;

		tmin = (pMin.x - ray.o.x) / ray.d.x;
		tmax = (pMax.x - ray.o.x) / ray.d.x;
		tymin = (pMin.y - ray.o.y) / ray.d.y;
		tymax = (pMax.y - ray.o.y) / ray.d.y;
		tzmin = (pMin.z - ray.o.z) / ray.d.z;
		tzmax = (pMax.z - ray.o.z) / ray.d.z;

		float tminbox = min_max(tmin, tmax, min_max(tymin, tymax, min_max(tzmin, tzmax, 0)));
		float tmaxbox = max_min(tmin, tmax, max_min(tymin, tymax, max_min(tzmin, tzmax, K_HUGE)));

		return (tminbox <= tmaxbox);
	};

	__device__ inline bool hit(const Ray& __restrict__ ray, const float3& __restrict__ invDir, const int dirIsNeg[3]) const
	{
		/*
		const Bounds3<float>& __restrict__ bounds = *this;

		float tmin = (bounds[dirIsNeg[0]].x - ray.o.x) * invDir.x;
		float tmax = (bounds[1 - dirIsNeg[0]].x - ray.o.x)* invDir.x;

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
		{
			return false;
		}

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
		{
			return false;
		}

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;
		
		return true;
		*/
		float tmin, tmax, tymin, tymax, tzmin, tzmax;

		tmin = ((*this)[dirIsNeg[0]].x - ray.o.x) * invDir.x;
		tmax = ((*this)[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
		tymin = ((*this)[dirIsNeg[1]].y - ray.o.y) * invDir.y;
		tymax = ((*this)[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;
		tzmin = ((*this)[dirIsNeg[2]].z - ray.o.z) * invDir.z;
		tzmax = ((*this)[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;

		float tminbox = min_max(tmin, tmax, min_max(tymin, tymax, min_max(tzmin, tzmax, 0)));
		float tmaxbox = max_min(tmin, tmax, max_min(tymin, tymax, max_min(tzmin, tzmax, K_HUGE)));

		return (tminbox <= tmaxbox);
	}

	float3 pMin, pMax;
};

//typedef Bounds3<float> Bounds3f;
//typedef Bounds3<int>   Bounds3i;