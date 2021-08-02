#ifndef _RAYTRACER_BRDF_BRDF_H_
#define _RAYTRACER_BRDF_BRDF_H_

#include "ShadeRec.cuh"
#include "Math.cuh"
#include "Random.cuh"

class BRDF
{
public:

	__device__ BRDF(void) {};

	__device__ BRDF(const BRDF& brdf) {};

	BRDF& operator= (const BRDF& rhs) {
		if (this == &rhs)
			return (*this);

		return (*this);
	};

	__device__ virtual float3 f(const ShadeRec& sr, const float3& wi, const float3& wo) const = 0;

	__device__ virtual float3 sample_f(const ShadeRec& sr, const float3& wo, float3& wi, float& pdf) const = 0;

	__device__ virtual float3 sample(const ShadeRec& sr, const float3& wo) const = 0;

	__device__ virtual float3 rho(const ShadeRec& sr, const float3& wo) const = 0;

	__device__ virtual ~BRDF() {};
};

#endif // _RAYTRACER_BRDF_BRDF_H_
