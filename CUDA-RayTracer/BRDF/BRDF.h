#ifndef _RAYTRACER_BRDF_BRDF_H_
#define _RAYTRACER_BRDF_BRDF_H_

#include "Utilities/ShadeRec.h"
#include "Utilities/Math.h"
#include "Utilities/Random.h"

class BRDF
{
public:

	__device__ BRDF(void) {};

	__device__ BRDF(const BRDF& brdf) {};

	//__device__ virtual BRDF* clone(void)const = 0;

	BRDF& operator= (const BRDF& rhs) {
		if (this == &rhs)
			return (*this);

		//if (sampler_ptr) {
		//	delete sampler_ptr;
		//	sampler_ptr = NULL;
		//}

		//if (rhs.sampler_ptr)
		//	sampler_ptr = rhs.sampler_ptr->clone();

		return (*this);
	};

	__device__ virtual float3 f(const ShadeRec& sr, const float3& wi, const float3& wo) const = 0;

	__device__ virtual float3 sample_f(const ShadeRec& sr, const float3& wo, float3& wi, float& pdf) const = 0;

	__device__ virtual float3 sample(const ShadeRec& sr, const float3& wo) const = 0;

	__device__ virtual float3 rho(const ShadeRec& sr, const float3& wo) const = 0;

	__device__ virtual void set_sampler(Sampler* sp)
	{
		//if (sampler_ptr) {
		//	delete sampler_ptr;
		//	sampler_ptr = nullptr;
		//}

		sampler_ptr = sp;
		//sampler_ptr->generate_samples();
	};

	__device__ virtual ~BRDF() {};

protected:
	Sampler* sampler_ptr;
};

#endif // _RAYTRACER_BRDF_BRDF_H_
