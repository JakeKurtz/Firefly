#ifndef _RAYTRACER_BRDF_LAMBERTIAN_H_
#define _RAYTRACER_BRDF_LAMBERTIAN_H_

#include "BRDF.h"

class Lambertian : public BRDF
{
public:

	__device__ void set_kd(const float _kd) {
		kd = _kd;
	};

	__device__ float get_kd(void) {
		return kd;
	};

	__device__ void set_cd(const float3& _cd) {
		cd = _cd;
	};

	__device__ float3 get_cd(void) {
		return cd;
	};

	__device__ virtual float3 f(const ShadeRec& sr, const float3& wi, const float3& wo) const {
		return (kd * cd * M_1_PI);
		//return (vec3) cd * (28.f / (23.f * M_PI)) * 0.5f * 0.5f * (vec3)(1.f - pow(1.f - 0.5f * glm::dot(sr.normal, wi), 5)) * (vec3)(1.f - pow(1 - 0.5f * glm::dot(sr.normal, wo), 5));
	};

	__device__ virtual float3 sample_f(const ShadeRec& sr, const float3& wo, float3& wi, float& pdf) const {

		float3 N = sr.normal;

		float e0 = random();
		float e1 = random();

		float sinTheta = sqrtf(1 - e0 * e0);
		float phi = 2 * M_PI * e1;
		float x = sinTheta * cosf(phi);
		float z = sinTheta * sinf(phi);
		float3 sp = make_float3(x, e0, z);

		float3 T = normalize(cross(N, get_orthogonal_vec(N)));
		float3 B = normalize(cross(N, T));

		wi = T * sp.x + N * sp.y + B * sp.z;
		pdf = abs(dot(sr.normal, wi)) * M_1_PI;

		return (kd * cd * M_1_PI);
	};

	__device__ virtual float3 sample(const ShadeRec& sr, const float3& wo) const {

		float3 N = sr.normal;

		float e0 = random();
		float e1 = random();

		float sinTheta = sqrtf(1 - e0 * e0);
		float phi = 2 * M_PI * e1;
		float x = sinTheta * cosf(phi);
		float z = sinTheta * sinf(phi);
		float3 sp = make_float3(x, e0, z);

		float3 T = normalize(cross(N, get_orthogonal_vec(N)));
		float3 B = normalize(cross(N, T));

		float3 wi = T * sp.x + N * sp.y + B * sp.z;

		return (wi);
	};

	__device__ double get_pdf(float3 n, float3 wi, float3 wo) const
	{
		return abs(dot(n, wi)) * M_1_PI;
	};

	__device__ virtual float3 rho(const ShadeRec& sr, const float3& wo) const {
		return (kd * cd);
	};

private:
	float kd;
	float3 cd;
};

#endif // !_RAYTRACER_BRDF_LAMBERTIAN_H_
