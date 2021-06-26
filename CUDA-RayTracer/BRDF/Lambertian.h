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

	__device__ void set_cd(const glm::vec3& _cd) {
		cd = _cd;
	};

	__device__ glm::vec3 get_cd(void) {
		return cd;
	};

	__device__ virtual glm::vec3 f(const ShadeRec& sr, const glm::dvec3& wi, const glm::dvec3& wo) const {
		return (kd * cd * M_1_PI);
		//return (vec3) cd * (28.f / (23.f * M_PI)) * 0.5f * 0.5f * (vec3)(1.f - pow(1.f - 0.5f * glm::dot(sr.normal, wi), 5)) * (vec3)(1.f - pow(1 - 0.5f * glm::dot(sr.normal, wo), 5));
	};

	__device__ virtual glm::vec3 sample_f(const ShadeRec& sr, const glm::dvec3& wo, glm::dvec3& wi, float& pdf) const {

		glm::vec3 N = sr.normal;

		float e0 = random();
		float e1 = random();

		float sinTheta = sqrtf(1 - e0 * e0);
		float phi = 2 * M_PI * e1;
		float x = sinTheta * cosf(phi);
		float z = sinTheta * sinf(phi);
		vec3 sp = dvec3(x, e0, z);

		glm::vec3 T = normalize(cross(N, get_orthogonal_vec(N)));
		glm::vec3 B = normalize(cross(N, T));

		wi = T * sp.x + N * sp.y + B * sp.z;
		pdf = glm::abs(glm::dot(sr.normal, wi)) * M_1_PI;

		return (kd * cd * M_1_PI);
	};

	__device__ virtual glm::vec3 sample(const ShadeRec& sr, const glm::dvec3& wo) const {

		glm::vec3 N = sr.normal;

		float e0 = random();
		float e1 = random();

		float sinTheta = sqrtf(1 - e0 * e0);
		float phi = 2 * M_PI * e1;
		float x = sinTheta * cosf(phi);
		float z = sinTheta * sinf(phi);
		vec3 sp = dvec3(x, e0, z);

		glm::vec3 T = normalize(cross(N, get_orthogonal_vec(N)));
		glm::vec3 B = normalize(cross(N, T));

		vec3 wi = T * sp.x + N * sp.y + B * sp.z;

		return (wi);
	};

	__device__ double get_pdf(glm::dvec3 n, glm::dvec3 wi, glm::dvec3 wo) const
	{
		return glm::abs(glm::dot(n, wi)) * M_1_PI;
	};

	__device__ virtual glm::vec3 rho(const ShadeRec& sr, const glm::dvec3& wo) const {
		return (kd * cd);
	};

private:
	float kd;
	glm::vec3 cd;
};

#endif // !_RAYTRACER_BRDF_LAMBERTIAN_H_
