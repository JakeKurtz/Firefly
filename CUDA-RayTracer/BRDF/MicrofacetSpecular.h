#ifndef _RAYTRACER_BRDF_MICROFACETSPECULAR_H_
#define _RAYTRACER_BRDF_MICROFACETSPECULAR_H_

#include "BRDF.h"

class MicrofacetSpecular : public BRDF
{
public:

	__device__ MicrofacetSpecular(void) :
		r(1.f),
		ks(1.f)
	{}

	__device__ virtual glm::vec3 f(const ShadeRec& sr, const glm::dvec3& wi, const glm::dvec3& wo) const
	{
		glm::vec3 L = glm::vec3(0.f);

		glm::dvec3 n = sr.normal;
		glm::dvec3 h = normalize(wo + wi);

		double n_dot_wi = glm::abs(dot(n, wi));
		double n_dot_wo = glm::abs(dot(n, wo));

		double D = ggxtr_ndf(n, h);
		double G = geo_atten(wi, wo, n);
		glm::dvec3 F = fresnel(sr.material_ptr->get_fresnel_reflectance(), h, wo);

		L = (double)ks * (D * G * F) / (4.f * n_dot_wo * n_dot_wi);
		return (L);
	};

	__device__ virtual glm::vec3 sample_f(const ShadeRec& sr, const glm::dvec3& wo, glm::dvec3& wi, float& pdf) const
	{
		float e0 = random();
		float e1 = random();

		float theta = atan(r * r * sqrt(e0 / (1.f - e0)));
		float phi = 2 * M_PI * e1;

		glm::vec3 h = glm::vec3(
			sin(theta) * cos(phi),
			cos(theta),
			sin(theta) * sin(phi)
		);

		glm::vec3 N = sr.normal;
		glm::vec3 T = normalize(cross(N, get_orthogonal_vec(N)));
		glm::vec3 B = normalize(cross(N, T));

		wi = -glm::reflect(wo, (dvec3)glm::normalize(T * h.x + N * h.y + B * h.z));
		pdf = get_pdf(sr.normal, wi, wo);

		return f(sr, wi, wo);
	};

	__device__ virtual glm::vec3 rho(const ShadeRec& sr, const glm::dvec3& wo) const
	{
		return glm::vec3(0.f);
	};

	__device__ void set_ks(const float k)
	{
		ks = k;
	};

	__device__ float get_ks(void)
	{
		return ks;
	};

	__device__ void set_roughness(const float roughness)
	{
		r = roughness;
	};

	__device__ float get_roughness(void)
	{
		return r;
	};

//private:
	float r;	// roughness 
	float ks;

	__device__ double ggxtr_ndf(glm::dvec3 n, glm::dvec3 h) const
	{
		double a2 = pow(r * r, 2);
		double NH2 = pow(glm::max(0.0, dot(n, h)), 2);
		return a2 / (M_PI * (pow(NH2 * (a2 - 1.f) + 1.f, 2)));
	};

	__device__ double geo_atten(glm::dvec3 wi, glm::dvec3 wo, glm::dvec3 n) const
	{
		double k = pow(r + 1.f, 2.f) / 8.f;

		double NL = glm::max(dot(n, wi), 0.0);
		double NV = glm::max(dot(n, wo), 0.0);

		double G1 = NL / (NL * (1.f - k) + k);
		double G2 = NV / (NV * (1.f - k) + k);

		return G1 * G2;
	};

	__device__ virtual glm::vec3 sample(const ShadeRec& sr, const glm::dvec3& wo) const
	{
		glm::vec3 N = sr.normal;

		float e0 = random();
		float e1 = random();

		float theta = atan(r*r * sqrt( e0 / (1.f - e0)));
		float phi = 2*M_PI*e1;

		glm::vec3 h = glm::vec3(
			sin(theta) * cos(phi),
			cos(theta),
			sin(theta) * sin(phi)
		);

		glm::vec3 T = normalize(cross(N, get_orthogonal_vec(N)));
		glm::vec3 B = normalize(cross(N, T));

		glm::vec3 sample = T * h.x + N * h.y + B * h.z;

		vec3 wi = -glm::reflect(wo, (dvec3)glm::normalize(T * h.x + N * h.y + B * h.z));

		return (wi);
	};

	__device__ double get_pdf(glm::dvec3 n, glm::dvec3 wi, glm::dvec3 wo) const
	{
		glm::dvec3 h = normalize(wo + wi);

		double h_dot_n = glm::abs(dot(h, n));
		double wo_dot_h = glm::abs(dot(wo, n));

		double D = ggxtr_ndf(n, h);

		return ( D * h_dot_n / ((4.0 * wo_dot_h)) );
	};
};

#endif // _RAYTRACER_BRDF_MICROFACETSPECULAR_H_

