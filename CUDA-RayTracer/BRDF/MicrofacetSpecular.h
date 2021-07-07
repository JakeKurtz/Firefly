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

	__device__ virtual float3 f(const ShadeRec& sr, const float3& wi, const float3& wo) const
	{
		float3 L = make_float3(0,0,0);

		float3 n = sr.normal;
		float3 h = normalize(wo + wi);

		float n_dot_wi = abs(dot(n, wi));
		float n_dot_wo = abs(dot(n, wo));

		float D = ggxtr_ndf(n, h);
		float G = geo_atten(wi, wo, n);
		float3 F = fresnel(sr.material_ptr->get_fresnel_reflectance(), h, wo);

		L = (float)ks * (D * G * F) / (4.f * n_dot_wo * n_dot_wi);

		return (L);
	};

	__device__ virtual float3 sample_f(const ShadeRec& sr, const float3& wo, float3& wi, float& pdf) const
	{
		float e0 = random();
		float e1 = random();

		float theta = atan(r * r * sqrtf(e0 / (1.f - e0)));
		float phi = 2 * M_PI * e1;

		float3 h = make_float3(
			sin(theta) * cos(phi),
			cos(theta),
			sin(theta) * sin(phi)
		);

		float3 N = sr.normal;
		float3 T = normalize(cross(N, get_orthogonal_vec(N)));
		float3 B = normalize(cross(N, T));

		wi = -reflect(wo, normalize(T * h.x + N * h.y + B * h.z));
		pdf = get_pdf(sr.normal, wi, wo);

		return f(sr, wi, wo);
	};

	__device__ virtual float3 rho(const ShadeRec& sr, const float3& wo) const
	{
		return make_float3(0,0,0);
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

	__device__ float ggxtr_ndf(float3 n, float3 h) const
	{
		float a2 = pow(r * r, 2);
		float NH2 = pow(fmaxf(0.0, dot(n, h)), 2);
		return a2 / (M_PI * (pow(NH2 * (a2 - 1.f) + 1.f, 2)));
	};

	__device__ float geo_atten(float3 wi, float3 wo, float3 n) const
	{
		float k = pow(r + 1.f, 2.f) / 8.f;

		float NL = fmaxf(dot(n, wi), 0.0);
		float NV = fmaxf(dot(n, wo), 0.0);

		float G1 = NL / (NL * (1.f - k) + k);
		float G2 = NV / (NV * (1.f - k) + k);

		return G1 * G2;
	};

	__device__ virtual float3 sample(const ShadeRec& sr, const float3& wo) const
	{
		float3 N = sr.normal;

		float e0 = random();
		float e1 = random();

		float theta = atan(r*r * sqrtf( e0 / (1.f - e0)));
		float phi = 2*M_PI*e1;

		float3 h = make_float3(
			sin(theta) * cos(phi),
			cos(theta),
			sin(theta) * sin(phi)
		);

		float3 T = normalize(cross(N, get_orthogonal_vec(N)));
		float3 B = normalize(cross(N, T));

		float3 sample = T * h.x + N * h.y + B * h.z;

		float3 wi = -reflect(wo, normalize(T * h.x + N * h.y + B * h.z));

		return (wi);
	};

	__device__ float get_pdf(float3 n, float3 wi, float3 wo) const
	{
		float3 h = normalize(wo + wi);

		float h_dot_n = abs(dot(h, n));
		float wo_dot_h = abs(dot(wo, n));

		float D = ggxtr_ndf(n, h);

		return ( D * h_dot_n / ((4.0 * wo_dot_h)) );
	};
};

#endif // _RAYTRACER_BRDF_MICROFACETSPECULAR_H_

