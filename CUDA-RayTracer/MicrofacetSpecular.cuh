#ifndef _RAYTRACER_BRDF_MICROFACETSPECULAR_H_
#define _RAYTRACER_BRDF_MICROFACETSPECULAR_H_

#include "BRDF.cuh"

class MicrofacetSpecular : public BRDF
{
public:

	__device__ MicrofacetSpecular(void) :
		r(1.f),
		ks(1.f)
	{}

	__device__ virtual float3 f(const ShadeRec& sr, const float3& wi, const float3& wo) const
	{
		float3 L = make_float3(0, 0, 0);

		float3 n = sr.normal;
		float3 h = normalize(wo + wi);

		double n_dot_wi = abs(dot(n, wi));
		double n_dot_wo = abs(dot(n, wo));

		double D = ggxtr_ndf(n, h);
		double G = geo_atten(wi, wo, n);
		float3 F = fresnel(sr.material_ptr->get_fresnel_reflectance(), h, wo);

		L = ks * (D * G * F) / (4.f * n_dot_wo * n_dot_wi);

		return (L);
	};

	__device__ virtual float3 sample_f(const ShadeRec& sr, const float3& wo, float3& wi, float& pdf) const
	{
		float e0 = random();
		float e1 = random();

		double theta = atan(r * r * sqrtf(e0 / (1.f - e0)));
		double phi = 2 * M_PI * e1;

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
		return make_float3(0, 0, 0);
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

	__device__ double ggxtr_ndf(float3 n, float3 h) const
	{
		double a2 = pow(r * r, 2);
		double NH2 = pow(fmaxf(0.0, dot(n, h)), 2);
		return a2 / (M_PI * (pow(NH2 * (a2 - 1.f) + 1.f, 2)));
	};

	__device__ double geo_atten(float3 wi, float3 wo, float3 n) const
	{
		double k = pow(r + 1.f, 2.f) / 8.f;

		double NL = fmaxf(dot(n, wi), 0.0);
		double NV = fmaxf(dot(n, wo), 0.0);

		double G1 = NL / (NL * (1.f - k) + k);
		double G2 = NV / (NV * (1.f - k) + k);

		return G1 * G2;
	};

	__device__ virtual float3 sample(const ShadeRec& sr, const float3& wo) const
	{
		float3 N = sr.normal;

		float e0 = random();
		float e1 = random();

		double theta = atan(r * r * sqrtf(e0 / (1.f - e0)));
		double phi = 2 * M_PI * e1;

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

		double h_dot_n = abs(dot(h, n));
		double wo_dot_h = abs(dot(wo, n));

		double D = ggxtr_ndf(n, h);

		return (D * h_dot_n / ((4.0 * wo_dot_h)));
	};
};

#endif // _RAYTRACER_BRDF_MICROFACETSPECULAR_H_

