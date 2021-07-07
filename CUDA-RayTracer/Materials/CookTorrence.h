#ifndef _RAYTRACER_MATERIALS_COOKTORRENCE_H_
#define _RAYTRACER_MATERIALS_COOKTORRENCE_H_

#include "Material.h"
#include "../BRDF/Lambertian.h"
#include "../BRDF/MicrofacetSpecular.h"
#include "../Utilities/Ray.h"
#include "../Tracers/Tracer.h"

class CookTorrence : public Material
{
public:

	__device__ CookTorrence(void) :
		ambient_brdf(new Lambertian()),
		diffuse_brdf(new Lambertian()),
		specular_brdf(new MicrofacetSpecular())
	{};

	__device__ void set_ka(const float ka)
	{
		ambient_brdf->set_kd(ka);
	};

	__device__ void set_kd(const float kd)
	{
		diffuse_brdf->set_kd(kd);
	};

	__device__ void set_ks(const float ks)
	{
		specular_brdf->set_ks(ks);
	};

	__device__ void set_roughness(const float roughness)
	{
		specular_brdf->set_roughness(roughness);
	};

	__device__ void set_cd(const float3& cd)
	{
		ambient_brdf->set_cd(cd);
		diffuse_brdf->set_cd(cd);
	};

	__device__ virtual void set_diffuse_sampler(Sampler* sp) { diffuse_brdf->set_sampler(sp); }

	__device__ double power_heuristic(int nf, double fPdf, int ng, double gPdf)
	{
		double f = nf * fPdf;
		double g = ng * gPdf;
		return (f * f) / (f * f + g * g);
	}

	 __device__ virtual float3 shade(ShadeRec& sr)
	 {
		float3 wo = -sr.ray.d;
		float3 L = make_float3(0,0,0);

		float3 f = make_float3(0,0,0);
		float3 Li = make_float3(0,0,0);

		int num_lights = sr.s.lights.size();
		double brdf_pdf, light_pdf, weight;
		
		for (int i = 0; i < num_lights; i++) {

			// Sample Lights
			float3 wi, sample_point;
			sr.s.lights[i]->get_direction(sr, wi, sample_point);
			float n_dot_wi = dot(sr.normal, wi);
			if (n_dot_wi > 0.f) {
				bool in_shadow = false;

				if (sr.s.lights[i]->casts_shadows()) {
					Ray shadow_ray(sr.local_hit_point, wi);
					in_shadow = sr.s.lights[i]->in_shadow(shadow_ray, sr);
				}
				// check this out
				// https://computergraphics.stackexchange.com/questions/4288/path-weight-for-direct-light-sampling
				if (!in_shadow) {
					
					brdf_pdf = specular_brdf->get_pdf(sr.normal, wi, wo);
					light_pdf = sr.s.lights[i]->get_pdf(sr);
					weight = power_heuristic(1, light_pdf, 1, brdf_pdf);

					f = (specular_brdf->f(sr, wi, wo) * weight) + (diffuse_brdf->f(sr, wi, wo)) * n_dot_wi;
					Li = sr.s.lights[i]->L(sr, wi, sample_point);

					if (f != make_float3(0,0,0) && light_pdf != 0.f && Li != make_float3(0,0,0)) {
						L += fmaxf(make_float3(0,0,0), f * Li / light_pdf);
					}
				}
			}

			// Sample BRDF
			wi = specular_brdf->sample(sr, wo);

			Ray visibility_ray(sr.local_hit_point, wi);
			ShadeRec foobar(sr.s);
			float tmin;
			n_dot_wi = dot(sr.normal, wi);

			if (n_dot_wi > 0.f && sr.s.lights[i]->visible(visibility_ray, tmin, foobar)) {
				bool in_shadow = false;

				if (sr.s.lights[i]->casts_shadows()) {
					Ray shadow_ray(sr.local_hit_point, wi);
					in_shadow = sr.s.lights[i]->in_shadow(shadow_ray, sr);
				}
				
				if (!in_shadow) {
					brdf_pdf = specular_brdf->get_pdf(sr.normal, wi, wo);
					light_pdf = sr.s.lights[i]->get_pdf(foobar, visibility_ray);
					weight = power_heuristic(1, brdf_pdf, 1, light_pdf);

					f = specular_brdf->f(sr, wi, wo) * n_dot_wi;
					Li = sr.s.lights[i]->L(sr, wi, sample_point);
					
					if (f != make_float3(0,0,0) && brdf_pdf != 0.f && Li != make_float3(0,0,0)) {
						L += fmaxf(make_float3(0,0,0), f * Li * weight / brdf_pdf);
					}
				}
			}
		}
		return (L);
	};

	__device__ virtual float3 sample_f(ShadeRec& sr, const float3& wo, float3& wi, float& pdf) 
	{
		if (random() < 0.5) {
			wi = diffuse_brdf->sample(sr, wo);
			sr.specular_sample = false;
		}
		else {
			wi = specular_brdf->sample(sr, wo);
			sr.specular_sample = true;
		}

		float pdf_diff = diffuse_brdf->get_pdf(sr.normal, wi, wo);
		float pdf_spec = specular_brdf->get_pdf(sr.normal, wi, wo);
		pdf = 0.5f * (pdf_diff + pdf_spec);

		return specular_brdf->f(sr, wi, wo) + diffuse_brdf->f(sr, wi, wo);
	}

private:
	Lambertian*				ambient_brdf;
	Lambertian*				diffuse_brdf;
	MicrofacetSpecular*		specular_brdf;
};

#endif // _RAYTRACER_MATERIALS_COOKTORRENCE_H_