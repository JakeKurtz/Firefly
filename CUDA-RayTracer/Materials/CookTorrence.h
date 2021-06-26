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

	__device__ void set_cd(const vec3& cd)
	{
		ambient_brdf->set_cd(cd);
		diffuse_brdf->set_cd(cd);
	};

	__device__ virtual void set_diffuse_sampler(Sampler* sp) { diffuse_brdf->set_sampler(sp); }

	__device__ float power_heuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = nf * fPdf;
		float g = ng * gPdf;
		return (f * f) / (f * f + g * g);
	}

	 __device__ virtual vec3 shade(ShadeRec& sr)
	 {
		dvec3 wo = -sr.ray.d;
		vec3 L = vec3(0.f);

		vec3 f = vec3(0.f);
		vec3 Li = vec3(0.f);
		//vec3 L = ambient_brdf->rho(sr, wo) *sr.s.ambient_ptr->L(sr);
		int num_lights = sr.s.lights.size();
		float brdf_pdf, light_pdf, weight;
		
		for (int i = 0; i < num_lights; i++) {

			// Sample Lights
			dvec3 wi, sample_point;
			sr.s.lights[i]->get_direction(sr, wi, sample_point);
			float n_dot_wi = dot(sr.normal, wi);
			if (n_dot_wi > 0.f) {
				bool in_shadow = false;

				if (sr.s.lights[i]->casts_shadows()) {
					Ray shadow_ray(sr.local_hit_point, wi);
					in_shadow = sr.s.lights[i]->in_shadow(shadow_ray, sr);
				}

				if (!in_shadow) {

					brdf_pdf = specular_brdf->get_pdf(sr.normal, wi, wo);
					light_pdf = sr.s.lights[i]->get_pdf(sr);
					weight = power_heuristic(1, light_pdf, 1, brdf_pdf);

					glm::dvec3 h = normalize(wo + wi);
					vec3 F = fresnel(sr.material_ptr->get_fresnel_reflectance(), h, wo);

					//printf("(%f,%f,%f)\n", Fo.x, Fo.y, Fo.z);

					f = (specular_brdf->f(sr, wi, wo) * weight) + (diffuse_brdf->f(sr, wi, wo)) * n_dot_wi;
					//f = (specular_brdf->f(sr, wi, wo) * weight) + (diffuse_brdf->f(sr, wi, wo) * (vec3(1) - Fi) * (vec3(1) - Fo)) * n_dot_wi;

					//f = lerp(diffuse_brdf->f(sr, wi, wo), (specular_brdf->f(sr, wi, wo) * weight), F) * n_dot_wi;
					Li = sr.s.lights[i]->L(sr, wi, sample_point);

					if (f != vec3(0.f) && light_pdf != 0.f && Li != vec3(0.f)) {
						L += glm::max(vec3(0.f), f * Li / light_pdf);
					}
				}
			}

			// Sample BRDF
			wi = specular_brdf->sample(sr, wo);

			Ray visibility_ray(sr.local_hit_point, wi);
			ShadeRec foobar(sr.s);
			double tmin;
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

					float n_dot_d = dot(-foobar.normal, wi);

					f = specular_brdf->f(sr, wi, wo) * n_dot_wi;
					Li = sr.s.lights[i]->L(sr, wi, sample_point);
					
					if (f != vec3(0.f) && brdf_pdf != 0.f && Li != vec3(0.f)) {
						L += glm::max(vec3(0.f), f * Li * weight / brdf_pdf);
					}
				}
			}
		}
		return (L);
	};

	__device__ virtual vec3 sample_f(ShadeRec& sr, const glm::dvec3& wo, glm::dvec3& wi, float& pdf) 
	{
		if (random() < 0.5) {
			wi = diffuse_brdf->sample(sr, wo);
			sr.specular_sample = false;
			//if (wo.z < 0) wi.z *= -1;
		}
		else {
			wi = specular_brdf->sample(sr, wo);
			sr.specular_sample = true;
			//if (wo.z * wi.z <= 0) return vec3(0.f);
		}

		double pdf_diff = diffuse_brdf->get_pdf(sr.normal, wi, wo);
		double pdf_spec = specular_brdf->get_pdf(sr.normal, wi, wo);
		pdf = 0.5f * (pdf_diff + pdf_spec);

		glm::dvec3 h = normalize(wo + wi);
		vec3 Fo = fresnel(sr.material_ptr->get_fresnel_reflectance(), h, wo);
		vec3 f = lerp(diffuse_brdf->f(sr, wi, wo), (specular_brdf->f(sr, wi, wo)), Fo);

		return specular_brdf->f(sr, wi, wo) + diffuse_brdf->f(sr, wi, wo);
	}

private:
	Lambertian*				ambient_brdf;
	Lambertian*				diffuse_brdf;
	MicrofacetSpecular*		specular_brdf;
};

#endif // _RAYTRACER_MATERIALS_COOKTORRENCE_H_