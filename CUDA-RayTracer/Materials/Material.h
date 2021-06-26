#ifndef _RAYTRACER_MATERIALS_MATERIAL_H_
#define _RAYTRACER_MATERIALS_MATERIAL_H_

#include "Utilities/ShadeRec.h"

#include "Scene/Scene.h"

#include "../Lights/AmbientLight.h"
#include "../Lights/PointLight.h"
#include "../Lights/Light.h"

class Material
{
public:
	__device__ virtual glm::vec3 shade(ShadeRec& sr) { return glm::vec3(0.f); };
	__device__ virtual glm::vec3 sample_f(ShadeRec& sr, const glm::dvec3& wo, glm::dvec3& wi, float& pdf) { return glm::vec3(0.f); };
	__device__ virtual glm::vec3 get_fresnel_reflectance() { return fresnel_reflectance; };
	__device__ virtual void set_fresnel_reflectance(glm::vec3 r) { fresnel_reflectance = r; };
	__device__ virtual float get_ks(void)
	{
		return 1.f;
	};
	__device__ virtual vec3 get_cd(void)
	{
		return vec3(1.f);
	};
	__device__ virtual bool is_emissive(void)
	{
		return (false);
	};
	__device__ virtual glm::dvec3 sample(glm::vec3 N) { return vec3(1); };
	__device__ virtual ~Material() {};
private:
	glm::vec3 fresnel_reflectance = glm::vec3(1.f);
};

#endif // _RAYTRACER_MATERIALS_MATERIAL_H_
