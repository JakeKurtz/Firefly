#ifndef _RAYTRACER_MATERIALS_MATERIAL_H_
#define _RAYTRACER_MATERIALS_MATERIAL_H_

#include "ShadeRec.cuh"
#include "Math.cuh"

//#include "../Lights/AmbientLight.h"
//#include "../Lights/PointLight.h"
//#include "../Lights/Light.h"

class Material
{
public:
	__device__ virtual float3 shade(ShadeRec& sr) { return make_float3(0, 0, 0); };
	__device__ virtual float3 f_diffuse(ShadeRec& sr) { return make_float3(0, 0, 0); };
	__device__ virtual float3 f_specular(ShadeRec& sr) { return make_float3(0, 0, 0); };
	__device__ virtual float3 sample_f(ShadeRec& sr, const float3& wo, float3& wi, float& pdf) { return make_float3(0, 0, 0); };
	__device__ virtual float3 sample_f_diffuse(ShadeRec& sr, const float3& wo, float3& wi, float& pdf) { return make_float3(0, 0, 0); };
	__device__ virtual float3 sample_f_specular(ShadeRec& sr, const float3& wo, float3& wi, float& pdf) { return make_float3(0, 0, 0); };
	__device__ virtual float3 get_fresnel_reflectance() { return fresnel_reflectance; };
	__device__ virtual void set_fresnel_reflectance(float3 r) { fresnel_reflectance = r; };
	__device__ virtual float get_ks(void)
	{
		return 1.f;
	};
	__device__ virtual float3 get_cd(void)
	{
		return make_float3(1, 1, 1);
	};

	__device__ virtual float get_kd(void)
	{
		return 1.f;
	};

	__device__ virtual bool is_emissive(void)
	{
		return (false);
	};
	__device__ virtual float3 sample(float3 N) { return make_float3(1, 1, 1); };
	__device__ virtual ~Material() {};
protected:
	float3 fresnel_reflectance = make_float3(1, 1, 1);
};

#endif // _RAYTRACER_MATERIALS_MATERIAL_H_
