#pragma once
#include <cuda_runtime.h>

class Isect;
class dLight;

struct dMaterial
{
	float3 baseColorFactor;
	float3 emissiveColorFactor;
	float3 fresnel;

	float roughnessFactor = 1.f;
	float metallicFactor = 0.f;

	float ks = 1.f;
	float kd = 1.f;
	float radiance = 0.f;

	int baseColorTexture;
	int normalTexture;
	int occlusionTexture;
	int emissiveTexture;
	int roughnessTexture;
	int metallicTexture;
	int metallicRoughnessTexture;
};

__device__ float3 get_albedo(const Isect& isect);
__device__ float get_roughness(const Isect& isect);
__device__ float get_metallic(const Isect& isect);
__device__ float3 get_normal(const Isect& isect);

__device__ float3 refract(const float3& I, const float3& N, const float& ior);
__device__ double power_heuristic(int nf, double fPdf, int ng, double gPdf);
__device__ double ggxtr_ndf(float3 n, float3 h, float r);
__device__ double geo_atten(float3 wi, float3 wo, float3 n, float r);

__device__ float3 diff_sample(const Isect& isect);
__device__ double diff_get_pdf();
__device__ float3 diff_f(const Isect& isect, const float3& wi, const float3& wo);
__device__ float3 diff_L(dLight** lights, const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point);

__device__ float3 ct_sample(const Isect& isect, const float3& wo);
__device__ float ct_get_pdf(float3 n, float3 wi, float3 wo, float r);
__device__ float3 ct_f(const Isect& isect, const float3& wi, const float3& wo);
__device__ float3 ct_sample_f(const Isect& isect, const float3& wo, float3& wi, float& pdf);
__device__ float3 ct_L(dLight** lights, const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point, float r);

__device__ float3 emissive_L(const Isect& isect, const float3& ray_dir);
__device__ float3 emissive_L(const Isect& isect);
__device__ float3 emissive_L(const dMaterial* material_ptr);