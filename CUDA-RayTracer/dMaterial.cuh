#pragma once
#include <cuda_runtime.h>

class Isect;
class dLight;
class LinearBVHNode;
class dTriangle;

struct dMaterial
{
	float3 baseColorFactor;
	float3 emissiveColorFactor;
	float3 fresnel = make_float3(0.04, 0.04, 0.04);

	float roughnessFactor = 1.f;
	float metallicFactor = 0.f;

	float ks = 1.f;
	float kd = 1.f;
	float radiance = 0.f;

	bool emissive = 0;

	int baseColorTexture = -1;
	int normalTexture = -1;
	int occlusionTexture = -1;
	int emissiveTexture = -1;
	int roughnessTexture = -1;
	int metallicTexture = -1;
	int metallicRoughnessTexture = -1;
};

__device__ float3 fresnel(float3 f0, float3 h, float3 wo);
__device__ float3 fresnel_roughness(float3 f0, float3 n, float3 wo, float r);

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

__device__ float3 spec_sample(const Isect& isect, const float3& wo);
__device__ float spec_get_pdf(float3 n, float3 wi, float3 wo, float r);
__device__ float3 spec_f(const Isect& isect, const float3& wi, const float3& wo);
__device__ float3 spec_sample_f(const Isect& isect, const float3& wo, float3& wi, float& pdf);
__device__ float3 spec_L(dLight** lights, const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point, float r);

__device__ float3 BRDF_L(dLight** lights, LinearBVHNode nodes[], dTriangle triangles[], const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point, float3& sample_dir);
__device__ float3 BRDF_f(const Isect& isect, const float3& wi, const float3& wo);
__device__ float BRDF_pdf(const Isect& isect, const float3 wi, const float3 wo);

__device__ float3 emissive_L(const Isect& isect, const float3& ray_dir);
__device__ float3 emissive_L(const Isect& isect);
__device__ float3 emissive_L(const dMaterial* material_ptr);