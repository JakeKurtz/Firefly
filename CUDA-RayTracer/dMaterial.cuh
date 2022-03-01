#pragma once
#include <cuda_runtime.h>

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