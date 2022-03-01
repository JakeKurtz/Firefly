#pragma once
#include <cuda_runtime.h>

struct dVertex {
	float3 position;
	float3 normal;
	float2 texcoords;
	float3 tangent;
	float3 bitangent;
};