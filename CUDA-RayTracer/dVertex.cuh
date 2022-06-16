#pragma once
#include <cuda_runtime.h>
#include "dMatrix.cuh"
#include "dMath.cuh"

struct dVertex {
	float3 position;
	float3 normal;
	float2 texcoords;
	float3 tangent;
	float3 bitangent;
};