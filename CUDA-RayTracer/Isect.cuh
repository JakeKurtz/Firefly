#pragma once
#include <cuda_runtime.h>
#include "dMath.cuh"
#include "dMaterial.cuh"

struct Isect
{
    float3			position;
    float3			normal;
    float3			tangent;
    float3			bitangent;
    float2			texcoord;
    float			distance = K_HUGE;
    dMaterial*      material = nullptr;
    bool			wasFound = false;
    int             triangle_id = -1;
};