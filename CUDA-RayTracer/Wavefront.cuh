#pragma once
#include <vector_types.h>
#include <cstdint>
#include <cfloat>

class dMaterial;
class dRay;
class Isect;

struct Paths
{
    float2* film_pos;
    float* n_samples;
    float3* throughput;
    uint32_t* length;

    dRay* ext_ray;
    Isect* ext_isect;
    float3* ext_brdf;
    float* ext_pdf;
    float* ext_cosine;
    bool* ext_specular;

    dRay* light_ray;
    int* light_id;
    float3* light_emittance;
    float3* light_brdf;
    float3* light_samplePoint;
    float* light_pdf;
    float* light_cosine;
    bool* light_inshadow;
};

struct Queues
{
    uint32_t* queue_newPath;
    uint32_t* queue_mat_diffuse;
    uint32_t* queue_mat_cook;
    uint32_t* queue_mat_mix;
    uint32_t* queue_extension;
    uint32_t* queue_shadow;

    uint32_t	queue_newPath_length;
    uint32_t	queue_mat_diffuse_length;
    uint32_t	queue_mat_cook_length;
    uint32_t	queue_mat_mix_length;
    uint32_t	queue_extension_length;
    uint32_t	queue_shadow_length;
};