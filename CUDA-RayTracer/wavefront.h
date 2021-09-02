#pragma once
#include <vector_types.h>
#include <cstdint>
#include <cfloat>

class Camera;
class Light;
class BVHAccel;
struct ViewPlane;
struct LinearBVHNode;
struct Ray;
struct ShadeRec;
struct Material;
struct Vertex;

enum class MaterialIndex { Diffuse, CookTor, Emissive };

struct Isect
{
    float3			position;
    float3			normal;
    float2			texcoord;
    float			distance = FLT_MAX;
    MaterialIndex	materialIndex;
    Material* material_ptr = nullptr;
    bool			wasFound = false;
};

struct Triangles
{
    float* inv_area;
    Material** material_ptr;
    MaterialIndex* materialIndex;
    Vertex* v0;
    Vertex* v1;
    Vertex* v2;
    float3* face_normal;
};

struct Paths
{
    float2* film_pos;
    float3* throughput;
    uint32_t* length;

    Ray* ext_ray;
    Isect* ext_isect;
    float3* ext_brdf;
    float* ext_pdf;
    float* ext_cosine;

    Ray* light_ray;
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
    uint32_t* queue_extension;
    uint32_t* queue_shadow;

    uint32_t	queue_newPath_length;
    uint32_t	queue_mat_diffuse_length;
    uint32_t	queue_mat_cook_length;
    uint32_t	queue_extension_length;
    uint32_t	queue_shadow_length;
};

Paths* paths;
Queues* queues;
Triangles* triangles_SoA;