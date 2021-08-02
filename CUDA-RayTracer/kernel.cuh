#pragma once

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <cstdint>
#include <cfloat>

class Camera;
class Triangle;
class Light;
class BVHAccel;
struct ViewPlane;
struct LinearBVHNode;
struct Ray;
struct ShadeRec;

/*
struct Ray
{
	float3 origin;
	float3 direction;
	float3 invD;
	float3 OoD;
	float minDistance = 0.0f;
	float maxDistance = FLT_MAX;
};
*/
struct Isect
{
	float3 position;
	float3 normal;
	float2 texcoord;
	float distance = FLT_MAX;
	uint32_t materialIndex = 0;
	bool wasFound = false;
};

// path data in SOA format
struct Paths
{
	float2* filmSamplePosition;
	float3* throughput;
	float3* result;
	uint32_t* length;
	Ray* extensionRay;
	Isect* extensionIntersection;
	float3* extensionBrdf;
	float* extensionBrdfPdf;
	float* extensionCosine;
	Ray* lightRay;
	float3* lightEmittance;
	float3* lightBrdf;
	float3* lightSamplePoint;
	float* lightBrdfPdf;
	float* lightPdf;
	float3* Li;
	float* lightCosine;
	bool* lightRayBlocked;
};

// wavefront queues
struct Queues
{
	uint32_t* newPathQueue;
	uint32_t* diffuseMaterialQueue;
	uint32_t* extensionRayQueue;
	uint32_t* lightRayQueue;

	uint32_t newPathQueueLength;
	uint32_t diffuseMaterialQueueLength;
	uint32_t extensionRayQueueLength;
	uint32_t lightRayQueueLength;
};

__constant__  Camera* g_camera_ptr;
__constant__  ViewPlane* g_viewplane_ptr;
__constant__  LinearBVHNode* g_bvh;
__constant__  Triangle* g_triangles;
__constant__  Light** g_lights;
__constant__  int g_num_lights = 1;
__device__ int shadowrays = 0;

__device__ Ray* g_raypool;
__device__ ShadeRec* g_srpool;

BVHAccel* bvh;

texture<float4, 1> t_BVHbounds;
texture<int4, 1> t_BVHnodes;

GLuint vbo;
uchar4* d_vbo = NULL;
struct cudaGraphicsResource* cuda_pbo_resource = NULL; // CUDA Graphics Resource (to transfer PBO)

bool g_bFirstTime = true;
unsigned int framenumber = 0;

