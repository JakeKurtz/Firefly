#pragma once

#define STB_IMAGE_WRITE_IMPLEMENTATION

#define GLFW_INCLUDE_NONE
//#define __CUDACC__

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <cuda_gl_interop.h>
#include <texture_indirect_functions.h>

#include <curand_kernel.h>
#include <cuda_texture_types.h>
#include <cstdint>
#include <cfloat>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

class Camera;
class Triangle;
class Light;
class BVHAccel;
struct ViewPlane;
struct LinearBVHNode;
struct Ray;
struct ShadeRec;
struct Material;
struct Vertex;

enum class MaterialIndex { Diffuse, CookTor, Emissive, Mix };

typedef unsigned int uint;
typedef unsigned char uchar;

struct Isect
{
    float3			position;
    float3			normal;
    float2			texcoord;
    float			distance = FLT_MAX;
    MaterialIndex	materialIndex;
    Material* material_ptr = nullptr;
    bool			wasFound = false;
    int             triangle_id = -1;
};

struct Paths
{
    float2* film_pos;
    float* n_samples;
    float3* throughput;
    uint32_t* length;

    Ray* ext_ray;
    Isect* ext_isect;
    float3* ext_brdf;
    float* ext_pdf;
    float* ext_cosine;
    bool* ext_specular;

    Ray* light_ray;
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

const int       IMAGE_WIDTH     = 1280;
const int       IMAGE_HEIGHT    = 720;
const int       SCR_WIDTH       = 1280; // 640
const int       SCR_HEIGHT      = 720; // 360
const int       CHANNEL_COUNT   = 4;
const int       DATA_SIZE       = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNEL_COUNT;
const GLenum    PIXEL_FORMAT    = GL_RGBA;

const uint32_t PATHCOUNT = SCR_WIDTH * SCR_HEIGHT;
const uint32_t MAXPATHLENGTH = 10;

const int BLOCKSIZE = 256;
const int GRIDSIZE = (PATHCOUNT + BLOCKSIZE - 1) / BLOCKSIZE;

__constant__	Camera*			g_camera_ptr;
__constant__	ViewPlane*		g_viewplane_ptr;
__constant__	LinearBVHNode*	g_nodes;
__constant__	Triangle*		g_triangles;
__constant__	Light**			g_lights;
__constant__	int				g_num_lights		= 1;
__device__		int				shadowrays			= 0;

cudaTextureObject_t albedo_maps;
texture<float3, 1> specular_maps[];
texture<float3, 1> normal_maps[];
texture<float3, 1> bump_maps[];

LinearBVHNode* d_nodes;
BVHAccel* bvh;

GLuint textureId;
GLubyte* imageData = 0;             // pointer to texture buffer
GLuint pbo;
uchar4* d_pbo = NULL;
struct cudaGraphicsResource* cuda_pbo = NULL; // CUDA Graphics Resource (to transfer PBO)

bool g_bFirstTime = true;
unsigned int frame = 0;

// Camera for now

float3  cam_pos = make_float3(-125.394539, 49.180557, 88.074020);
float3  cam_lookat = make_float3(0, 70, 0);
float3  cam_dir;
float3  cam_right;
float3  cam_up;
float3  cam_worldUp = make_float3(0, 1, 0);
float   cam_zoom = 15;
float   cam_lens_radius = 0.f;
float   cam_f = 35.f;
float   cam_d = 50.f;
float   cam_exposure = 1.f;
float   cam_yaw = 5.171003;
float   cam_pitch = -3.352996;
float   cam_xoffset = 0;
float   cam_yoffset = 0;
float   cam_movement_spd = 10;

// Our state
static bool show_demo_window = true;
static bool show_another_window = false;
static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

bool buffer_reset = false;
bool paths_reset = false;
bool cameraWasMoving = false;
int clear_counter = 15;

float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
bool mouseDown = false;
bool click = false;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

// image buffer storing accumulated pixel samples
float4* accumulatebuffer;
int* n_samples;

Paths* paths;
Queues* queues;

float3* d_pMin, *h_pMin;
float3* d_pMax, *h_pMax;
int* d_primitivesOffset, *h_primitivesOffset;
int* d_secondChildOffset, *h_secondChildOffset;
uint16_t* d_nPrimitives, *h_nPrimitives;  // 0 -> interior node
uint8_t* d_axis, *h_axis;

GLFWwindow* window;
const char* glsl_version = "#version 130";