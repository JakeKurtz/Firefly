#pragma once

#include "Scene.h"
#include "dCamera.cuh"
#include "dMaterial.cuh"
#include "dVertex.cuh"
#include "dTriangle.cuh"
#include "BVH.h"
#include "dLight.cuh"
#include "dDirectionalLight.cuh"
#include "dEnvironmentLight.cuh"

enum SceneUpdateFlags {
	UpdateScene				= 0x01,
	UpdateCamera			= 0x02,
	UpdateModels			= 0x04,
	UpdateTransforms		= 0x08,
	UpdateMaterials			= 0x10,
	UpdateLights			= 0x20,
	UpdateEnvironmentLight	= 0x40
};

class dScene
{
public:
	dScene(Scene* scene);
	dCamera* get_camera();
	LinearBVHNode* get_nodes();
	dTriangle* get_triangles();
	dLight** get_lights();
	int get_nmb_lights();
	dEnvironmentLight* get_environment_light();
	bool update();

	unsigned char update_flags = 0;

private:
	Scene* h_scene;
	dCamera* d_camera;

	dMaterial** d_material_list;
	std::map<std::string, int>	material_dictionary;

	dTransform** d_transform_list;

	BVHAccel* bvh;
	std::vector<BVHPrimitiveInfo> BVH_triangle_info;
	LinearBVHNode* d_nodes;
	LinearBVHNode* d_nodes_original;
	dTriangle* d_triangles;
	dLight** d_lights;
	dEnvironmentLight* d_environment_light;
	dEnvironmentLight** d_environment_light_old;

	float3* light_directions;
	dMaterial** light_materials;

	int nmb_lights = 0;

	int nmb_dir_lights = 0;
	int nmb_pnt_lights = 0;
	int nmb_area_lights = 0;

	bool BVH_triangle_info_loaded = false;
	bool BVH_initialized = false;
	bool materials_loaded = false;
	bool transforms_loaded = false;
	bool models_loaded = false;
	bool camera_loaded = false;

	void load_scene();
	void load_models();
	void load_transforms();
	void load_materials();
	void load_lights();
	void load_camera();
	void load_nodes();

	void init_BVH_triangle_info();
	void init_BVH();

	void update_camera();
	void update_transforms();
	void update_lights();
	void update_environment_light();
	void update_materials();
	Bounds3f update_bvh(LinearBVHNode* nodes, int current_node_index);
	void update_nodes();
};