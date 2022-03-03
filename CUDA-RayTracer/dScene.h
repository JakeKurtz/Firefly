#pragma once

#include "Scene.h"
#include "dCamera.cuh"
#include "dMaterial.cuh"
#include "dVertex.cuh"
#include "dTriangle.cuh"
#include "BVH.h"
#include "dLight.cuh"
#include "dDirectionalLight.cuh"

class dScene
{
public:
	dScene(Scene* scene);
	dCamera* get_camera();
	LinearBVHNode* get_nodes();
	dTriangle* get_triangles();
	dLight** get_lights();
	int get_nmb_lights();
	void update();

private:
	Scene* h_scene;
	dCamera* d_camera;

	dMaterial** d_material_list;
	std::map<std::string, int>	material_dictionary;

	BVHAccel* bvh;
	std::vector<BVHPrimitiveInfo> BVH_triangle_info;
	LinearBVHNode* d_nodes;
	dTriangle* d_triangles;
	dLight** d_lights;

	int nmb_lights = 0;

	bool BVH_triangle_info_loaded = false;
	bool BVH_initialized = false;
	bool materials_loaded = false;
	bool models_loaded = false;
	bool camera_loaded = false;

	void load_scene();
	void load_models();
	void load_materials();
	void load_lights();
	void load_camera();
	void load_nodes();

	void init_BVH_triangle_info();
	void init_BVH();

	void update_camera();
	void update_models();
	void update_materials();
};