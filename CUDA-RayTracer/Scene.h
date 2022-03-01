#pragma once
#include "GLCommon.h"

#include "Model.h"
#include "Camera.h"
#include "EnvironmentLight.h"
#include "PointLight.h"
#include "DirectionalLight.h"

class Scene
{
public:
	vector<Model*> models;
	vector<PointLight*> point_lights;
	vector<DirectionalLight*> dir_lights;
	EnvironmentLight* environment_light;
	Camera* camera = nullptr;

	vector<Texture*> textures_loaded;

	std::map<std::string, Material*> materials_loaded;

	string directory;

	int nmb_triangles = 0;

	Scene();
	Scene(vector<Model*> models, vector<Light*> lights, Camera* camera);
	void load(string const& path, glm::vec3 translate = glm::vec3(0), glm::vec3 scale = glm::vec3(1));
	void set_camera(Camera* camera);
	void add_light(PointLight* light);
	void add_light(DirectionalLight* light);
	void set_environment_light(EnvironmentLight* _environmentLight);
	void send_uniforms(Shader& shader);
	void bind_environment_textures(Shader& shader);

	int get_nmb_of_triangles();

private:
	void load_model(vector<Mesh*>& meshes, aiMatrix4x4 model_mat, aiNode* node, const aiScene* scene);
	Light* load_light(aiLight* light, const aiScene* scene);
	Mesh* load_mesh(aiMesh* mesh, aiMatrix4x4 accTransform, const aiScene* scene);
	Material* load_material(aiMaterial* material, const aiScene* scene);
	Texture* load_texture(aiTextureType type, string typeName, aiMaterial* mat, const aiScene* scene);
	Camera* load_camera(aiCamera* camera, const aiScene* scene);
};

