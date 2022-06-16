#pragma once

#include "GLCommon.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "RenderObject.h"
#include "Mesh.h"
#include "Shader.h"
#include "Texture.h"
#include "Transform.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

class Model : public RenderObject
{
public:
    Model(vector<Mesh*> meshes, glm::mat4x4 model_mat);

    void draw(Shader& shader);

    std::vector<Mesh*> get_meshes();

    void set_directory(string dir);
    string get_directory();

private:
    std::vector<Mesh*> meshes;
    string directory;
};

float signed_volume_of_tetrahedron(glm::vec3 a, glm::vec3 b, glm::vec3 c);

glm::vec3 center_of_mass(Model* model);
glm::vec3 center_of_mass(vector<Vertex> vertices, std::vector<unsigned int> indices);
glm::vec3 center_of_mass(vector<glm::vec3> vertices);

glm::vec3 center_of_geometry(vector<Vertex> vertices, std::vector<unsigned int> indices);
glm::vec3 center_of_geometry(vector<glm::vec3> vertices);