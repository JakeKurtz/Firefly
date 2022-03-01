#pragma once

#include "GLCommon.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Mesh.h"
#include "Shader.h"
#include "Texture.h"
#include "Geometry.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

class Model : public Geometry
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