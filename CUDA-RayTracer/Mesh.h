#ifndef MESH_H
#define MESH_H

#include "GLCommon.h"

#include "Shader.h"
#include "Texture.h"
#include "Material.h"

#include <vector>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
    glm::vec3 tangent;
    glm::vec3 bitangent;
};

class Mesh {
public:
    Mesh();
    Mesh(std::string name, std::vector<Vertex> vertices, std::vector<unsigned int> indices, Material* material);

    void draw(Shader& shader);

    std::vector<Vertex> get_vertices();
    std::vector<unsigned int> get_indices();
    Material* get_material();
    std::string get_name();
    int get_nmb_of_triangles();

private:
    // mesh Data
    std::string name = "null";
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material* material;
    unsigned int VAO;

    // render data 
    unsigned int VBO, EBO;
    int nmb_triangles = 0;

    // initializes all the buffer objects/arrays
    void setup();
};
#endif