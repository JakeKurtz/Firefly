#ifndef MESH_H
#define MESH_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <vector>

#include "../../Utilities/CudaHelpers.h"

using namespace std;

struct Vertex {
    float3 Position;
    float3 Normal;
    float2 TexCoords;
    float3 Tangent;
    float3 Bitangent;
};

//struct Texture {
//    unsigned int id;
//    string type;
//    string path;
//};

class Mesh {
public:
    vector<Vertex>          vertices;
    vector<unsigned int>    indices;

    Vertex*                 d_vertices;
    unsigned int*           d_indices;

    Mesh(vector<Vertex> vertices, vector<unsigned int> indices)
    {
        this->vertices = vertices;
        this->indices = indices;

        //setup_mesh();
    }

private:

    void setup_mesh()
    {
        size_t sizeof_vertices = sizeof(Vertex) * vertices.size();
        checkCudaErrors(cudaMalloc((void**)&d_vertices, sizeof_vertices));
        checkCudaErrors(cudaMemcpy(d_vertices, vertices.data(), sizeof_vertices, cudaMemcpyHostToDevice));

        size_t sizeof_indices = sizeof(unsigned int) * indices.size();
        checkCudaErrors(cudaMalloc((void**)&d_indices, sizeof_indices));
        checkCudaErrors(cudaMemcpy(d_indices, indices.data(), sizeof_indices, cudaMemcpyHostToDevice));
    }
};
#endif