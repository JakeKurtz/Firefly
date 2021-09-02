#ifndef MESH_H
#define MESH_H

#include <string>
#include <vector>
#include "Triangle.cuh"
#include "CudaHelpers.cuh"

using namespace std;

class Mesh {
public:
    vector<Vertex>          vertices;
    vector<unsigned int>    indices;
    vector<int>             texture_ids;
    Material*                material;

    Vertex* d_vertices;
    unsigned int* d_indices;

    Mesh(vector<Vertex> vertices, vector<unsigned int> indices, Material* material)
    {
        this->vertices = vertices;
        this->indices = indices;
        this->material = material;

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