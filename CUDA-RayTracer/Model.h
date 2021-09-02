#ifndef MODEL_H
#define MODEL_H

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/pbrmaterial.h>

#define AI_CONFIG_PP_RVC_FLAGS aiComponent_NORMALS

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

#include "BVH.cuh"
#include "Mesh.h"
#include "Texture.cuh"
#include "cudaTexture.cuh"

using namespace std;

__global__ void d_process_mesh(
    Vertex* in_vertices, unsigned int* in_indicies,
    size_t nmb_triangles, Triangle* triangles, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < nmb_triangles; i += stride) {
        int j = i * 3;

        unsigned int iv0 = in_indicies[j];
        unsigned int iv1 = in_indicies[j + 1];
        unsigned int iv2 = in_indicies[j + 2];

        Vertex v0 = in_vertices[iv0];
        Vertex v1 = in_vertices[iv1];
        Vertex v2 = in_vertices[iv2];

        new (&triangles[i + offset]) Triangle(v0, v1, v2);
    }
}

__global__ void init_mesh_material(Material** material_ptr, float3 baseColor, float3 fresnel, float roughness, float kd, float ks)
{
    (*material_ptr) = new Material();
    (*material_ptr)->baseColor = baseColor;
    (*material_ptr)->fresnel = fresnel;
    (*material_ptr)->roughness = roughness;
    (*material_ptr)->kd = kd;
    (*material_ptr)->ks = ks;
}

__global__ void set_mesh_material(Triangle* triangles, size_t nmb_triangles, Material* material_ptr, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < nmb_triangles; i += stride) {
        triangles[i + offset].material_ptr = material_ptr;
    }
}

__host__ std::vector<Bounds3f> h_process_mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<BVHPrimitiveInfo>& triangle_info, int offset)
{
    std::vector<Bounds3f> primitive_bounds;
    size_t id = offset;
    for (size_t i = 0; i < indices.size(); i += 3) {
        unsigned int iv0 = indices[i];
        unsigned int iv1 = indices[i + 1];
        unsigned int iv2 = indices[i + 2];

        Vertex v0 = vertices[iv0];
        Vertex v1 = vertices[iv1];
        Vertex v2 = vertices[iv2];

        Bounds3f bounds = Union(Bounds3f(v0.Position, v1.Position), v2.Position);
        triangle_info.push_back({ id++, bounds });
    }
    return primitive_bounds;
}

/*Triangle* loadModels(std::vector<Model*> models, std::vector<BVHPrimitiveInfo>& triangle_info, int& nmb_triangles)
{
    size_t nmb_indicies = 0;
    std::vector<tuple<Mesh, int, int>> mesh_info; // mesh, offset, number of triangles
    for (Model* model : models) {
        for (Mesh mesh : model->meshes)
        {
            int offset = (nmb_indicies / 3);
            int num_triangles = mesh.indices.size() / 3;
            mesh_info.push_back(std::make_tuple(mesh, offset, num_triangles));

            nmb_indicies += mesh.indices.size();
        }
    }

    // Allocate memory on GPU
    nmb_triangles = nmb_indicies / 3;
    size_t sizeof_triangles = sizeof(Triangle) * nmb_triangles;

    Triangle* d_triangles;
    checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof_triangles));

    std::cerr << "Loading " << nmb_triangles << " triangles... ";
    clock_t start, stop;
    start = clock();

    int index = 0;

    for (auto info : mesh_info)
    {
        Mesh mesh = std::get<0>(info);
        int offset = std::get<1>(info);
        int nmb_triangles = std::get<2>(info);

        // Construct triangle info list for BVH
        h_process_mesh(mesh.vertices, mesh.indices, triangle_info, offset);

        int blockSize = 256;
        int numBlocks = (nmb_triangles + blockSize - 1) / blockSize;

        // Create triangles in GPU memory
        Vertex* d_vertices;
        unsigned int* d_indices;

        size_t sizeof_vertices = sizeof(Vertex) * mesh.vertices.size();
        checkCudaErrors(cudaMalloc((void**)&d_vertices, sizeof_vertices));
        checkCudaErrors(cudaMemcpy(d_vertices, mesh.vertices.data(), sizeof_vertices, cudaMemcpyHostToDevice));

        size_t sizeof_indices = sizeof(unsigned int) * mesh.indices.size();;
        checkCudaErrors(cudaMalloc((void**)&d_indices, sizeof_indices));
        checkCudaErrors(cudaMemcpy(d_indices, mesh.indices.data(), sizeof_indices, cudaMemcpyHostToDevice));

        d_process_mesh << < numBlocks, blockSize >> > (d_vertices, d_indices, nmb_triangles, d_triangles, offset);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(d_vertices));
        checkCudaErrors(cudaFree(d_indices));

        Material** d_material_ptr;
        checkCudaErrors(cudaMalloc((void**)&d_material_ptr, sizeof(Material*)));

        // hard coded materials cuz im lazy and possibly stupid :p //

        if (index == 0)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(0.2), make_float3(0), 0.01f, 1, 0); // Ground
        else if (index == 1)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(0.72, 0.43, 0.47), make_float3(0.21), 0.1, 0.5, 0.5);
        else if (index == 2)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(0.2), make_float3(0), 1, 1, 0);
        else if (index == 3)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(0.2), make_float3(0), 1, 1, 0);
        else if (index == 4)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(0.2), make_float3(0), 1, 1, 0);
        else if (index == 5)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(0.2), make_float3(0), 1, 1, 0);
        else if (index == 6)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(1, 0, 1), make_float3(1.f), 0.01f, 0, 1);
        else if (index == 7)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(1, 0.1, 0.1), make_float3(0.05f), 0.01f, 0.5, 0.5);
        else if (index == 8)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(0.1, 1, 0.1), make_float3(1.f), 1.f, 1, 0);
        else if (index == 9)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(0.1, 0.1, 1), make_float3(1), 0.1f, 0.5, 0.5);
        else if (index == 10)
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(1), make_float3(1.f), 0.25f, 0, 1);
        else
            init_mesh_material << < 1, 1 >> > (d_material_ptr, make_float3(1), make_float3(1.f), 0.5f, 0, 1);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Set mesh material
        //set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, 0, offset);

        if (index == 0)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Diffuse, offset);
        else if (index == 1)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Mix, offset);
        else if (index == 2)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Diffuse, offset);
        else if (index == 3)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Diffuse, offset);
        else if (index == 4)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Diffuse, offset);
        else if (index == 5)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Diffuse, offset);
        else if (index == 6)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::CookTor, offset);
        else if (index == 7)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Mix, offset);
        else if (index == 8)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Diffuse, offset);
        else if (index == 9)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::Mix, offset);
        else if (index == 10)
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::CookTor, offset);
        else
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, MaterialIndex::CookTor, offset);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        index++;
    }

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n\n";

    return d_triangles;
}
*/

class Model
{
public:
    vector<Mesh> meshes;
    string directory;
    bool gammaCorrection;

    Triangle* d_triangles;
    vector<Triangle> h_triangles;
    vector<BVHPrimitiveInfo> triangle_info;

    vector<Texture> h_albedo_maps;
    vector<Texture> h_specular_maps;
    vector<Texture> h_normal_maps;
    vector<Texture> h_bump_maps;

    vector<cudaTexture> d_albedo_maps;
    vector<cudaTexture> d_specular_maps;
    vector<cudaTexture> d_normal_maps;
    vector<cudaTexture> d_bump_maps;

    Model(string const& path, bool gamma = false)
        : gammaCorrection(gamma)
    {
        loadModel(path);
    }

private:

    void loadModel(string const& path)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_RemoveComponent | aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
        {
            cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
            return;
        }

        directory = path.substr(0, path.find_last_of('/'));

        processNode(scene->mRootNode, scene);
    };

    void processNode(aiNode* node, const aiScene* scene)
    {
        // process each mesh located at the current node
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            // the node object only contains indices to index the actual objects in the scene. 
            // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }
        // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            processNode(node->mChildren[i], scene);
        }
    }

    Mesh processMesh(aiMesh* mesh, const aiScene* scene)
    {
        // data to fill
        vector<Vertex> vertices;
        vector<unsigned int> indices;
        Material* material_foobar = new Material();

        // walk through each of the mesh's vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            Vertex vertex;
            float3 vector;

            // positions
            vector.x = mesh->mVertices[i].x;
            vector.y = mesh->mVertices[i].y;
            vector.z = mesh->mVertices[i].z;
            vertex.Position = vector;

            // normals
            if (mesh->HasNormals())
            {
                vector.x = mesh->mNormals[i].x;
                vector.y = mesh->mNormals[i].y;
                vector.z = mesh->mNormals[i].z;
                vertex.Normal = vector;
            }
            // texture coordinates
            if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
            {
                float2 vec;
                vec.x = mesh->mTextureCoords[0][i].x;
                vec.y = mesh->mTextureCoords[0][i].y;
                vertex.TexCoords = vec;

                vector.x = mesh->mTangents[i].x;
                vector.y = mesh->mTangents[i].y;
                vector.z = mesh->mTangents[i].z;
                vertex.Tangent = vector;

                vector.x = mesh->mBitangents[i].x;
                vector.y = mesh->mBitangents[i].y;
                vector.z = mesh->mBitangents[i].z;
                vertex.Bitangent = vector;
            }
            else
                vertex.TexCoords = make_float2(0.0f, 0.0f);

            vertices.push_back(vertex);
        }
        // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];

            // retrieve all indices of the face and store them in the indices vector
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }

        // process materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        aiColor3D Kd(0.f, 0.f, 0.f);
        material->Get(AI_MATKEY_COLOR_DIFFUSE, Kd);
        material_foobar->baseColor = make_float3(Kd.r, Kd.g, Kd.b);

        aiColor3D F(0.f, 0.f, 0.f);
        material->Get(AI_MATKEY_COLOR_SPECULAR, F);        
        material_foobar->fresnel = make_float3(F.r, F.g, F.b);

        // Ns
        float Pm = 0.f;
        material->Get(AI_MATKEY_OPACITY, Pm);
        material_foobar->kd = (1.f - Pm);
        // Ns
        float Ks = 1.f;
        material->Get(AI_MATKEY_SHININESS, Ks);
        material_foobar->ks = Ks;
        // Ni
        float Pr = 1.f;
        material->Get(AI_MATKEY_REFRACTI, Pr);
        material_foobar->roughness = Pr;

        aiString name;
        MaterialIndex material_index;
        material->Get(AI_MATKEY_NAME, name);

        int shadingModel;
        material->Get(AI_MATKEY_SHADING_MODEL, shadingModel);

        if (shadingModel == 2) material_foobar->materialIndex = MaterialIndex::Mix;
        if (shadingModel == 3) material_foobar->materialIndex = MaterialIndex::Diffuse;
        //else material_foobar->materialIndex = MaterialIndex::CookTor;

        // 1. diffuse maps
        material_foobar->albedo_tex_id = loadMaterialTextures(material, aiTextureType_DIFFUSE, h_albedo_maps, d_albedo_maps);
        // 2. specular maps
        material_foobar->specular_tex_id = loadMaterialTextures(material, aiTextureType_SPECULAR, h_specular_maps, d_specular_maps);
        // 3. normal maps
        material_foobar->normal_tex_id = loadMaterialTextures(material, aiTextureType_HEIGHT, h_normal_maps, d_normal_maps);
        // 4. height maps
        material_foobar->bump_tex_id = loadMaterialTextures(material, aiTextureType_AMBIENT, h_bump_maps, d_bump_maps);

        return Mesh(vertices, indices, material_foobar);
    }

    int loadMaterialTextures(aiMaterial* mat, aiTextureType type, vector<Texture>& h_textures, vector<cudaTexture>& d_textures)
    {
        if (mat->GetTextureCount(type) == 1)
        {
            aiString str;
            mat->GetTexture(type, 0, &str);

            //Texture h_texture(str.C_Str(), this->directory);
            //h_textures.push_back(h_texture);

            cudaTexture d_texture(str.C_Str(), this->directory);
            d_textures.push_back(d_texture);

            return d_texture.textureObject;
        }
        else {
            return -1;
        }
    }
};

Triangle* loadModels(std::vector<Model*> models, std::vector<BVHPrimitiveInfo>& triangle_info, int& nmb_triangles)
{
    size_t nmb_indicies = 0;
    std::vector<tuple<Mesh, int, int>> mesh_info; // mesh, offset, number of triangles
    for (Model* model : models) {
        for (Mesh mesh : model->meshes)
        {
            int offset = (nmb_indicies / 3);
            int num_triangles = mesh.indices.size() / 3;
            mesh_info.push_back(std::make_tuple(mesh, offset, num_triangles));

            nmb_indicies += mesh.indices.size();
        }
    }

    // Allocate memory on GPU
    nmb_triangles = nmb_indicies / 3;
    size_t sizeof_triangles = sizeof(Triangle) * nmb_triangles;

    Triangle* d_triangles;
    checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof_triangles));

    std::cerr << "Loading " << nmb_triangles << " triangles... ";
    clock_t start, stop;
    start = clock();

    int index = 0;

    for (auto info : mesh_info)
    {
        Mesh mesh = std::get<0>(info);
        int offset = std::get<1>(info);
        int nmb_triangles = std::get<2>(info);

        // Construct triangle info list for BVH
        h_process_mesh(mesh.vertices, mesh.indices, triangle_info, offset);

        int blockSize = 256;
        int numBlocks = (nmb_triangles + blockSize - 1) / blockSize;

        // Create triangles in GPU memory
        Vertex* d_vertices;
        unsigned int* d_indices;

        size_t sizeof_vertices = sizeof(Vertex) * mesh.vertices.size();
        checkCudaErrors(cudaMalloc((void**)&d_vertices, sizeof_vertices));
        checkCudaErrors(cudaMemcpy(d_vertices, mesh.vertices.data(), sizeof_vertices, cudaMemcpyHostToDevice));

        size_t sizeof_indices = sizeof(unsigned int) * mesh.indices.size();;
        checkCudaErrors(cudaMalloc((void**)&d_indices, sizeof_indices));
        checkCudaErrors(cudaMemcpy(d_indices, mesh.indices.data(), sizeof_indices, cudaMemcpyHostToDevice));

        d_process_mesh << < numBlocks, blockSize >> > (d_vertices, d_indices, nmb_triangles, d_triangles, offset);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(d_vertices));
        checkCudaErrors(cudaFree(d_indices));

        Material* d_material_ptr;
        checkCudaErrors(cudaMalloc((void**)&d_material_ptr, sizeof(Material)));
        checkCudaErrors(cudaMemcpy(d_material_ptr, mesh.material, sizeof(Material), cudaMemcpyHostToDevice));

        set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, offset);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        index++;
    }

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n\n";

    return d_triangles;
}

#endif
