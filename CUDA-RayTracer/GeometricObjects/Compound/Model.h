#ifndef MODEL_H
#define MODEL_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Mesh.h"

#define AI_CONFIG_PP_RVC_FLAGS aiComponent_NORMALS

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

class Model
{
public:
    vector<Mesh> meshes;
    string directory;
    bool gammaCorrection;
    std::vector<tuple<std::vector<BVHPrimitiveInfo>, Triangle*, int>> processed_meshes;

    Model(string const& path, bool gamma = false) 
        : gammaCorrection(gamma)
    {
        loadModel(path);
    }

private:
    // loads a model with supported ASSIMP extensions from file and stores the resulting meshes in the meshes vector.
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
        /*
        for (Mesh mesh : meshes)
        {
            size_t nmb_indicies = mesh.indices.size();
            size_t nmb_triangles = nmb_indicies / 3;

            Triangle* d_triangles;
            Triangle* h_triangles = (Triangle*)malloc(nmb_triangles * sizeof(Triangle));

            std::vector<BVHPrimitiveInfo> triangle_info;

            std::cerr << "Loading " << nmb_triangles << " triangles... ";

            clock_t start, stop;
            start = clock();

            h_process_mesh(mesh.vertices, mesh.indices, triangle_info);

            int blockSize = 256;
            int numBlocks = (nmb_triangles + blockSize - 1) / blockSize;

            size_t sizeof_triangles = sizeof(Triangle) * nmb_triangles;
            checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof_triangles));
            checkCudaErrors(cudaMemcpy(d_triangles, h_triangles, sizeof_triangles, cudaMemcpyHostToDevice));

            d_process_mesh << < numBlocks, blockSize >> > (mesh.d_vertices, mesh.d_indices, nmb_triangles, d_triangles);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            CookTorrence** d_material_ptr;
            checkCudaErrors(cudaMalloc((void**)&d_material_ptr, sizeof(CookTorrence*)));

            init_mesh_material <<< 1, 1 >>> (d_material_ptr, vec3(0.94901, 0.94117, 0.90196), 0.2, 1.f, 1.f, 0.01f);
            //init_mesh_material <<< 1, 1 >>> (d_material_ptr, vec3(0.f, 0.6, 0.f), 0.2, 1.f, 1.f, 0.01);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            set_mesh_material <<< numBlocks, blockSize >>> (d_triangles, nmb_triangles, d_material_ptr);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            stop = clock();
            double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
            std::cerr << "took " << timer_seconds << " seconds.\n\n";

            processed_meshes.push_back(std::make_tuple(triangle_info, d_triangles, nmb_triangles));
        }*/
        /*
        // Figure out how much memory we need
        size_t nmb_indicies = 0;
        std::vector<tuple<Mesh, int, int>> mesh_info; // mesh, offset, number of triangles
        for (Model model : models) {
            for (Mesh mesh : model.meshes)
            {
                int offset = (nmb_indicies / 3);
                int num_triangles = mesh.indices.size() / 3;
                mesh_info.push_back(std::make_tuple(mesh, offset, num_triangles));

                nmb_indicies += mesh.indices.size();
            }
        }

        // Load and setup meshes on the GPU
        size_t nmb_triangles = nmb_indicies / 3;
        Triangle* d_triangles;
        Triangle* h_triangles = (Triangle*)malloc(nmb_triangles * sizeof(Triangle));
        std::vector<BVHPrimitiveInfo> triangle_info;

        std::cerr << "Loading " << nmb_triangles << " triangles... ";

        clock_t start, stop;
        start = clock();

        for (auto info : mesh_info)
        {

            Mesh mesh           = std::get<0>(info);
            int offset          = std::get<1>(info);
            int nmb_triangles   = std::get<2>(info);

            // Construct triangle info list for BVH
            h_process_mesh(mesh.vertices, mesh.indices, triangle_info);

            int blockSize = 256;
            int numBlocks = (nmb_triangles + blockSize - 1) / blockSize;

            size_t sizeof_triangles = sizeof(Triangle) * nmb_triangles;
            checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof_triangles));
            checkCudaErrors(cudaMemcpy(d_triangles, h_triangles, sizeof_triangles, cudaMemcpyHostToDevice));

            // Create triangles in GPU memory
            d_process_mesh << < numBlocks, blockSize >> > (mesh.d_vertices, mesh.d_indices, nmb_triangles, d_triangles);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            CookTorrence** d_material_ptr;
            checkCudaErrors(cudaMalloc((void**)&d_material_ptr, sizeof(CookTorrence*)));

            //init_mesh_material << < 1, 1 >> > (d_material_ptr, vec3(0.94901, 0.94117, 0.90196), 0.2, 1.f, 1.f, 0.01f);
            //init_mesh_material <<< 1, 1 >>> (d_material_ptr, vec3(0.f, 0.6, 0.f), 0.2, 1.f, 1.f, 0.01);
            //init_mesh_material <<< 1, 1 >>> (d_material_ptr, mesh.Cd, mesh.Ka, mesh.Kd, mesh.Ks, mesh.Pr);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // Set mesh material
            set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            //processed_meshes.push_back(std::make_tuple(triangle_info, d_triangles, nmb_triangles));
        }

        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n\n";
        */
    };

    // processes a node in a recursive fashion. Processes each individual mesh located at the node and repeats this process on its children nodes (if any).
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

        // walk through each of the mesh's vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            Vertex vertex;
            glm::vec3 vector;

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
                glm::vec2 vec;
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
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);

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
        //aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        // we assume a convention for sampler names in the shaders. Each diffuse texture should be named
        // as 'texture_diffuseN' where N is a sequential number ranging from 1 to MAX_SAMPLER_NUMBER. 
        // Same applies to other texture as the following list summarizes:
        // diffuse: texture_diffuseN
        // specular: texture_specularN
        // normal: texture_normalN

        // 1. diffuse maps
        //vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
        //textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
        // 2. specular maps
        //vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
        //textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
        // 3. normal maps
        //std::vector<Texture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
        //textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
        // 4. height maps
        //std::vector<Texture> heightMaps = loadMaterialTextures(material, aiTextureType_AMBIENT, "texture_height");
        //textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());

        // return a mesh object created from the extracted mesh data
        return Mesh(vertices, indices);
    }
    // checks all material textures of a given type and loads the textures if they're not loaded yet.
    // the required info is returned as a Texture struct.
    /*vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, string typeName)
    {
        vector<Texture> textures;
        for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
        {
            aiString str;
            mat->GetTexture(type, i, &str);
            // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
            bool skip = false;
            for (unsigned int j = 0; j < textures_loaded.size(); j++)
            {
                if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
                {
                    textures.push_back(textures_loaded[j]);
                    skip = true; // a texture with the same filepath has already been loaded, continue to next one. (optimization)
                    break;
                }
            }
            if (!skip)
            {   // if texture hasn't been loaded already, load it
                Texture texture(str.C_Str(), this->directory, typeName, D2D);
                //texture.id = TextureFromFile(str.C_Str(), this->directory);
                //texture.type = typeName;
                //texture.path = str.C_Str();
                textures.push_back(texture);
                textures_loaded.push_back(texture);  // store it as texture loaded for entire model, to ensure we won't unnecesery load duplicate textures.
            }
        }
        return textures;
    }*/
};

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

        Bounds3f bounds = bounds = Union(Bounds3f((vec3)v0.Position, (vec3)v1.Position), (vec3)v2.Position);

        triangle_info.push_back({ id++, bounds });
    }
    return primitive_bounds;
}

__global__ void init_mesh_material(CookTorrence** material_ptr, vec3 Cd, float Ka, float Kd, float Ks, float Pr, vec3 Fr)
{
    (*material_ptr) = new CookTorrence();
    (*material_ptr)->set_cd(Cd);
    (*material_ptr)->set_ka(Ka);
    (*material_ptr)->set_kd(Kd);
    (*material_ptr)->set_ks(Ks);
    (*material_ptr)->set_roughness(Pr);
    (*material_ptr)->set_fresnel_reflectance(Fr);
}

__global__ void set_mesh_material(Triangle* triangles, size_t nmb_triangles, CookTorrence** material_ptr, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < nmb_triangles; i += stride) {
        triangles[i + offset].set_material(*material_ptr) ;
    }
}

Triangle* loadModels(std::vector<Model*> models, std::vector<BVHPrimitiveInfo>& triangle_info, int& nmb_triangles)
{
    // Figure out how much memory we need
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
    Triangle* d_triangles;
    Triangle* h_triangles = (Triangle*)malloc(nmb_triangles * sizeof(Triangle));

    size_t sizeof_triangles = sizeof(Triangle) * nmb_triangles;
    checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof_triangles));
    checkCudaErrors(cudaMemcpy(d_triangles, h_triangles, sizeof_triangles, cudaMemcpyHostToDevice));

    std::cerr << "Loading " << nmb_triangles << " triangles... ";

    clock_t start, stop;
    start = clock();

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

        CookTorrence** d_material_ptr;
        checkCudaErrors(cudaMalloc((void**)&d_material_ptr, sizeof(CookTorrence*)));

        //init_mesh_material << < 1, 1 >> > (d_material_ptr, vec3(0.94901, 0.94117, 0.90196), 0.2, 1.f, 1.f, 0.01f);
        if (nmb_triangles > 2)
            init_mesh_material <<< 1, 1 >>> (d_material_ptr, vec3(1,0.2,0), 0.2, 1.f, 1.f, 0.5f, vec3(1.00, 0.86, 0.57));
        else
            init_mesh_material <<< 1, 1 >>> (d_material_ptr, vec3(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX), 0.2, 1.f, 1.f, 0.01f, vec3(1.f));
        //init_mesh_material <<< 1, 1 >>> (d_material_ptr, vec3(0.f, 0.6, 0.f), 0.2, 1.f, 1.f, 10.f);
        //init_mesh_material <<< 1, 1 >>> (d_material_ptr, mesh.Cd, mesh.Ka, mesh.Kd, mesh.Ks, mesh.Pr);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Set mesh material
        set_mesh_material << < numBlocks, blockSize >> > (d_triangles, nmb_triangles, d_material_ptr, offset);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        //processed_meshes.push_back(std::make_tuple(triangle_info, d_triangles, nmb_triangles));
    }

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n\n";

    return d_triangles;
}


#endif