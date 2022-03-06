#include "dScene.h"
#include "dMath.cuh"
#include "massert.h"
#include "CudaHelpers.h"
#include "dMatrix.cuh"

#define LOG

void reorder_primitives(dTriangle triangles[], int ordered_prims[], int size);
void process_mesh(dVertex vertices[], unsigned int indicies[], dMaterial* materials[], int mat_index, int offset, int size, dTriangle* triangles);
void add_directional_lights(float3* directions, dMaterial** materials, int size, dLight** d_lights);

uint get_mip_map_levels(cudaExtent size)
{
    size_t sz = MAX(MAX(size.width, size.height), size.depth);

    uint levels = 0;

    while (sz)
    {
        sz /= 2;
        levels++;
    }

    return levels;
}

dScene::dScene(Scene* h_scene)
{
	(this)->h_scene = h_scene;
	load_scene();
}

dCamera* dScene::get_camera()
{
    return d_camera;
}

LinearBVHNode* dScene::get_nodes()
{
    return d_nodes;
}

dTriangle* dScene::get_triangles()
{
    return d_triangles;
}

dLight** dScene::get_lights()
{
    return d_lights;
}

int dScene::get_nmb_lights()
{
    return nmb_lights;
}

void dScene::update()
{
    update_camera();
}

void dScene::load_scene()
{
#ifdef LOG
    std::cerr << "(path tracer) Loading scene graph with " << h_scene->get_nmb_of_triangles() << " triangles." << endl;
    clock_t start, stop;
    start = clock();
#endif
    load_materials();
    load_models();
    load_camera();
    load_lights();
    init_BVH_triangle_info();
    init_BVH();
    load_nodes();
#ifdef LOG
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "(path tracer) Scene graph loaded successfully. Took " << timer_seconds << "s." << endl;
#endif
}

void dScene::load_models()
{
    #ifdef LOG
        std::cerr << "\t loading Models." << endl;
    #endif
    // allocate memory on GPU for the scene
    m_assert(materials_loaded, "load_models: materials must be loaded before the models");

    size_t nmb_sceneTriangles = h_scene->get_nmb_of_triangles();
    size_t sizeof_sceneTriangles = sizeof(dTriangle) * nmb_sceneTriangles;
    checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof_sceneTriangles), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_sceneTriangles/1000.f + "Kb)" + " for triangles.");

    int blockSize = 256;
    int start_offest = 0;

    for (Model* model : h_scene->models) {
        #ifdef LOG
            std::cerr << "\t\t file: " << model->get_directory() << endl;
        #endif
        for (Mesh* mesh : model->get_meshes()) {
            #ifdef LOG
                std::cerr << "\t\t\t loading: " << mesh->get_name() << endl;
            #endif
            int nmb_meshTriangles = mesh->get_nmb_of_triangles();
            int numBlocks = (nmb_meshTriangles + blockSize - 1) / blockSize;

            // find material index
            int mat_index = 0;
            auto it = material_dictionary.find(mesh->get_material()->name);

            if (it == material_dictionary.end()) {
                #ifdef LOG
                    std::cerr << "\t\t\t\t failed to find the material \"" << mesh->get_material()->name << "\" in the material list. Set to default material." << endl;
                #endif
                mat_index = 0; // default material
            }
            else {
                #ifdef LOG
                    std::cerr << "\t\t\t\t assigning " << mesh->get_material()->name << "." << endl;
                #endif
                mat_index = it->second;
            }

            // Vertices //

            vector<dVertex> h_vertices;
            dVertex* d_vertices;

            // convert vertices into dVertices
            for (Vertex v : mesh->get_vertices()) {
                dVertex vertex;
                vertex.position = float3_cast(v.position);
                vertex.normal = float3_cast(v.normal);
                vertex.texcoords = float2_cast(v.texCoords);
                vertex.tangent = float3_cast(v.tangent);
                vertex.bitangent = float3_cast(v.bitangent);
                h_vertices.push_back(vertex);
            }

            // allocate/transfer triangle vertices
            size_t sizeof_vertices = sizeof(dVertex) * h_vertices.size();
            checkCudaErrors(cudaMalloc((void**)&d_vertices, sizeof_vertices), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_vertices/1000.f + "kB)" + " for mesh vertices.");
            checkCudaErrors(cudaMemcpy(d_vertices, h_vertices.data(), sizeof_vertices, cudaMemcpyHostToDevice), "CUDA ERROR: failed to copy mesh vertices to device.");

            // Indices //

            vector<unsigned int> h_indices = mesh->get_indices();
            unsigned int* d_indices;

            // allocate/transfer triangle indicies
            size_t sizeof_indices = sizeof(unsigned int) * h_indices.size();;
            checkCudaErrors(cudaMalloc((void**)&d_indices, sizeof_indices), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_indices/1000.f + "kB)" + " for mesh indicies.");
            checkCudaErrors(cudaMemcpy(d_indices, h_indices.data(), sizeof_indices, cudaMemcpyHostToDevice), "CUDA ERROR: failed to copy mesh indicies to device.");

            process_mesh(d_vertices, d_indices, d_material_list, mat_index, start_offest, nmb_meshTriangles, d_triangles);
            checkCudaErrors(cudaGetLastError(), "CUDA ERROR: the kernel \"d_process_mesh\" failed.");
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaFree(d_vertices));
            checkCudaErrors(cudaFree(d_indices));

            start_offest += nmb_meshTriangles;
        }
    }
    #ifdef LOG
        std::cerr << "\t models completed." << endl;
    #endif
    models_loaded = true;
}

void dScene::load_materials()
{
    #ifdef LOG
        std::cerr << "\t loading materials." << endl;
    #endif
    // allocate memory (host)
    int nmb_materials = h_scene->materials_loaded.size();
    size_t sizeof_materials = nmb_materials * sizeof(dMaterial*);
    checkCudaErrors(cudaMallocManaged((void**)&d_material_list, sizeof_materials), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_materials\1000.f + "kB)" + " for material list.");

    // convert mat type and populate list with data
    int i = 0;
    for (auto material : h_scene->materials_loaded)
    {
        #ifdef LOG
                std::cerr << "\t\t loading: " << material.first << std::endl;
        #endif
        material_dictionary.insert(std::pair<string, int>(material.first, i));

        checkCudaErrors(cudaMallocManaged((void**)&(d_material_list[i]), sizeof(dMaterial*)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dMaterial*)\1000.f + "kB)" + " for dMaterial.");
        d_material_list[i]->baseColorFactor = float3_cast(material.second->baseColorFactor);
        d_material_list[i]->roughnessFactor = material.second->roughnessFactor;
        d_material_list[i]->emissiveColorFactor = float3_cast(material.second->emissiveColorFactor);
        d_material_list[i]->ks = 1.f;
        d_material_list[i]->kd = 1.f;
        d_material_list[i]->radiance = 1.f;

        d_material_list[i]->baseColorTexture = load_texture(material.second->baseColorTexture);
        d_material_list[i]->roughnessTexture = load_texture(material.second->roughnessTexture);
        d_material_list[i]->metallicRoughnessTexture = load_texture(material.second->metallicRoughnessTexture);

        i++;
    }
    #ifdef LOG
        std::cerr << "\t materials completed." << endl;
    #endif
    materials_loaded = true;
}

int dScene::load_texture(Texture* tex) 
{
    if (tex != NULL) {
        /*cudaTextureObject_t textureObject;
        cudaMipmappedArray_t mipmapArray;
        auto size = make_cudaExtent(tex->width, tex->height, 0);
        uint levels = get_mip_map_levels(size);

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        checkCudaErrors(cudaMallocMipmappedArray(&mipmapArray, &desc, size, levels));

        cudaArray_t level0;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr(tex->data, size.width * sizeof(uchar4), size.width, size.height);
        copyParams.dstArray = level0;
        copyParams.extent = size;
        copyParams.extent.depth = 1;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        // compute rest of mipmaps based on level 0
        //generateMipMaps(mipmapArray, size);

        // generate bindless texture object

        cudaResourceDesc resDescr;
        memset(&resDescr, 0, sizeof(cudaResourceDesc));

        resDescr.resType = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap = mipmapArray;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.mipmapFilterMode = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;
        texDescr.addressMode[2] = cudaAddressModeWrap;

        texDescr.maxMipmapLevelClamp = float(levels - 1);

        texDescr.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&textureObject, &resDescr, &texDescr, NULL));
        return textureObject;
        */
        return -1;
    }
    else {
        return -1;
    }
}

void dScene::load_lights()
{
#ifdef LOG
    std::cerr << "\t loading lights." << endl;
#endif

    float3* directions;
    dMaterial** materials;

    // allocate memory (host)
    nmb_lights = h_scene->get_lights().size();

    size_t sizeof_directions = nmb_lights * sizeof(float3);
    size_t sizeof_materials = nmb_lights * sizeof(dMaterial*);
    size_t sizeof_lights = nmb_lights * sizeof(dLight*);

    checkCudaErrors(cudaMalloc((void**)&d_lights, sizeof_lights), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_lights / 1000.f + "Kb)" + " for light list.");
    checkCudaErrors(cudaMallocManaged((void**)&directions, sizeof_directions), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_directions\1000.f + "kB)" + " for direction list.");
    checkCudaErrors(cudaMallocManaged((void**)&materials, sizeof_materials), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_materials\1000.f + "kB)" + " for material list.");

    int i = 0;
    for (auto light : h_scene->get_lights())
    {
        checkCudaErrors(cudaMallocManaged((void**)&(materials[i]), sizeof(dMaterial*)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dMaterial*)\1000.f + "kB)" + " for dMaterial.");
        materials[i]->baseColorFactor = make_float3(0.f);
        materials[i]->emissiveColorFactor = make_float3(0.f);
        materials[i]->ks = 0.f;
        materials[i]->kd = 0.f;
        materials[i]->radiance = light->getIntensity();
        materials[i]->emissiveColorFactor = float3_cast(light->getColor());

        checkCudaErrors(cudaMallocManaged((void**)&(directions[i]), sizeof(float3)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(float3)\1000.f + "kB)" + " for float3.");
        directions[i] = float3_cast(light->getDirection());

        i++;
    }

    add_directional_lights(directions, materials, h_scene->get_lights().size(), d_lights);
}

void dScene::load_camera()
{
    d_camera = new dCamera();
    checkCudaErrors(cudaMallocManaged(&d_camera, sizeof(dCamera)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dCamera)\1000.f + "kB)" + " for camera.");
    camera_loaded = true;
    update_camera();
}

void dScene::load_nodes()
{
    m_assert(BVH_initialized, "load_nodes:");

    size_t node_size = sizeof(LinearBVHNode) * bvh->get_nmb_nodes();
    checkCudaErrors(cudaMalloc((void**)&d_nodes, node_size));
    checkCudaErrors(cudaMemcpy(d_nodes, bvh->get_nodes(), node_size, cudaMemcpyHostToDevice));

    int* d_ordered_prims;
    size_t ordered_prims_size = sizeof(int) * bvh->get_ordered_prims().size();
    checkCudaErrors(cudaMalloc((void**)&d_ordered_prims, ordered_prims_size));
    checkCudaErrors(cudaMemcpy(d_ordered_prims, bvh->get_ordered_prims().data(), ordered_prims_size, cudaMemcpyHostToDevice));

    reorder_primitives(d_triangles, d_ordered_prims, h_scene->get_nmb_of_triangles());
}

void dScene::init_BVH_triangle_info()
{
#ifdef LOG
    std::cerr << "\t gathering triangle information for BVH construction: ";
    clock_t start, stop;
    start = clock();
#endif
    int start_offest = 0;
    //size_t id = 0;
    for (Model* model : h_scene->models) {
        for (Mesh* mesh : model->get_meshes()) {
            // Construct triangle info list for BVH
            size_t id = start_offest;

            auto indices = mesh->get_indices();
            auto vertices = mesh->get_vertices();

            for (size_t i = 0; i < indices.size(); i += 3) {
                unsigned int iv0 = indices[i];
                unsigned int iv1 = indices[i + 1];
                unsigned int iv2 = indices[i + 2];

                float3 v0 = float3_cast(vertices[iv0].position);
                float3 v1 = float3_cast(vertices[iv1].position);
                float3 v2 = float3_cast(vertices[iv2].position);

                Bounds3f bounds = Union(Bounds3f(v0, v1), v2);
                BVH_triangle_info.push_back({ id++, bounds });
            }
            start_offest += mesh->get_nmb_of_triangles();
        }
    }
#ifdef LOG
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << timer_seconds << "s" << endl;
#endif
    BVH_triangle_info_loaded = true;
}

void dScene::init_BVH()
{
    m_assert(BVH_triangle_info_loaded, "init_BVH: BVH triangle info wasn't loaded.");
    #ifdef LOG
            std::cerr << "\t building BVH: ";
            clock_t start, stop;
            start = clock();
    #endif
    bvh = new BVHAccel(BVH_triangle_info, SplitMethod::SAH, 8);
    #ifdef LOG
            stop = clock();
            double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
            std::cerr << timer_seconds << "s" << endl;
    #endif
    BVH_initialized = true;
}

void dScene::update_camera()
{
    m_assert(camera_loaded, "update_camera:");
    d_camera->position = float3_cast(h_scene->camera->position);
    d_camera->zoom = h_scene->camera->zoom;
    d_camera->lens_radius = h_scene->camera->lens_radius;
    d_camera->f = h_scene->camera->focal_distance;
    d_camera->d = 1.f;
    d_camera->exposure_time = h_scene->camera->exposure;
    d_camera->inv_view_proj_mat = Matrix4x4_cast(glm::inverse(h_scene->camera->proj_mat * h_scene->camera->lookat_mat));
}
