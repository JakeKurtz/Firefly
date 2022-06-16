#include "dScene.h"
#include "dMath.cuh"
#include "massert.h"
#include "CudaHelpers.h"
#include "dMatrix.cuh"
#include "Rectangle.cuh"

#include <stb_image.h>
#include "dTexture.cuh"

#define LOG

void reorder_primitives(dTriangle triangles[], int ordered_prims[], int size);
void process_mesh(dVertex vertices[], unsigned int indicies[], dTransform* transforms[], int trans_index, dMaterial* materials[], int mat_index, int offset, int size, dTriangle* triangles);
void update_bvh(int size, dTriangle* triangles);
void add_directional_lights(float3* directions, dMaterial** materials, int size, dLight** d_lights);
void add_area_lights(_Rectangle** objs, dMaterial** materials, int size, dLight** d_lights);
void add_environment_light(cudaTextureObject_t hrd_texture, int tex_width, int tex_height, float3 color, dEnvironmentLight* d_environment_light, dLight** d_lights);

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

dEnvironmentLight* dScene::get_environment_light()
{
    return d_environment_light;
}

bool dScene::update()
{
    bool ret = (update_flags != 0);

    if (update_flags & UpdateScene) {
        load_scene();
    }
    else {
        if (update_flags & UpdateCamera) update_camera();
        if (update_flags & UpdateMaterials) update_materials();
        if (update_flags & UpdateLights) update_lights();
        if (update_flags & UpdateEnvironmentLight) update_environment_light();
        if (update_flags & UpdateTransforms) update_transforms();
    }
    update_flags = 0;

    return ret;
}

void dScene::load_scene()
{
#ifdef LOG
    std::cerr << "(path tracer) Loading scene graph with " << h_scene->get_nmb_of_triangles() << " triangles." << endl;
    clock_t start, stop;
    start = clock();
#endif
    load_materials();
    load_transforms();
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
    m_assert(transforms_loaded, "load_models: transforms must be loaded before the models");

    size_t nmb_sceneTriangles = h_scene->get_nmb_of_triangles();
    size_t sizeof_sceneTriangles = sizeof(dTriangle) * nmb_sceneTriangles;
    checkCudaErrors(cudaMallocManaged((void**)&d_triangles, sizeof_sceneTriangles), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_sceneTriangles/1000.f + "Kb)" + " for triangles.");

    int blockSize = 256;
    int start_offest = 0;
    int trans_index = 0;
    for (RenderObject* r_obj : h_scene->models) {
        Model* model = dynamic_cast<Model*>(r_obj);

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

            process_mesh(d_vertices, d_indices, d_transform_list, trans_index, d_material_list, mat_index, start_offest, nmb_meshTriangles, d_triangles);
            checkCudaErrors(cudaGetLastError(), "CUDA ERROR: the kernel \"d_process_mesh\" failed.");
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaFree(d_vertices));
            checkCudaErrors(cudaFree(d_indices));

            start_offest += nmb_meshTriangles;
        }
        trans_index++;
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
    checkCudaErrors(cudaMallocManaged((void**)&d_material_list, sizeof_materials), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_materials/1000.f + "kB)" + " for material list.");

    // convert mat type and populate list with data
    int i = 0;
    for (auto material : h_scene->materials_loaded)
    {
        #ifdef LOG
                std::cerr << "\t\t loading: " << material.first << std::endl;
        #endif
        material_dictionary.insert(std::pair<string, int>(material.first, i));

        checkCudaErrors(cudaMallocManaged((void**)&(d_material_list[i]), sizeof(dMaterial*)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dMaterial*)/1000.f + "kB)" + " for dMaterial.");
        d_material_list[i]->baseColorFactor = float3_cast(material.second->baseColorFactor);
        d_material_list[i]->roughnessFactor = material.second->roughnessFactor;
        d_material_list[i]->metallicFactor = material.second->metallicFactor;
        d_material_list[i]->emissiveColorFactor = float3_cast(material.second->emissiveColorFactor);
        d_material_list[i]->fresnel = make_float3(0.04f);
        d_material_list[i]->ks = 1.f;
        d_material_list[i]->kd = 1.f;
        d_material_list[i]->radiance = 0.f;
        d_material_list[i]->emissive = false;

        d_material_list[i]->baseColorTexture = load_texture_uchar(material.second->baseColorTexture);
        d_material_list[i]->normalTexture = load_texture_uchar(material.second->normalTexture);
        d_material_list[i]->metallicRoughnessTexture = load_texture_uchar(material.second->metallicRoughnessTexture);
        d_material_list[i]->roughnessTexture = load_texture_uchar(material.second->roughnessTexture);
        d_material_list[i]->metallicTexture = load_texture_uchar(material.second->metallicTexture);

        i++;
    }
    #ifdef LOG
        std::cerr << "\t materials completed." << endl;
    #endif
    materials_loaded = true;
}

void dScene::load_transforms()
{
#ifdef LOG
    std::cerr << "\t loading transforms." << endl;
#endif
    // allocate memory (host)
    int nmb_transforms = h_scene->models.size();
    size_t sizeof_transform = nmb_transforms * sizeof(dTransform*);
    checkCudaErrors(cudaMallocManaged((void**)&d_transform_list, sizeof_transform), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_transform / 1000.f + "kB)" + " for transforms list.");

    // convert transform type for device and populate list with data
    int i = 0;
    for (auto model : h_scene->models)
    {
#ifdef LOG
        std::cerr << "\t\t loading: " << model->get_name() << " transform." << std::endl;
#endif
        checkCudaErrors(cudaMallocManaged((void**)&(d_transform_list[i]), sizeof(dTransform*)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dTransform*) / 1000.f + "kB)" + " for dTransform.");
        Matrix4x4 mat = Matrix4x4_cast(model->get_transform()->get_matrix());
        d_transform_list[i]->matrix = mat;
        d_transform_list[i]->inv_matrix = mat.inv();
    }
#ifdef LOG
    std::cerr << "\t transforms completed." << endl;
#endif
    transforms_loaded = true;
}

void dScene::load_lights()
{
#ifdef LOG
    std::cerr << "\t loading lights." << endl;
#endif

    _Rectangle** objs;

    // allocate memory (host)
    nmb_dir_lights = 1;//h_scene->get_lights().size();
    nmb_area_lights = 1;

    size_t sizeof_directions = nmb_dir_lights * sizeof(float3);
    size_t sizeof_positions = nmb_pnt_lights * sizeof(float3);
    size_t sizeof_objs = nmb_area_lights * sizeof(_Rectangle*);

    nmb_lights = nmb_dir_lights + nmb_pnt_lights + nmb_area_lights + 1;

    size_t sizeof_materials = nmb_lights * sizeof(dMaterial*);
    size_t sizeof_lights = nmb_lights * sizeof(dLight*);

    checkCudaErrors(cudaMalloc((void**)&d_lights, sizeof_lights), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_lights / 1000.f + "Kb)" + " for light list.");
    checkCudaErrors(cudaMallocManaged((void**)&light_directions, sizeof_directions), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_directions/1000.f + "kB)" + " for direction list.");
    checkCudaErrors(cudaMallocManaged((void**)&objs, sizeof_objs), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_objs/1000.f + "kB)" + " for object list.");
    checkCudaErrors(cudaMallocManaged((void**)&light_materials, sizeof_materials), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof_materials/1000.f + "kB)" + " for material list.");
    
#ifdef LOG
    std::cerr << "\t loading directional lights." << endl;
#endif

    int i = 0;
    for (auto light : h_scene->get_lights())
    {
        checkCudaErrors(cudaMallocManaged((void**)&(light_materials[i]), sizeof(dMaterial*)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dMaterial*)/1000.f + "kB)" + " for dMaterial.");
        light_materials[i]->radiance = light->getIntensity();
        light_materials[i]->emissiveColorFactor = float3_cast(light->getColor());

        checkCudaErrors(cudaMallocManaged((void**)&(light_directions[i]), sizeof(float3)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(float3)/1000.f + "kB)" + " for float3.");
        light_directions[i] = float3_cast(light->getDirection());

        i++;
    }
    
    add_directional_lights(light_directions, light_materials, h_scene->get_lights().size(), d_lights);
    
    // Area Lights

#ifdef LOG
    std::cerr << "\t loading area lights." << endl;
#endif
    
    checkCudaErrors(cudaMallocManaged((void**)&(light_materials[0]), sizeof(dMaterial*)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dMaterial*)\1000.f + "kB)" + " for dMaterial.");
    light_materials[0]->radiance = 125.f;
    light_materials[0]->emissiveColorFactor = make_float3(1.f);
    light_materials[0]->emissive = true;

    checkCudaErrors(cudaMallocManaged((void**)&(objs[0]), sizeof(_Rectangle*)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(_Rectangle*)\1000.f + "kB)" + " for float3.");
    objs[0] = new _Rectangle(make_float3(-0.5f, 100.f, -0.5f), make_float3(1.f, 0.f, 0.f), make_float3(0.f, 0.f, 1.f), make_float3(0.f, 1.f, 0.f));

    add_area_lights(objs, light_materials, 1, d_lights);

    // Environment Light //

#ifdef LOG
    std::cerr << "\t loading environment light." << endl;
#endif

    float3 color;
    cudaTextureObject_t hrd_texture;

    checkCudaErrors(cudaMallocManaged((void**)&d_environment_light, sizeof(dEnvironmentLight)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dEnvironmentLight) / 1000.f + "kB)" + " for dEnvironmentLight.");

    std::string filepath = h_scene->environment_light->get_texture_filepath();
    int tex_width = h_scene->environment_light->get_tex_width();
    int tex_height = h_scene->environment_light->get_tex_height();

    color = float3_cast(h_scene->environment_light->get_color());
    hrd_texture = load_texture_float(filepath);

    add_environment_light(hrd_texture, tex_width, tex_height, color, d_environment_light, d_lights);
}

void dScene::load_camera()
{
    d_camera = new dCamera();
    checkCudaErrors(cudaMallocManaged(&d_camera, sizeof(dCamera)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dCamera)/1000.f + "kB)" + " for camera.");
    camera_loaded = true;
    update_camera();
}

void dScene::load_nodes()
{
    m_assert(BVH_initialized, "load_nodes:");

    //size_t node_size = sizeof(LinearBVHNode) * bvh->get_nmb_nodes();
    //checkCudaErrors(cudaMalloc((void**)&d_nodes, node_size));
    //checkCudaErrors(cudaMemcpy(d_nodes, bvh->get_nodes(), node_size, cudaMemcpyHostToDevice));

    size_t node_size = sizeof(LinearBVHNode) * bvh->get_nmb_nodes();
    checkCudaErrors(cudaMallocManaged((void**)&d_nodes, node_size));
    checkCudaErrors(cudaMemcpy(d_nodes, bvh->get_nodes(), node_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMallocManaged((void**)&d_nodes_original, node_size));
    checkCudaErrors(cudaMemcpy(d_nodes_original, bvh->get_nodes(), node_size, cudaMemcpyHostToDevice));


    int* d_ordered_prims;
    size_t ordered_prims_size = sizeof(int) * bvh->get_ordered_prims().size();
    checkCudaErrors(cudaMalloc((void**)&d_ordered_prims, ordered_prims_size));
    checkCudaErrors(cudaMemcpy(d_ordered_prims, bvh->get_ordered_prims().data(), ordered_prims_size, cudaMemcpyHostToDevice));

    reorder_primitives(d_triangles, d_ordered_prims, h_scene->get_nmb_of_triangles());
}

void dScene::init_BVH_triangle_info()
{

    BVH_triangle_info.clear();

#ifdef LOG
    std::cerr << "\t gathering triangle information for BVH construction: ";
    clock_t start, stop;
    start = clock();
#endif
    int start_offest = 0;
    //size_t id = 0;
    for (RenderObject* r_obj : h_scene->models) {
        Model* model = dynamic_cast<Model*>(r_obj);
        for (Mesh* mesh : model->get_meshes()) {
            // Construct triangle info list for BVH
            size_t id = start_offest;

            auto indices = mesh->get_indices();
            auto vertices = mesh->get_vertices();

            for (size_t i = 0; i < indices.size(); i += 3) {
                unsigned int iv0 = indices[i];
                unsigned int iv1 = indices[i + 1];
                unsigned int iv2 = indices[i + 2];

                glm::mat4x4 mat = model->get_transform()->get_matrix();

                float3 v0 = float3_cast(glm::vec3(mat * glm::vec4(vertices[iv0].position, 1.f)));
                float3 v1 = float3_cast(glm::vec3(mat * glm::vec4(vertices[iv1].position, 1.f)));
                float3 v2 = float3_cast(glm::vec3(mat * glm::vec4(vertices[iv2].position, 1.f)));

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
    d_camera->inv_view_mat = Matrix4x4_cast(glm::inverse(h_scene->camera->lookat_mat));
}

void dScene::update_lights()
{
    int i = 0;
    for (auto light : h_scene->get_lights())
    {
        light_materials[i]->radiance = light->getIntensity();
        light_materials[i]->emissiveColorFactor = float3_cast(light->getColor());

        light_directions[i] = float3_cast(light->getDirection());

        i++;
    }

    add_directional_lights(light_directions, light_materials, h_scene->get_lights().size(), d_lights);
}

void dScene::update_environment_light()
{
    float3 color;
    cudaTextureObject_t hrd_texture;

    std::string filepath = h_scene->environment_light->get_texture_filepath();
    int tex_width = h_scene->environment_light->get_tex_width();
    int tex_height = h_scene->environment_light->get_tex_height();

    color = float3_cast(h_scene->environment_light->get_color());
    hrd_texture = load_texture_float(filepath);

    add_environment_light(hrd_texture, tex_width, tex_height, color, d_environment_light, d_lights);
}

void dScene::update_materials() {
#ifdef LOG
        //std::cerr << "\t loading materials." << endl;
#endif
        int i = 0;
        for (auto material : h_scene->materials_loaded)
        {
#ifdef LOG
            //std::cerr << "\t\t loading: " << material.first << std::endl;
#endif
            d_material_list[i]->baseColorFactor = float3_cast(material.second->baseColorFactor);
            d_material_list[i]->roughnessFactor = material.second->roughnessFactor;
            d_material_list[i]->metallicFactor = material.second->metallicFactor;
            d_material_list[i]->emissiveColorFactor = float3_cast(material.second->emissiveColorFactor);
            d_material_list[i]->fresnel = make_float3(0.04f);
            d_material_list[i]->ks = 1.f;
            d_material_list[i]->kd = 1.f;
            d_material_list[i]->radiance = 0.f;
            d_material_list[i]->emissive = false;

            //d_material_list[i]->baseColorTexture = load_texture_uchar(material.second->baseColorTexture);
            //d_material_list[i]->normalTexture = load_texture_uchar(material.second->normalTexture);
            //d_material_list[i]->metallicRoughnessTexture = load_texture_uchar(material.second->metallicRoughnessTexture);
            //d_material_list[i]->roughnessTexture = load_texture_uchar(material.second->roughnessTexture);
            //d_material_list[i]->metallicTexture = load_texture_uchar(material.second->metallicTexture);

            i++;
        }
#ifdef LOG
        //std::cerr << "\t materials completed." << endl;
#endif
}

void dScene::update_transforms() {
#ifdef LOG
    //std::cerr << "\t loading materials." << endl;
#endif
    int i = 0;
    for (auto model : h_scene->models)
    {
#ifdef LOG
        //std::cerr << "\t\t loading: " << material.first << std::endl;
#endif
        Matrix4x4 mat = Matrix4x4_cast(model->get_transform()->get_matrix());
        d_transform_list[i]->matrix = mat;
        d_transform_list[i]->inv_matrix = mat.inv();

        i++;
    }
    update_bvh(d_nodes, 0);

#ifdef LOG
    //std::cerr << "\t materials completed." << endl;
#endif
    //apply_transforms(h_scene->nmb_triangles, d_triangles);
}

Bounds3f dScene::update_bvh(LinearBVHNode* nodes, int current_node_index)
{
    LinearBVHNode* node = &nodes[current_node_index];

    if (node->nPrimitives > 0) 
    {
        for (int i = 0; i < node->nPrimitives; ++i) {

            dTriangle t = d_triangles[node->primitivesOffset + i];

            Matrix4x4 m = t.transform->matrix;

            float3 v0 = m * t.v0.position;
            float3 v1 = m * t.v1.position;
            float3 v2 = m * t.v2.position;

            Bounds3f bounds = Union(Bounds3f(v0, v1), v2);

            if (i > 0) node->bounds = Union(node->bounds, bounds);
            else node->bounds = bounds;
        }
        return node->bounds;
    } 
    else 
    {
        Bounds3f c0 = update_bvh(nodes, current_node_index + 1); // left
        Bounds3f c1 = update_bvh(nodes, node->secondChildOffset); // right

        node->bounds = Union(c0, c1);

        return node->bounds;
    }
}

void dScene::update_nodes()
{
    //m_assert(BVH_initialized, "load_nodes:");

    //size_t node_size = sizeof(LinearBVHNode) * bvh->get_nmb_nodes();
    //checkCudaErrors(cudaMalloc((void**)&d_nodes, node_size));
    //checkCudaErrors(cudaMemcpy(d_nodes, bvh->get_nodes(), node_size, cudaMemcpyHostToDevice));

    size_t node_size = sizeof(LinearBVHNode) * bvh->get_nmb_nodes();
    //checkCudaErrors(cudaMallocManaged((void**)&d_nodes, node_size));
    checkCudaErrors(cudaMemcpy(d_nodes, bvh->get_nodes(), node_size, cudaMemcpyHostToDevice));


    //int* d_ordered_prims;
    //size_t ordered_prims_size = sizeof(int) * bvh->get_ordered_prims().size();
    //checkCudaErrors(cudaMalloc((void**)&d_ordered_prims, ordered_prims_size));
    //checkCudaErrors(cudaMemcpy(d_ordered_prims, bvh->get_ordered_prims().data(), ordered_prims_size, cudaMemcpyHostToDevice));

    //reorder_primitives(d_triangles, d_ordered_prims, h_scene->get_nmb_of_triangles());
}