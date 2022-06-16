#include "Scene.h"

static inline glm::mat4 mat4_cast(const aiMatrix4x4& m) { return glm::transpose(glm::make_mat4(&m.a1)); }
static inline glm::mat4 mat4_cast(const aiMatrix3x3& m) { return glm::transpose(glm::make_mat3(&m.a1)); }

Scene::Scene()
{
    //environmentLight = new EnvironmentLight(glm::vec3(0.8f, 0.2f, 0.2f));
}

Scene::Scene(vector<RenderObject*> _models, vector<Light*> _lights, Camera* _camera)
{
    models = _models;
    //lights = _lights;
    camera = _camera;

    //environmentLight = new EnvironmentLight(glm::vec3(0.8f));
}

void Scene::load(string const& path, glm::vec3 translate, glm::vec3 scale)
{
    // read file via ASSIMP
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_RemoveComponent | aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    std::string base_filename = path.substr(path.find_last_of("/\\") + 1);

    // remove extension from filename
    std::string::size_type const p(base_filename.find_last_of('.'));
    std::string file_name = base_filename.substr(0, p);

    // check for errors
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
    {
        cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
        return;
    }
    // retrieve the directory path of the filepath
    directory = path.substr(0, path.find_last_of('/'));

    if (scene->HasMeshes()) {

        vector<Mesh*> meshes;
        meshes.reserve(scene->mNumMeshes);

        aiMatrix4x4 model_mat = scene->mRootNode->mTransformation;

        load_model(meshes, model_mat, scene->mRootNode, scene);

        Model* model = new Model(meshes, mat4_cast(model_mat));
        model->set_name(file_name);
        model->set_directory(path);
        model->get_transform()->scale(scale);
        model->get_transform()->translate(translate);

        models.push_back(model);
    }

    if (scene->HasLights()) {
        for (unsigned int i = 0; i < scene->mNumLights; i++) {
            aiLight* light = scene->mLights[i];
            load_light(light, scene);
        }
    }

    if (scene->HasCameras()) {
        for (unsigned int i = 0; i < scene->mNumCameras; i++) {
            aiCamera* camera = scene->mCameras[i];
            load_camera(camera, scene);
        }
    }
}

void Scene::add_curve(Curve* curve)
{
    curves.push_back(curve);
}

void Scene::add_model(Model* r_obj)
{
    models.push_back(r_obj);

    for (auto mesh : r_obj->get_meshes()) {
        auto mat = mesh->get_material();
        materials_loaded.insert(std::pair<string, Material*>(mat->name, mat));
    }
}

void Scene::add_render_object(RenderObject* r_obj)
{
    if (r_obj->type() == TYPE_CURVE) {
        curves.push_back(r_obj);
    }

    if (r_obj->type() == TYPE_TRIANGLE_MESH) {
        Model* model = dynamic_cast<Model*>(r_obj);
        add_model(model);
    }
}

void Scene::update_triangle_count()
{
    nmb_triangles = 0;

    for (auto r_obj : models) {
        Model* model = dynamic_cast<Model*>(r_obj);
        for (auto mesh : model->get_meshes()) {
            nmb_triangles += mesh->get_nmb_of_triangles();
        }
    }
}

void Scene::set_camera(Camera* camera)
{
    this->camera = camera;
}

void Scene::add_light(PointLight* light)
{
    point_lights.push_back(light);
}

void Scene::add_light(DirectionalLight* light)
{
    dir_lights.push_back(light);
}

void Scene::set_environment_light(EnvironmentLight* _environment_light)
{
    environment_light = _environment_light;
}

void Scene::load_model(vector<Mesh*>& meshes, aiMatrix4x4 accTransform, aiNode* node, const aiScene* scene)
{
    accTransform = node->mTransformation * accTransform;

    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(load_mesh(mesh, accTransform, scene));
        nmb_triangles += mesh->mNumFaces;
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        load_model(meshes, accTransform, node->mChildren[i], scene);
    }
}

Light* Scene::load_light(aiLight* light, const aiScene* scene)
{
    return nullptr;
}

Mesh* Scene::load_mesh(aiMesh* mesh, aiMatrix4x4 accTransform, const aiScene* scene)
{
    string name = mesh->mName.C_Str();

    // data to fill
    vector<Vertex> vertices;
    vector<unsigned int> indices;

    glm::mat4 mat = glm::transpose(glm::inverse(mat4_cast(accTransform)));

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        glm::vec3 vector;

        // positions
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.position = glm::vec3(mat4_cast(accTransform) * glm::vec4(vector, 1));

        // normals
        if (mesh->HasNormals()) {
            vector.x = mesh->mNormals[i].x;
            vector.y = mesh->mNormals[i].y;
            vector.z = mesh->mNormals[i].z;

            vertex.normal = glm::normalize(glm::vec3(mat * glm::vec4(vector, 1)));
        }

        // texture coordinates
        if (mesh->mTextureCoords[0]) {
            glm::vec2 vec;
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;
            vertex.texCoords = vec;

            // tangent
            vector.x = mesh->mTangents[i].x;
            vector.y = mesh->mTangents[i].y;
            vector.z = mesh->mTangents[i].z;
            vertex.tangent = glm::normalize(glm::vec3(mat * glm::vec4(vector, 1)));

            // bitangent
            vector.x = mesh->mBitangents[i].x;
            vector.y = mesh->mBitangents[i].y;
            vector.z = mesh->mBitangents[i].z;
            vertex.bitangent = glm::normalize(glm::vec3(mat * glm::vec4(vector, 1)));
        }
        else {
            vertex.texCoords = glm::vec2(0.0f, 0.0f);
        }

        vertices.push_back(vertex);
    }

    // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];

        // retrieve all indices of the face and store them in the indices vector
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    return new Mesh(name, vertices, indices, load_material(material, scene));
}

Material* Scene::load_material(aiMaterial* material, const aiScene* scene)
{
    aiString name;
    material->Get(AI_MATKEY_NAME, name);

    if (name.length == 0) name = aiString(to_string((int)materials_loaded.size()));

    auto it = materials_loaded.find(name.C_Str());

    if (it == materials_loaded.end()) {
        Material* mat = new Material();

        mat->name = name.C_Str();

        material->Get(AI_MATKEY_TWOSIDED, mat->doubleSided);

        aiColor3D baseColorFactor, emissiveColorFactor;
        ai_real roughnessFactor, metallicFactor;
        material->Get(AI_MATKEY_COLOR_DIFFUSE, baseColorFactor);
        material->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColorFactor);
        material->Get(AI_MATKEY_ROUGHNESS_FACTOR, mat->roughnessFactor);
        material->Get(AI_MATKEY_METALLIC_FACTOR, mat->metallicFactor);

        mat->baseColorFactor = glm::make_vec3(&baseColorFactor.r);
        mat->emissiveColorFactor = glm::make_vec3(&emissiveColorFactor.r);

        mat->roughnessFactor = 1.f;
        mat->metallicFactor = 0.f;

        mat->baseColorTexture = load_texture(aiTextureType_DIFFUSE, "baseColorTexture", material, scene);
        mat->normalTexture = load_texture(aiTextureType_NORMALS, "normalTexture", material, scene);
        mat->occlusionTexture = load_texture(aiTextureType_AMBIENT_OCCLUSION, "occlusionTexture", material, scene);
        mat->emissiveTexture = load_texture(aiTextureType_EMISSIVE, "emissiveTexture", material, scene);
        mat->roughnessTexture = load_texture(aiTextureType_DIFFUSE_ROUGHNESS, "roughnessTexture", material, scene);
        mat->metallicTexture = load_texture(aiTextureType_METALNESS, "metallicTexture", material, scene);
        mat->metallicRoughnessTexture = load_texture(aiTextureType_UNKNOWN, "metallicRoughnessTexture", material, scene);

        materials_loaded.insert(std::pair<string, Material*>(mat->name, mat));

        return mat;
    }
    else {
        return materials_loaded[name.C_Str()];
    }
}

Texture* Scene::load_texture(aiTextureType type, string typeName, aiMaterial* mat, const aiScene* scene)
{
    Texture* texture = nullptr;

    if (mat->GetTextureCount(type)) {
        aiString str;
        mat->GetTexture(type, 0, &str);

        // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
        bool skip = false;
        for (unsigned int j = 0; j < textures_loaded.size(); j++)
        {
            if (std::strcmp(textures_loaded[j]->filepath.data(), str.C_Str()) == 0)
            {
                texture = textures_loaded[j];
            }
        }
        if (!skip)
        {   // if texture hasn't been loaded already, load it
            if ('*' == *str.C_Str()) {
                texture = new Texture(scene->GetEmbeddedTexture(str.C_Str()), GL_TEXTURE_2D, 0, 0, 0, typeName);
            }
            else {
                string path = this->directory + '/' + str.C_Str();
                texture = new Texture(path, GL_TEXTURE_2D, 0, 0, 0, typeName);
            }
            textures_loaded.push_back(texture);
        }
    }
    return texture;
}

Camera* Scene::load_camera(aiCamera* camera, const aiScene* scene)
{
    return nullptr;
}

void Scene::send_uniforms(Shader& shader)
{
    for (unsigned int i = 0; i < point_lights.size(); i++)
    {
        shader.setVec3("pnt_lights[" + std::to_string(i) + "].position", point_lights[i]->getPosition());
        shader.setVec3("pnt_lights[" + std::to_string(i) + "].color", point_lights[i]->getColor());
        shader.setFloat("pnt_lights[" + std::to_string(i) + "].intensity", point_lights[i]->getIntensity());
    }

    for (unsigned int i = 0; i < dir_lights.size(); i++)
    {
        shader.setVec3("dir_lights[" + std::to_string(i) + "].direction", dir_lights[i]->getDirection());
        shader.setVec3("dir_lights[" + std::to_string(i) + "].color", dir_lights[i]->getColor());
        shader.setFloat("dir_lights[" + std::to_string(i) + "].intensity", dir_lights[i]->getIntensity());
    }

    // TODO: cone lights at some point
}

void Scene::bind_environment_textures(Shader& shader)
{
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_CUBE_MAP, environment_light->getIrradianceMapID());

    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_CUBE_MAP, environment_light->getPrefilterMapID());

    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, environment_light->get_brdfLUT_ID());
}

int Scene::get_nmb_of_triangles()
{
    update_triangle_count(); // NOTE: this is wasteful computation. Maybe create flags for when to update this stuff?
    return nmb_triangles;
}

vector<DirectionalLight*> Scene::get_lights()
{
    return dir_lights;
}
