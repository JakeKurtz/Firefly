#include "GLCommon.h"
#include "globals.h"
#include <cstdlib>

#include "Scene.h"
#include "dScene.h"
#include "PathTracer.h"
#include "Rasterizer.h"
#include "PerspectiveCamera.h"

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"

#include <stb_image.h>
#include <stb_image_write.h>
#include "SceneViewWindow.h"
#include "ObjectEditWindow.h"


const char* glsl_version = "#version 330 core";

int width = 1024;
int height = 1024;

int render_mode = 0;

bool buffer_reset = false;

float lastX;
float lastY;
bool firstMouse = true;
bool mouseDown = false;
bool mouse_button_3_down = false;
bool click = false;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

static bool show_demo_window = true;
static bool show_another_window = false;
static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
glm::vec3 background_color = glm::vec3(0.0f);
GLuint image_texture;
char* image_buffer = new char[width * height * 3];

Rasterizer* r;
PathTracer* pt;

Scene* s;
dScene* ds;

int light_select_type = 0;
int environment_color_mode = 1;

Camera* active_camera;
PerspectiveCamera* camera = new PerspectiveCamera(glm::vec3(0.289340049, 4.11911869, 10.5660067), (float)width / (float)height, glm::radians(45.f), 0.01f, 10000.f);
//PerspectiveCamera* camera = new PerspectiveCamera(glm::vec3(0.f, 0.f, 10.f), (float)width / (float)height, glm::radians(45.f), 0.01f, 10000.f);

static void glfw_error_callback(int error, const char* description);
static void glfw_window_size_callback(GLFWwindow* window, int width, int height);
static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos);
static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
static void glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void glfw_process_input(GLFWwindow* window);

static void glfw_init(GLFWwindow** window, const int width, const int height)
{
    //
    // INITIALIZE GLFW/GLAD
    //

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glsl_version = "#version 330";

    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef PXL_FULLSCREEN
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    *window = glfwCreateWindow(mode->width, mode->height, "GLFW / CUDA Interop", monitor, NULL);
#else
    * window = glfwCreateWindow(width, height, "GLFW / CUDA Interop", NULL, NULL);
#endif

    if (*window == NULL)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(*window);

    //if (glewInit() != GLEW_OK)
    //    exit(EXIT_FAILURE);

    glfwSetErrorCallback(glfw_error_callback);
    glfwSetKeyCallback(*window, glfw_key_callback);
    glfwSetFramebufferSizeCallback(*window, glfw_window_size_callback);
    glfwSetCursorPosCallback(*window, glfw_mouse_callback);
    glfwSetMouseButtonCallback(*window, glfw_mouse_button_callback);
    glfwSetScrollCallback(*window, glfw_mouse_scroll_callback);

    // set up GLAD
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // tell GLFW to capture our mouse
    glfwSetInputMode(*window, GLFW_CURSOR, GLFW_CURSOR);

    // ignore vsync for now
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

    GLFW_INIT = true;
}
static void imgui_init(GLFWwindow** window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    //io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    //ImGui::StyleColorsDark();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(*window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    show_demo_window = true;
    show_another_window = false;
    ImVec4 clear_color;

    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

}

// raw_mouse_scroll = (1.f - low2)(high1-low1) / (high2-low2) + low2
// NOTE: This is a nasty hack, but it works for now.
float raw_mouse_scroll = 8.6153846153846f;
float image_scale = 1.f;
float image_pan_x = 0;
float image_pan_y = 0;

bool interact_with_scene_view = true;
bool interact_with_render_view = true;

bool show_scene_view = false;
bool show_render_view = false;

bool show_properties_view = false;

vector<RenderingContext*> contexts;
RenderingContext* active_context;

Material* material_selected = nullptr;
Curve* curve_selected = nullptr;
Model* model_selected = nullptr;
EnvironmentLight* environment_light_selected = nullptr;
DirectionalLight* dir_light_selected = nullptr;
PointLight* point_light_selected = nullptr;

EnvironmentLight* color_environmentLight;// = new EnvironmentLight(glm::vec3(0.f));
EnvironmentLight* hrd_environmentLight;// = new EnvironmentLight("../hrdi/HDR_029_Sky_Cloudy_Env.hdr");

ObjectEditWindow* OEW;

float remap2(float high1, float low1, float high2, float low2, float value) {
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
}

void render_view() 
{
    pt->render_image(ds);
    GLuint fbo_tex = pt->interop->col_tex;

    glBindTexture(GL_TEXTURE_2D, image_texture);
    glGetTextureImage(fbo_tex, 0, GL_RGB, GL_UNSIGNED_BYTE, width * height * 3, image_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_buffer);

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_MenuBar;
    window_flags |= ImGuiWindowFlags_NoCollapse;

    if (!ImGui::Begin("Render View", NULL, window_flags))
    {
        ImGui::End();
        return;
    }
    bool title_bar_hovered = ImGui::IsItemHovered();

    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Image"))
        {
            if (ImGui::MenuItem("Save")) {
                stbi_flip_vertically_on_write(true);
                stbi_write_png("../scene_output.png", width, height, 3, image_buffer, width * 3);
            }
            ImGui::MenuItem("Save As");
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View"))
        {
            if (ImGui::MenuItem("Zoom In", "Ctrl++")) {
                image_scale += 1.5;
            }
            if (ImGui::MenuItem("Zoom Out", "Ctrl+-")) {
                image_scale -= 1.5;
            }
            if (ImGui::MenuItem("Fit on Screen", "Ctrl+0")) {

            }
            if (ImGui::MenuItem("100%", "Ctrl+1")) {
                image_scale = 1.f;
                image_pan_x = 0.f;
                image_pan_y = 0.f;
            }
            if (ImGui::MenuItem("Flip Horizontal")) {

            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    if (ImGui::IsWindowFocused() && !title_bar_hovered) {
        interact_with_render_view = true;
    }
    else {
        interact_with_render_view = false;
    }

    ImGuiIO& io = ImGui::GetIO();

    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImVec2 mp = ImGui::GetIO().MousePos;

    ImVec2 v = ImGui::GetWindowSize();
    ImGui::Text("Screen Size: %f %f", v.x, v.y);
    ImGui::Text("Mouse Pos: %f %f", mp.x, mp.y);
    ImGui::Text("interact_with_scene_view: %d", interact_with_render_view);

    float ratio = 1.f;
    float height_p = height * ratio * image_scale;
    float width_p = width * ratio * image_scale;

    ImVec2 cursor_pos = ImVec2((v.x - width_p) * 0.5 + (image_pan_x * image_scale), (v.y - height_p) * 0.5 + (image_pan_y * image_scale));
    ImGui::SetCursorPos(cursor_pos); // center image

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddImage((void*)image_texture,
        pos,
        ImVec2(pos.x + (width_p), pos.y + (height_p)),
        ImVec2(0, 1),
        ImVec2(1, 0));

    ImGui::End();
}

void curve_view() {

}

void camera_props()
{
    ImGui::Text("Camera");

    if (ImGui::CollapsingHeader("Lens"))
    {
        ImGui::Text("Type");

        ImGui::Text("Focal Length");
        //ImGui::DragFloat("##fd", &camera->focal_distance, 0.01f, 0.f, 1000.f, "%.2f");

        ImGui::Separator();
    }

    if (ImGui::CollapsingHeader("Depth of Field"))
    {
        ImGui::Text("Focal Distance");
        ImGui::DragFloat("##fd", &active_camera->focal_distance, 0.01f, 0.f, 1000.f, "%.2f");

        if (ImGui::IsItemActive()) {
            active_context->get_dscene()->update_flags |= UpdateCamera;
        }

        if (ImGui::TreeNode("Apature")) {
            ImGui::Text("Size");
            ImGui::DragFloat("##lr", &active_camera->lens_radius, 0.01f, 0.001f, 10.f, "%.4f");
            if (ImGui::IsItemActive()) {
                active_context->get_dscene()->update_flags |= UpdateCamera;
            }
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    if (ImGui::CollapsingHeader("Film"))
    {
        ImGui::Text("Exposure");
        ImGui::DragFloat("##exposure", &active_camera->exposure, 0.01f, 0.1f, 5.f, "%.2f");
        if (ImGui::IsItemActive()) {
            active_context->get_dscene()->update_flags |= UpdateCamera;
        }
        ImGui::Separator();
    }
}

void curve_props() {
    ImGui::Text("Curve");

    if (curve_selected != nullptr) {
        ImGui::Text(("name: " + curve_selected->get_name()).c_str());

        int res = curve_selected->get_depth();
        ImGui::SliderInt("Resolution##resolution", (int*)&res, 0, 20);

        if (res != curve_selected->get_depth()) {
            curve_selected->set_depth(res);
            curve_selected->update();
        }

    }
}

void scene_graph() 
{
    static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;
    
    ImGui::Begin("Scene Graph");

    if (ImGui::TreeNode("SCENE"))
    {

        static int selection_mask = (1 << 2);
        int node_clicked = -1;
        int id = 0;

        if (ImGui::TreeNode("MODELS")) {
            for (RenderObject* r_obj : s->models) {

                Model* model = dynamic_cast<Model*>(r_obj);

                ImGuiTreeNodeFlags node_flags = base_flags;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = model->get_id();
                bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, node_flags, ("MODEL: " + model->get_name()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    model_selected = model;
                    OEW->set_model(model);
                }

                if (model_node_open)
                {
                    for (Mesh* mesh : model->get_meshes()) {

                        id = mesh->get_id();
                        bool mesh_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, node_flags, ("MESH: " + mesh->get_name()).c_str(), id);
                        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                            node_clicked = id;
                        }

                        if (mesh_node_open) {
                            
                            ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                            ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("MATERIAL: " + mesh->get_material()->name).c_str(), id);
                            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                                node_clicked = id;
                                material_selected = mesh->get_material();
                            }
                            ImGui::TreePop();
                        }
                    }
                    ImGui::TreePop();
                }
            }

            if (node_clicked != -1)
            {
                selection_mask = (1 << node_clicked);
            }
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("CURVES")) {
            for (RenderObject* r_obj : s->curves) {

                Curve* curve = dynamic_cast<Curve*>(r_obj);

                ImGuiTreeNodeFlags node_flags = base_flags;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = curve->get_id();
                //bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, node_flags, ("CURVE: " + curve->get_name()).c_str(), id);
                ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("CURVE: " + curve->get_name()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    curve_selected = curve;
                    OEW->set_curve(curve);
                }
                ImGui::TreePop();
                //if (model_node_open)
                //{
                    /*for (Mesh* mesh : curve->get_meshes()) {

                        id = mesh->get_id();
                        bool mesh_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, node_flags, ("MESH: " + mesh->get_name()).c_str(), id);
                        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                            node_clicked = id;
                        }

                        if (mesh_node_open) {

                            ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                            ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("MATERIAL: " + mesh->get_material()->name).c_str(), id);
                            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                                node_clicked = id;
                                material_selected = mesh->get_material();
                            }
                            ImGui::TreePop();
                        }
                    }
                    ImGui::TreePop();
                    */
                //}
            }

            if (node_clicked != -1)
            {
                selection_mask = (1 << node_clicked);
            }
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("LIGHTS")) {

            ImGuiTreeNodeFlags node_flags = base_flags;

            for (DirectionalLight* light : s->dir_lights) {
                ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = light->getId();
                bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("DIR: " + light->get_name()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    light_select_type = 1;
                    dir_light_selected = light;
                }
            }
            for (PointLight* light : s->point_lights) {
                ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = light->getId();
                bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("PNT: " + light->get_name()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    light_select_type = 2;
                    point_light_selected = light;
                }
            }
            if (s->environment_light != nullptr) {
                ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = 0;
                bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("ENV: " + s->environment_light->get_texture_filepath()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    light_select_type = 0;
                    environment_light_selected = s->environment_light;
                }
            }
            ImGui::TreePop();
        }
    }
    ImGui::End();
}

void material_props() 
{
    ImGui::Text("Material");

    if (material_selected != nullptr) {
        ImGui::Text(("name: "+material_selected->name).c_str());

        ImGui::ColorEdit3("Base Color##base_color", (float*)&material_selected->baseColorFactor, ImGuiColorEditFlags_NoInputs);

        if (ImGui::IsItemActive()) {
            active_context->get_dscene()->update_flags |= UpdateMaterials;
        }

        ImGui::ColorEdit3("Emissive Color##emissive_color", (float*)&material_selected->emissiveColorFactor, ImGuiColorEditFlags_NoInputs);

        ImGui::SliderFloat("Roughness Factor", &material_selected->roughnessFactor, 0.01f, 1.0f);

        if (ImGui::IsItemActive()) {
            active_context->get_dscene()->update_flags |= UpdateMaterials;
        }

        ImGui::SliderFloat("Metallic Factor", &material_selected->metallicFactor, 0.01f, 1.0f);

        if (ImGui::IsItemActive()) {
            active_context->get_dscene()->update_flags |= UpdateMaterials;
        }
    }
}

void model_props() 
{
    ImGui::Text("Model");
}

void mesh_props() 
{
    ImGui::Text("Mesh");
}

void light_props() 
{
    ImGui::Text("Light");
    if (active_context != nullptr) {
        switch (light_select_type) {
        case 0:

            ImGui::Text("Environment");

            ImGui::Combo("Color", &environment_color_mode, "RGB\0HRD Texture\0");

            if (environment_color_mode == 0) {

                if (s->environment_light != color_environmentLight) {
                    s->set_environment_light(color_environmentLight);
                    active_context->get_dscene()->update_flags |= UpdateEnvironmentLight;
                }

                glm::vec3 color = color_environmentLight->get_color();
                ImGui::ColorEdit3("Color##env_color", (float*)&color, ImGuiColorEditFlags_NoInputs);

                ImGui::DragFloat("Intensity##fd", &active_context->get_dscene()->get_environment_light()->ls, 0.01f, 0.f, 1000.f, "%.2f");

                if (color != color_environmentLight->get_color()) {
                    color_environmentLight->set_color(color);
                    active_context->get_dscene()->update_flags |= UpdateEnvironmentLight;
                }
            }
            else if (environment_color_mode == 1) {

                if (s->environment_light != hrd_environmentLight) {
                    s->set_environment_light(hrd_environmentLight);
                    active_context->get_dscene()->update_flags |= UpdateEnvironmentLight;
                }

                static char buf1[1024] = "../hrdi/HDR_029_Sky_Cloudy_Env.hdr";

                ImGui::InputText("filepath", buf1, 1024);

                if (ImGui::Button("load")) {
                    // TODO: better error checking for null strings.
                    if (strcmp(buf1, hrd_environmentLight->get_texture_filepath().c_str())) {
                        hrd_environmentLight->set_texture_filepath(buf1);
                        active_context->get_dscene()->update_flags |= UpdateEnvironmentLight;
                    }
                }
                ImGui::DragFloat("Intensity##fd", &active_context->get_dscene()->get_environment_light()->ls, 0.01f, 0.f, 1000.f, "%.2f");
            }
            //float intensity = environment_light_selected->get_ls();
            //ImGui::DragFloat("Intensity##fd", &intensity, 0.01f, 0.f, 1000.f, "%.2f");
            break;
        case 1:
        {
            ImGui::Text("Directional");

            glm::vec3 dir = dir_light_selected->getDirection();
            ImGui::DragFloat3("Direction", (float*)&dir, 0.01, 0.f, 1.f);
            dir_light_selected->setDirection(dir);

            if (ImGui::IsItemActive()) {
                active_context->get_dscene()->update_flags |= UpdateLights;
            }

            glm::vec3 color = dir_light_selected->getColor();
            ImGui::ColorEdit3("Color##dir_color", (float*)&color, ImGuiColorEditFlags_NoInputs);
            dir_light_selected->setColor(color);

            if (ImGui::IsItemActive()) {
                active_context->get_dscene()->update_flags |= UpdateLights;
            }

            float intensity = dir_light_selected->getIntensity();
            ImGui::DragFloat("Intensity##fd", &intensity, 0.01f, 0.f, 1000.f, "%.2f");
            dir_light_selected->setIntensity(intensity);

            if (ImGui::IsItemActive()) {
                active_context->get_dscene()->update_flags |= UpdateLights;
            }
        }
        break;
        case 2:
        {
            ImGui::Text("Point");

            glm::vec3 pos = point_light_selected->getPosition();
            ImGui::DragFloat3("Position", (float*)&pos, 0.1, 0.f, 100.f);
            point_light_selected->setPosition(pos);

            if (ImGui::IsItemActive()) {
                active_context->get_dscene()->update_flags |= UpdateLights;
            }

            glm::vec3 color = point_light_selected->getColor();
            ImGui::ColorEdit3("Color##pnt_color", (float*)&color, ImGuiColorEditFlags_NoInputs);
            point_light_selected->setColor(color);

            if (ImGui::IsItemActive()) {
                active_context->get_dscene()->update_flags |= UpdateLights;
            }

            float intensity = point_light_selected->getIntensity();
            ImGui::DragFloat("Intensity##fd", &intensity, 0.01f, 0.f, 1000.f, "%.2f");
            point_light_selected->setIntensity(intensity);

            if (ImGui::IsItemActive()) {
                active_context->get_dscene()->update_flags |= UpdateLights;
            }
        }
        break;
        }
    }
}

void properties_view()
{
    if (!ImGui::Begin("Properties"))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }

    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    if (ImGui::BeginTabBar("MyTabBar", tab_bar_flags))
    {
        if (ImGui::BeginTabItem("Scene Properties"))
        {
            camera_props();
            ImGui::Separator();

            light_props();
            ImGui::Separator();

            model_props();
            ImGui::Separator();

            mesh_props();
            ImGui::Separator();

            curve_props();
            ImGui::Separator();

            material_props();

            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Rendering Properties"))
        {
            //if (render_mode >= 1) pathtracing_props();
            //if (render_mode == 0) raster_props();

            active_context->render_properties();

            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void render_gui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    properties_view();

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("import"))
            {
                //Do something
            }
            if (ImGui::MenuItem("save as"))
            {
                //Do something
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Scene"))
        {
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoScrollbar;
    window_flags |= ImGuiWindowFlags_NoCollapse;

    curve_view();
    scene_graph();

    for (auto c : contexts) {
        if (c->focused()) {
            active_context = c;
            active_camera = c->get_camera();
        }

        c->draw();
    }

    ImGui::Begin("Rendering Properties", &show_another_window);
    active_context->render_properties();
    ImGui::End();

    camera_props();

    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main(int argc, char** argv)
{
    GLFWwindow* window;
    glfw_init(&window, width, height);
    imgui_init(&window);

    active_camera = camera;

    color_environmentLight = new EnvironmentLight(glm::vec3(0.f));
    hrd_environmentLight = new EnvironmentLight("../hrdi/HDR_029_Sky_Cloudy_Env.hdr");

    DirectionalLight* light_1 = new DirectionalLight();
    light_1->setIntensity(25.f);
    light_1->setDirection(glm::vec3(0.f, 0.98, 0.001));
    light_1->setColor(glm::vec3(1.f));

    DirectionalLight* light_2 = new DirectionalLight();
    light_2->setIntensity(50.f);
    light_2->setDirection(glm::vec3(0,0,1));
    light_2->setColor(glm::vec3(1.f));

    DirectionalLight* light_3 = new DirectionalLight();
    light_3->setIntensity(0.f);
    light_3->setDirection(glm::vec3(10.f, 10.f, -10.f) - glm::vec3(0.f, 0, 0.f));
    light_3->setColor(glm::vec3(1.f));

    s = new Scene();
    s->load("../models/greek_sculpture.glb");
    //s->add_light(light_1);
    //s->add_light(light_2);
    s->add_light(light_3);

    s->set_environment_light(hrd_environmentLight);
    s->set_camera(active_camera);

    std::vector<glm::vec3> control_points_test;
    control_points_test.push_back(glm::vec3(-1, 0, 0));
    control_points_test.push_back(glm::vec3(-1, 1, 0));
    control_points_test.push_back(glm::vec3(1, 1, 0));
    control_points_test.push_back(glm::vec3(1, 0, 0));

    Bezier* curve_test = new Bezier(control_points_test);

    //s->add_curve(curve_test);
    s->add_render_object(curve_test);

    ds = new dScene(s);
    pt = new PathTracer(width, height);
    r = new Rasterizer(width, height);

    SceneViewWindow* SVW = new SceneViewWindow(1024, 1024, s, ds);
    OEW = new ObjectEditWindow(1024, 1024);

    //Model* model = dynamic_cast<Model*>(s->models[0]);

    //OEW->set_model(model);

    contexts.push_back(SVW);
    contexts.push_back(OEW);

    //Model* model = dynamic_cast<Model*>(s->models[0]);
    //OEW->set_model(model);

    while (!glfwWindowShouldClose(window))
    {
        s->set_camera(active_camera);

        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Per-frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfw_process_input(window);

        render_gui();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}

static void glfw_error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    buffer_reset = true;
    // get context
    //Interop* interop = (Interop*)glfwGetWindowUserPointer(window);
    //interop->set_size(width, height);
    //if (active_context != nullptr) active_context->window_size_callback(window, width, height);
}
static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (active_context != nullptr) active_context->mouse_callback(window, xpos, ypos);
}
static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (active_context != nullptr) active_context->mouse_button_callback(window, button, action, mods);
}
static void glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (active_context != nullptr) active_context->mouse_scroll_callback(window, xoffset, yoffset);
}
static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    /*
    float3 front;
    front.x = cos(cam_yaw) * cos(cam_pitch);
    front.y = sin(cam_pitch);
    front.z = sin(cam_yaw) * cos(cam_pitch);

    cam_dir = normalize(front);
    cam_right = normalize(cross(cam_dir, cam_worldUp));
    cam_up = normalize(cross(cam_right, cam_dir));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos -= cam_dir * cam_movement_spd;
    if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos += cam_dir * cam_movement_spd;
    if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos -= cam_right * cam_movement_spd;
    if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos += cam_right * cam_movement_spd;
    if (key == GLFW_KEY_Q && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos += cam_up * cam_movement_spd;
    if (key == GLFW_KEY_E && (action == GLFW_PRESS || action == GLFW_REPEAT))
        cam_pos -= cam_up * cam_movement_spd;
    buffer_reset = true;
    */
}
static void glfw_process_input(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (active_context != nullptr) active_context->process_input(window);
}