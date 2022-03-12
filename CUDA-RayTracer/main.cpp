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

const char* glsl_version = "#version 330 core";

int width = 1024;
int height = 1024;

int render_mode = 0;

bool buffer_reset = false;

float lastX;
float lastY;
bool firstMouse = true;
bool mouseDown = false;
bool click = false;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

static bool show_demo_window = true;
static bool show_another_window = false;
static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
glm::vec3 background_color = glm::vec3(0.0f);

PerspectiveCamera* camera = new PerspectiveCamera(glm::vec3(3.66268253, 1.98525786, -5.15738964), (float)width / (float)height, glm::radians(45.f), 0.01f, 10000.f);

static void glfw_error_callback(int error, const char* description);
static void glfw_window_size_callback(GLFWwindow* window, int width, int height);
static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos);
static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
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

    if (glewInit() != GLEW_OK)
        exit(EXIT_FAILURE);

    glfwSetErrorCallback(glfw_error_callback);
    glfwSetKeyCallback(*window, glfw_key_callback);
    glfwSetFramebufferSizeCallback(*window, glfw_window_size_callback);
    glfwSetCursorPosCallback(*window, glfw_mouse_callback);
    glfwSetMouseButtonCallback(*window, glfw_mouse_button_callback);

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
}
void render_gui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    if (show_demo_window)
        ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
        static float f = 0.0f;
        static int counter = 0;
        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &show_another_window);
        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color
        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }

    // 3. Show another simple window.
    {
        ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
        if (ImGui::TreeNode("Camera Settings"))
        {

            ImGui::RadioButton("Rasterizer", &render_mode, 0);
            ImGui::RadioButton("Raytracer", &render_mode, 1);
            ImGui::RadioButton("Pathtracer", &render_mode, 2);

            ImGui::Text("Background Color");
            ImGui::ColorEdit3("##background_color", (float*)&background_color);

            //ImGui::Text("Samples: %d", frame);

            ImGui::Text("Focal Distance");
            ImGui::DragFloat("##fd", &camera->focal_distance, 0.01f, 0.f, 1000.f, "%.2f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::Text("Apature Size");
            ImGui::DragFloat("##lr", &camera->lens_radius, 0.01f, 0.001f, 10.f, "%.4f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::Text("Zoom");
            ImGui::DragFloat("##zoom", &camera->zoom, 1.f, 1.f, 500.f, "%.2f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::Text("?");
            ImGui::DragFloat("##?", &camera->d, 1.f, 1.f, 500.f, "%.2f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::Text("Exposure");
            ImGui::DragFloat("##exposure", &camera->exposure, 0.01f, 0.1f, 5.f, "%.2f");

            if (ImGui::IsItemActive())
                buffer_reset = true;

            ImGui::TreePop();
        }
        ImGui::End();
    }

    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main(int argc, char** argv)
{

    GLFWwindow* window;
    glfw_init(&window, width, height);
    imgui_init(&window);

    EnvironmentLight* color_environmentLight = new EnvironmentLight(glm::vec3(0.f));

    DirectionalLight* light_1 = new DirectionalLight();
    light_1->setIntensity(25.f);
    light_1->setDirection(glm::vec3(0.f, 0.98, 0.001));
    light_1->setColor(glm::vec3(1.f));

    DirectionalLight* light_2 = new DirectionalLight();
    light_2->setIntensity(50.f);
    light_2->setDirection(glm::vec3(1,1,1));
    light_2->setColor(glm::vec3(1.f));

    DirectionalLight* light_3 = new DirectionalLight();
    light_3->setIntensity(50.f);
    light_3->setDirection(glm::vec3(-10.f, 10.f, -10.f) - glm::vec3(0.f, 0, 0.f));
    light_3->setColor(glm::vec3(1.f));

    Scene* s = new Scene();
    //s->load("../models/test_scene.glb");
    s->load("../models/bunny.glb");
    //s->load("../models/monkey.glb");
    //s->load("../models/dragon.glb");
    s->add_light(light_1);
    //s->add_light(light_2);
    //s->add_light(light_3);

    s->set_environment_light(color_environmentLight);
    s->set_camera(camera);

    dScene* ds = new dScene(s);
    PathTracer* pt = new PathTracer(width, height);
    Rasterizer* r = new Rasterizer(width, height);

    while (!glfwWindowShouldClose(window))
    {
        // Per-frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfw_process_input(window);

        if (render_mode == 0)
            r->draw(s);
        else if (render_mode == 1) {
            pt->draw_debug(ds);
        }
        else if (render_mode == 2) {
            pt->draw(ds);
            if (buffer_reset) {
                pt->clear_buffer();
                buffer_reset = false;
            }
        }

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
}
static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    if (mouseDown && !(ImGui::GetIO().WantCaptureMouse)) {
        camera->processMouseMovement(xoffset, yoffset);
        buffer_reset = true;
    }
}
static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouseDown = true;
            click = true;
        }
        else if (action == GLFW_RELEASE) {
            mouseDown = false;
        }
    }

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

    //Camera controls
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera->processKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera->processKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera->processKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera->processKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera->processKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera->processKeyboard(DOWN, deltaTime);
}