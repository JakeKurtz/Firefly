#include "SceneViewWindow.h"

SceneViewWindow::SceneViewWindow(int width, int height, Scene* s, dScene* ds)
{
    this->window_name = "Scene View##"+ to_string(gen_id());

    this->width = width;
    this->height = height;

    this->s = s;
    this->ds = ds;

    this->camera = new PerspectiveCamera(glm::vec3(0.f, 0.f, 10.f), (float)width / (float)height, glm::radians(120.f), 0.01f, 10000.f);

    this->path_tracer = new PathTracer(width, height);
    this->raster = new Rasterizer(width, height);

    image_buffer = new char[width * height * 3];

    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    glBindTexture(GL_TEXTURE_2D, 0);
}

void SceneViewWindow::draw()
{
    frame_time();

    GLuint fbo_tex;
    if (render_mode == RASTER_MODE) {
        fbo_tex = render_raster();
    }
    //else if (render_mode == 1) {
    //    pt->draw_debug(ds);
    //    fbo_tex = pt->interop->col_tex;
    //}
    else if (render_mode == PATHTRACE_MODE) {
        fbo_tex = render_pathtrace();
    }

    glBindTexture(GL_TEXTURE_2D, image_texture);
    glGetTextureImage(fbo_tex, 0, GL_RGB, GL_UNSIGNED_BYTE, width * height * 3, image_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_buffer);
    glBindTexture(GL_TEXTURE_2D, 0);

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_MenuBar;
    window_flags |= ImGuiWindowFlags_NoCollapse;

    if (!ImGui::Begin(window_name.c_str(), NULL, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
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
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Shading"))
        {
            int render_mode_before = render_mode;
            ImGui::Combo("Render Type", &render_mode, "OpenGL Rasterizer\0Ray Tracer\0Path Tracer\0");

            if ((int)render_mode != render_mode_before)
                buffer_reset = true;

            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    if (ImGui::IsWindowFocused() && !title_bar_hovered) {
        currently_interacting = true;
    }
    else {
        currently_interacting = false;
    }

    ImGuiIO& io = ImGui::GetIO();

    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImVec2 mp = ImGui::GetIO().MousePos;

    ImVec2 v = ImGui::GetWindowSize();

    float ratio = max((float)v.x / (float)width, (float)v.y / (float)height); // scale image to window size
    float height_p = height * ratio;
    float width_p = width * ratio;

    ImVec2 cursor_pos = ImVec2((v.x - width_p) * 0.5, (v.y - height_p) * 0.5);
    ImGui::SetCursorPos(cursor_pos); // center image

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddImage((void*)image_texture,
        pos,
        ImVec2(pos.x + (width_p), pos.y + (height_p)),
        ImVec2(0, 1),
        ImVec2(1, 0));

    ImGui::Text("Screen Size: %f %f", v.x, v.y);
    ImGui::Text("Mouse Pos: %f %f", mp.x, mp.y);
    ImGui::Text("interact_with_scene_view: %d", currently_interacting);
    ImGui::Text("samples: %d", *path_tracer->nmb_completed_pixels);

    ImGui::End();
}

void SceneViewWindow::set_raster_output_type(int t)
{
    raster_output_type = t;
}

void SceneViewWindow::window_size_callback(GLFWwindow* window, int width, int height)
{
}

void SceneViewWindow::mouse_callback(GLFWwindow* window, double xpos, double ypos)
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

    if (mouse_button_3_down && focused()) {
        camera->processMouseMovement(xoffset, yoffset);
        ds->update_flags |= UpdateCamera;
    }

    if (mouse_button_3_down && focused()) {
        image_pan_x += xoffset * (1.f / image_scale);
        image_pan_y -= yoffset * (1.f / image_scale);
    }
}

void SceneViewWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
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

    if (button == GLFW_MOUSE_BUTTON_3) {
        if (action == GLFW_PRESS) {
            mouse_button_3_down = true;
            click = true;
        }
        else if (action == GLFW_RELEASE) {
            mouse_button_3_down = false;
        }
    }
}

void SceneViewWindow::mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (focused()) {
        raw_mouse_scroll -= (float)yoffset;
        if (raw_mouse_scroll < 1.f) raw_mouse_scroll = 1.f;
        if (raw_mouse_scroll > 100.f) raw_mouse_scroll = 100.f;
        //image_scale = remap2(100, 1, 10, 0.25, raw_mouse_scroll);
    }
}

void SceneViewWindow::key_callback()
{
}

void SceneViewWindow::process_input(GLFWwindow* window)
{
    //Camera controls
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera->processKeyboard(FORWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera->processKeyboard(BACKWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera->processKeyboard(LEFT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera->processKeyboard(RIGHT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera->processKeyboard(UP, delta_time);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera->processKeyboard(DOWN, delta_time);
}
