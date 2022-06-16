#include "ObjectEditWindow.h"

/*
ObjectEditWindow::ObjectEditWindow(int width, int height, Curve* curve)
{
    this->width = width;
    this->height = height;

    init();

    s->add_curve(curve);
    
    (this)->curve = curve;
    (this)->transform_world = (this)->curve->get_transform();

    object_type = OBJ_TYPE_CURVE;

    curve->center();
    //centroid = center_of_mass(model);
    //transform_edit->translate(-centroid);
    //transform_edit->set_centroid(centroid);
    
}
*/
ObjectEditWindow::ObjectEditWindow(int width, int height)
{
    this->width = width;
    this->height = height;

    this->window_name = "Edit View##" + to_string(gen_id());

    this->raster = new Rasterizer(width, height);
    this->path_tracer = new PathTracer(width, height);

    transform_edit = new Transform();

    init_scene();
    init_image_buffer();
}

void ObjectEditWindow::init_scene()
{
    DirectionalLight* light = new DirectionalLight();
    light->setIntensity(10.f);
    light->setDirection(glm::vec3(10.f));
    light->setColor(glm::vec3(1.f));

    env_light_hrd = new EnvironmentLight("../hrdi/HDR_029_Sky_Cloudy_Env.hdr");
    env_light_color = new EnvironmentLight(glm::vec3(0.5f));

    this->s = new Scene();

    s->add_light(light);
    s->set_environment_light(env_light_color);

    camera = new PerspectiveCamera(glm::vec3(3.69999862, 21.1000443, 100.f), (float)width / (float)height, glm::radians(45.f), 0.01f, 10000.f);
    //camera = new PerspectiveCamera(glm::vec3(0,0, 100.f), (float)width / (float)height, glm::radians(45.f), 0.01f, 10000.f);
    camera->updateCameraVectors();

    s->set_camera(camera);

    this->ds = new dScene(s);
}

void ObjectEditWindow::init_image_buffer()
{
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

void ObjectEditWindow::draw()
{
    frame_time();

    GLuint fbo_tex = 0;
     
    if (r_obj != nullptr) {
        r_obj->set_transform(transform_edit);

        if (render_mode == RASTER_MODE) {
            fbo_tex = render_raster();
        }
        else if (render_mode == PATHTRACE_MODE) {
            fbo_tex = render_pathtrace();
        }
        else if (render_mode == WIRE_MODE) {
            raster->draw_wireframe(s, camera);
            fbo_tex = raster->fbo->color_attachments[0]->id;
        }

        r_obj->set_transform(transform_world);
    }

    // REFACTOR: Duplicate Code.
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
    // END REFACTOR: Duplicate Code.

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
            ImGui::Combo("Render Type", &render_mode, "OpenGL Rasterizer\0MC Path Tracer\0Wireframe\0");

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
    // REFACTOR: Duplicate Code.
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

    // END REFACTOR: Duplicate Code.
    ImGui::End();
}

void ObjectEditWindow::set_curve(Curve* curve)
{
    r_obj = curve;

    s->models.clear();
    s->curves.clear();
    s->add_render_object(curve);

    (this)->transform_world = curve->get_transform();

    curve->center(); // This is a bug. I'm too lazy to fix it right now :D

    transform_edit->reset();
    transform_edit->set_centroid(glm::vec3(0.f));
    transform_edit->translate(-glm::vec3(0.f));
    transform_edit->apply();
}

void ObjectEditWindow::set_model(Model* model)
{
    r_obj = model;

    s->models.clear();
    s->curves.clear();
    s->add_render_object(model);

    (this)->transform_world = model->get_transform();

    centroid = center_of_mass(model);

    transform_edit->reset();
    transform_edit->set_centroid(centroid);
    transform_edit->translate(-centroid);
    transform_edit->apply();

    ds->update_flags |= UpdateTransforms;
    ds->update_flags |= UpdateScene;
    ds->update();
}

void ObjectEditWindow::window_size_callback(GLFWwindow* window, int width, int height)
{
}

void ObjectEditWindow::mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    // REFACTOR: Duplicate Code.
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
    // END REFACTOR: Duplicate Code.

    if (mouse_button_3_down && focused()) {

        xoffset *= 0.01;
        yoffset *= 0.01;

        transform_edit->rotate(-yoffset, xoffset, 0.f);
        transform_edit->apply();

        ds->update_flags |= UpdateTransforms;
    }

    //if (mouse_button_3_down && focused()) {
        //image_pan_x += xoffset * (1.f / image_scale);
        //image_pan_y -= yoffset * (1.f / image_scale);

        //float dir = 1;

        //if (xoffset < 0 || yoffset < 0) dir = 0.01;

        //transform_edit->scale(dir * glm::vec2(xoffset, yoffset).length());
    //}
}

void ObjectEditWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    // REFACTOR: Duplicate Code.
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
    // END REFACTOR: Duplicate Code.
}

void ObjectEditWindow::mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (focused()) {
        //raw_mouse_scroll += (float)yoffset * 0.01;
        if (yoffset < 0) raw_mouse_scroll = 0.9f;
        if (yoffset > 0) raw_mouse_scroll = 1.1f;
        transform_edit->scale(raw_mouse_scroll);
        transform_edit->apply();

        ds->update_flags |= UpdateTransforms;
    }
}

void ObjectEditWindow::key_callback()
{
}

void ObjectEditWindow::process_input(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera->position.y += 0.1;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera->position.y -= 0.1;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera->position.x += 0.1;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera->position.x -= 0.1;

    camera->updateCameraVectors();
    //ds->update_flags |= UpdateCamera;
}
