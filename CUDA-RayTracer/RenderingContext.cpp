#include "RenderingContext.h"

RenderingContext::RenderingContext()
{
}

RenderingContext::RenderingContext(int width, int height, Scene* s, dScene* ds, Camera* camera)
{

}

void RenderingContext::frame_time()
{
    float current_frame_time = glfwGetTime();
    delta_time = current_frame_time - last_frame_time;
    last_frame_time = current_frame_time;
}

GLuint RenderingContext::render_raster()
{
    raster->draw(s, camera);

    switch (raster_output_type) {
    case POSITION:
        return raster->gBuffer->color_attachments[0]->id;
    case NORMAL:
        return raster->gBuffer->color_attachments[1]->id;
    case ALBEDO:
        return raster->gBuffer->color_attachments[2]->id;
    case MRO:
        return raster->gBuffer->color_attachments[3]->id;
    case EMISSION:
        return raster->gBuffer->color_attachments[4]->id;
    case DEPTH:
        return raster->gBuffer->depth_attachment->id;
    default:
        return raster->fbo->color_attachments[0]->id;
    }
}

GLuint RenderingContext::render_pathtrace()
{
    path_tracer->render_image(ds);
    return path_tracer->interop->col_tex;
}

void RenderingContext::raster_props()
{
    ImGui::Text("Rasterizer");

    if (ImGui::CollapsingHeader("Debug"))
    {
        ImGui::Combo("Deferred Buffers", &raster_output_type, "Shaded\0Position\0Normal\0Albedo\0Metallic Rough AO\0Emission\0Depth\0");
    }

    if (ImGui::CollapsingHeader("Shadows"))
    {

    }
}

void RenderingContext::pathtracing_props()
{
    ImGui::Text("Path Tracing");

    if (ImGui::CollapsingHeader("Sampling"))
    {
        ImGui::Combo("Integrator", &integrator_mode, "Path Tracing\0Branched Path Tracing\0");
        if (integrator_mode == 0) {

            int max_samples = path_tracer->max_samples;

            ImGui::SliderInt("Samples", &path_tracer->max_samples, 1, 5000);

            if (max_samples != path_tracer->max_samples) {
                path_tracer->reset_image();
            }

        }
        if (integrator_mode == 1) {
            ImGui::Text("AA");
            ImGui::Separator();
            ImGui::Text("Diffuse");
            ImGui::Separator();
            ImGui::Text("Specular");
            ImGui::Separator();
        }
    }

    if (ImGui::CollapsingHeader("Paths"))
    {
        ImGui::Text("Path Count");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Paths generated per tile.");

        ImGui::Separator();

        ImGui::Text("Max Bounces");
        //ImGui::DragFloat("##fd", &camera->focal_distance, 0.01f, 0.f, 1000.f, "%.2f");
    }

    if (ImGui::CollapsingHeader("Performance"))
    {
        if (ImGui::TreeNode("Tiles")) {

            int tile_size = path_tracer->tile_size;

            ImGui::InputInt("Tile Size", &tile_size, 1, 1024, ImGuiInputTextFlags_EnterReturnsTrue);

            if (tile_size != path_tracer->tile_size) {
                path_tracer->set_tile_size(tile_size);
                path_tracer->reset_image();
            }

            ImGui::Text("Order");
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    if (ImGui::Button("render")) {
        path_tracer->reset_image();
    }
}

void RenderingContext::render_properties()
{
    if (render_mode == PATHTRACE_MODE) pathtracing_props();
    else raster_props();
}

void RenderingContext::set_width(int width)
{
    this->width = width;
}

void RenderingContext::set_height(int height)
{
    this->height = height;
}

bool RenderingContext::focused()
{
    return currently_interacting;
}

Scene* RenderingContext::get_scene()
{
    return s;
}

dScene* RenderingContext::get_dscene()
{
    return ds;
}

Camera* RenderingContext::get_camera()
{
    return camera;
}

/*
Scene* RenderingContext::get_scene()
{
    return s;
}

Camera* RenderingContext::get_camera()
{
    return camera;
}
*/