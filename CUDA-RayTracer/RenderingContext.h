#pragma once
#include <stb_image_write.h>
#include "../imgui/imgui.h"

#include "Camera.h"
#include "PerspectiveCamera.h"
#include "Scene.h"
#include "Rasterizer.h"
#include "PathTracer.h"

const int RASTER_MODE = 0;
const int PATHTRACE_MODE = 1;
const int WIRE_MODE = 2;

const int SHADED	= 0;
const int POSITION	= 1;
const int NORMAL	= 2;
const int ALBEDO	= 3;
const int MRO		= 4;
const int EMISSION	= 5;
const int DEPTH		= 6;

enum class RenderMode { Raster, PathTrace };

class RenderingContext
{
public:

	RenderingContext();
	RenderingContext(int width, int height, Scene* s, dScene* ds, Camera* camera);

	virtual void draw() = 0;

	virtual void window_size_callback(GLFWwindow* window, int width, int height) = 0;
	virtual void mouse_callback(GLFWwindow* window, double xpos, double ypos) = 0;
	virtual void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) = 0;
	virtual void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) = 0;
	virtual void key_callback() = 0;
	virtual void process_input(GLFWwindow* window) = 0;

	void frame_time();

	GLuint render_raster();
	GLuint render_pathtrace();

	void raster_props();
	void pathtracing_props();

	void render_properties();

	void set_width(int width);
	void set_height(int height);

	bool focused();

	//void set_scene(Scene* s);
	//void set_dscene(dScene* ds);
	//void set_camera(Camera* c);

	Scene* get_scene();
	dScene* get_dscene();
	Camera* get_camera();

protected:

	std::string window_name;

	int width;
	int height;

	bool currently_interacting = true;
	bool buffer_reset = false;

	GLuint image_texture;
	char* image_buffer;

	Camera* camera;				// 

	Scene* s;					// Scene data
	dScene* ds;					// Scene data loaded on GPU

	Rasterizer* raster;			// OpenGL render pipeline
	PathTracer* path_tracer;	// CUDA Monte Carlo path tracer

	int render_mode = RASTER_MODE;
	int integrator_mode = 0;
	int raster_output_type = 0;

	float delta_time = 0.f;
	float last_frame_time = 0.f;
	float current_frame_time = 0.f;

	float raw_mouse_scroll = 1;
	float image_scale = 1.f;
	float image_pan_x = 0;
	float image_pan_y = 0;

	float lastX;
	float lastY;
	bool firstMouse = true;
	bool mouseDown = false;
	bool mouse_button_3_down = false;
	bool click = false;
};

