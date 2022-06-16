#pragma once

#include "RenderingContext.h"

class SceneViewWindow : public RenderingContext
{
public:
    SceneViewWindow(int width, int height, Scene* s, dScene* ds);
    void draw();
    void set_raster_output_type(int t);

	void window_size_callback(GLFWwindow* window, int width, int height);
	void mouse_callback(GLFWwindow* window, double xpos, double ypo);
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	void key_callback();
	void process_input(GLFWwindow* window);
};

