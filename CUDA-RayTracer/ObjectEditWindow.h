#pragma once

#include "RenderingContext.h"
#include "RenderObject.h"

const int OBJ_TYPE_CURVE = 0;
const int OBJ_TYPE_MODEL = 1;

class ObjectEditWindow : public RenderingContext
{
public:
	ObjectEditWindow(int width, int height);

	void init_scene();
	void init_image_buffer();

	void draw();

	void set_curve(Curve* curve);
	void set_model(Model* model);

	void window_size_callback(GLFWwindow* window, int width, int height);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	void key_callback();
	void process_input(GLFWwindow* window);
private:
	Transform* transform_edit;
	Transform* transform_world;
	Curve* curve = nullptr;
	Model* model = nullptr;

	EnvironmentLight* env_light_color;
	EnvironmentLight* env_light_hrd;

	RenderObject* r_obj = nullptr;

	glm::vec3 centroid = glm::vec3(0.f);

	int object_type = OBJ_TYPE_CURVE;

	float yaw = 0.f;
	float pitch = 0.f;
};

