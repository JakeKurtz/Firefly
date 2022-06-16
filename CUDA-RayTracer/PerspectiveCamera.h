#pragma once
#include "Camera.h"

class PerspectiveCamera : public Camera
{
private:
	float aspectRatio;
	float yfov;

public:
	PerspectiveCamera();
	PerspectiveCamera(float aspectRatio, float yfov, float znear, float zfar);
	PerspectiveCamera(glm::vec3 position, float aspectRatio, float yfov, float znear, float zfar);

	void set_aspect_ratio(float aspect_ratio);
};

