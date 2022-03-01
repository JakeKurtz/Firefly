#pragma once
#include "GLCommon.h"

class Geometry
{
public:
	glm::vec3 position = glm::vec3(0.f);;
	glm::vec3 scale = glm::vec3(1.f);
	glm::vec3 rotate;
	glm::mat4 model_mat = glm::mat4();

	void updateTRS();
};

