#pragma once
#include "Light.h"

class DirectionalLight :
	public Light
{
private:
	glm::vec3 direction;

public:
	DirectionalLight();
	DirectionalLight(glm::vec3 direction, glm::vec3 color);

	void setDirection(glm::vec3 _direction);
	glm::vec3 getDirection();
};

