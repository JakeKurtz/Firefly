#include "DirectionalLight.h"

DirectionalLight::DirectionalLight()
{
	direction = glm::vec3(1.f);
	color = glm::vec3(1.f);
}

DirectionalLight::DirectionalLight(glm::vec3 _direction, glm::vec3 _color)
{
	direction = _direction;
	color = _color;
}

void DirectionalLight::setDirection(glm::vec3 _direction) { direction = glm::normalize(_direction); }

glm::vec3 DirectionalLight::getDirection() { return direction; }
