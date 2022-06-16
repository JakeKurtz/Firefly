#include "DirectionalLight.h"
#include "globals.h"

DirectionalLight::DirectionalLight()
{
	id = gen_id();
	direction = glm::vec3(1.f);
	color = glm::vec3(1.f);
}

DirectionalLight::DirectionalLight(glm::vec3 _direction, glm::vec3 _color)
{
	id = gen_id();
	direction = _direction;
	color = _color;
}

void DirectionalLight::setDirection(glm::vec3 _direction) { direction = glm::normalize(_direction); }

glm::vec3 DirectionalLight::getDirection() { return direction; }
