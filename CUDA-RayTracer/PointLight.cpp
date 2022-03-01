#include "PointLight.h"

PointLight::PointLight()
{
	position = glm::vec3(0.f);
	color = glm::vec3(1.f);
}

PointLight::PointLight(glm::vec3 _position, glm::vec3 _color)
{
	position = _position;
	color = _color;
}

glm::vec3 PointLight::getPosition() { return position; }

void PointLight::setPosition(glm::vec3 _position) { position = _position; }