#include "Light.h"

Light::Light()
{
}

std::string Light::getName() { return std::string(); }

int Light::getId() { return id; }

glm::vec3 Light::getColor() { return color; }

float Light::getIntensity() { return intensity; }

float Light::getRange() { return range; }

void Light::setColor(glm::vec3 _color) { color = _color; }

void Light::setIntensity(float _intensity) { intensity = _intensity; }

void Light::setRange(float _range) { range = _range; }
