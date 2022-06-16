#pragma once

#include "GLCommon.h"
#include <string>

class Light
{
protected:
	std::string name;
	int id;

	glm::vec3 color;
	float intensity = 1.f;
	float range = 100.f;

public:
	Light();

	std::string get_name();
	int getId();
	glm::vec3 getColor();
	float getIntensity();
	float getRange();

	void setColor(glm::vec3 _color);
	void setIntensity(float _intensity);
	void setRange(float _range);
};

