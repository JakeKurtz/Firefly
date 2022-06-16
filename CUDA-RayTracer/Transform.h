#pragma once
#include "GLCommon.h"

class Transform
{
public:

	Transform();

	void translate(float x, float y, float z);
	void translate(glm::vec3 p);
	
	void rotate(float x, float y, float z);
	void rotate(glm::vec3 r);

	void scale(float x, float y, float z);
	void scale(float s);
	void scale(glm::vec3 s);

	void apply();
	void reset();

	glm::mat4 get_matrix();
	void set_matrix(glm::mat4 matrix);

	void set_centroid(glm::vec3 centroid);

protected:
	glm::vec3 centroid;
	glm::vec3 p = glm::vec3(0.f);
	glm::vec3 s = glm::vec3(1.f);
	glm::vec3 r = glm::vec3(0.f);
	glm::mat4 matrix = glm::mat4(1.f);
};

