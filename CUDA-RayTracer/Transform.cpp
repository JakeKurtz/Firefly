#include "Transform.h"

Transform::Transform() 
{
	centroid = glm::vec3(0.f);

	p = glm::vec3(0.f);
	s = glm::vec3(1.f);
	r = glm::vec3(0.f);
	matrix = glm::mat4(1.f);
}

void Transform::apply()
{
	matrix = glm::translate(matrix, centroid);

	matrix = glm::translate(matrix, p);
	p = glm::vec3(0.f);

	matrix = glm::rotate(matrix, r.x, glm::vec3(1,0,0));
	matrix = glm::rotate(matrix, r.y, glm::vec3(0,1,0));
	matrix = glm::rotate(matrix, r.z, glm::vec3(0,0,1));
	r = glm::vec3(0.f);

	matrix = glm::scale(matrix, s);
	s = glm::vec3(1.f);

	matrix = glm::translate(matrix, -centroid);
}

void Transform::reset()
{
	matrix = glm::mat4(1.f);
}

glm::mat4 Transform::get_matrix()
{
	return matrix;
}

void Transform::set_matrix(glm::mat4 matrix)
{
	(this)->matrix = matrix;
}

void Transform::set_centroid(glm::vec3 centroid)
{
	(this)->centroid = centroid;
}

void Transform::translate(float x, float y, float z)
{
	(this)->p = glm::vec3(x,y,z);
}

void Transform::translate(glm::vec3 p)
{
	(this)->p = p;
}

void Transform::rotate(float x, float y, float z)
{
	(this)->r = glm::vec3(x, y, z);
}

void Transform::rotate(glm::vec3 rotate)
{
	(this)->r = rotate;
}

void Transform::scale(float x, float y, float z)
{
	(this)->r = glm::vec3(x, y, z);
}

void Transform::scale(float s)
{
	(this)->s = glm::vec3(s);
}

void Transform::scale(glm::vec3 s)
{
	(this)->s = s;
}