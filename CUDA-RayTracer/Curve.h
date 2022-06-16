#pragma once

#include <string>
#include <vector>

#include "RenderObject.h"

#include "GLCommon.h"
#include "globals.h"
#include "Shader.h"
#include "Material.h"
#include "Transform.h"

class Curve : public RenderObject
{
public:
	Curve();
	Curve(std::vector<glm::vec3> vertices);

	void draw(Shader& shader);

	void set_depth(int depth);
	int get_depth();

	void center();

	void virtual update();

protected:

	std::vector<glm::vec3> control_points;
	std::vector<glm::vec3> vertices;

	// mesh Data
	Material* material;
	int degree = 3;
	int depth = 3;

	// render data 
	unsigned int VAO_cp, VAO_curve;
	unsigned int VBO_cp, VBO_curve;
	int nmb_triangles = 0;

	std::vector<glm::vec3> subdiv_open(std::vector<glm::vec3> const& points, int degree, int depth);
	void setup_buffers();
};

