#pragma once
#include <string>
#include "Transform.h"
#include "Shader.h"
#include "globals.h"

const int TYPE_CURVE = 0;
const int TYPE_TRIANGLE_MESH = 1;

class RenderObject
{
public:
	int get_id();

	void set_name(std::string name);
	std::string get_name();

	void set_transform(Transform* transform);
	Transform* get_transform();

	int type();

	void virtual draw(Shader& shader) = 0;

protected:
	int id = gen_id();
	std::string name;
	int obj_type;

	Transform* transform = new Transform();
};

