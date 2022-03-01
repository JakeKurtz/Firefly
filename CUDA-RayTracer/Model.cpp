#pragma once

#include "Model.h"

Model::Model(vector<Mesh*> meshes, glm::mat4x4 model_mat)
{
	this->meshes = meshes;
	this->model_mat = model_mat;
}

void Model::draw(Shader& shader)
{
	for (unsigned int i = 0; i < meshes.size(); i++)
		meshes[i]->draw(shader);
}

std::vector<Mesh*> Model::get_meshes()
{
	return meshes;
}

void Model::set_directory(string dir)
{
	directory = dir;
}

string Model::get_directory()
{
	return directory;
}
