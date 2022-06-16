#pragma once

#include "Model.h"
#include "globals.h"

Model::Model(vector<Mesh*> meshes, glm::mat4x4 model_mat)
{
	this->meshes = meshes;
	this->transform->set_matrix(model_mat);
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

float signed_volume_of_tetrahedron(glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
	return glm::dot(a, glm::cross(b, c)) / 6.f;
}

glm::vec3 center_of_mass(Model* model)
{
	float volume = 0.f;
	glm::vec3 centroid = glm::vec3(0.f);

	for (auto mesh : model->get_meshes()) {

		auto indices = mesh->get_indices();
		auto vertices = mesh->get_vertices();

		for (int i = 0; i < indices.size(); i += 3) {
			auto iv0 = indices[i];
			auto iv1 = indices[i + 1];
			auto iv2 = indices[i + 2];

			auto v0 = vertices[iv0];
			auto v1 = vertices[iv1];
			auto v2 = vertices[iv2];

			float signed_volume = signed_volume_of_tetrahedron(v0.position, v1.position, v2.position);

			volume += signed_volume;
			centroid += signed_volume * (v0.position + v1.position + v2.position) / 4.f;
		}
	}
	return centroid / volume;
}

glm::vec3 center_of_mass(vector<Vertex> vertices, std::vector<unsigned int> indices)
{

	float volume = 0.f;
	glm::vec3 centroid = glm::vec3(0.f);

	for (int i = 0; i < indices.size(); i += 3) {
		auto iv0 = indices[i];
		auto iv1 = indices[i + 1];
		auto iv2 = indices[i + 2];

		auto v0 = vertices[iv0];
		auto v1 = vertices[iv1];
		auto v2 = vertices[iv2];

		float signed_volume = signed_volume_of_tetrahedron(v0.position, v1.position, v2.position);

		volume += signed_volume;
		centroid += volume * (v0.position + v1.position + v2.position) / 4.f;
	}

	return centroid / volume;
}

glm::vec3 center_of_mass(vector<glm::vec3> vertices)
{
	return glm::vec3();
}

glm::vec3 center_of_geometry(vector<Vertex> vertices, std::vector<unsigned int> indices)
{
	return glm::vec3();
}

glm::vec3 center_of_geometry(vector<glm::vec3> vertices)
{
	return glm::vec3();
}
