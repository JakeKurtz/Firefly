#pragma once

#include "GLCommon.h"

#include "Texture.h"
#include "Shader.h"

#include <string>

struct Material
{
	int id;
	std::string name;

	int alphaMode;
	float alphaCutoff;

	bool doubleSided;

	glm::vec3 baseColorFactor;
	glm::vec3 emissiveColorFactor;
	glm::vec3 fresnel;

	float roughnessFactor = 1.f;
	float metallicFactor = 0.f;

	Texture* baseColorTexture;
	Texture* normalTexture;
	Texture* occlusionTexture;
	Texture* emissiveTexture;
	Texture* roughnessTexture;
	Texture* metallicTexture;
	Texture* metallicRoughnessTexture;

	Material();
	void send_uniforms(Shader& shader);
};

