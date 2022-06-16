#include "Material.h"
#include "globals.h"

Material::Material()
{
	id = gen_id();
}

void Material::send_uniforms(Shader& shader)
{
	shader.setVec3("baseColorFactor", baseColorFactor);
	shader.setVec3("emissiveColorFactor", emissiveColorFactor);

	shader.setFloat("roughnessFactor", roughnessFactor);
	shader.setFloat("metallicFactor", metallicFactor);

	shader.setBool("baseColorTexture_sample", (baseColorTexture != nullptr));
	shader.setBool("normalTexture_sample", (normalTexture != nullptr));
	shader.setBool("occlusionTexture_sample", (occlusionTexture != nullptr));
	shader.setBool("emissiveTexture_sample", (emissiveTexture != nullptr));
	shader.setBool("metallicRughnessTexture_sample", (metallicRoughnessTexture != nullptr));
	shader.setBool("roughnessTexture_sample", (roughnessTexture != nullptr));
	shader.setBool("metallicTexture_sample", (metallicTexture != nullptr));

	shader.setInt("baseColorTexture", 0);
	shader.setInt("normalTexture", 1);
	shader.setInt("occlusionTexture", 2);
	shader.setInt("emissiveTexture", 3);
	shader.setInt("metallicRoughnessTexture", 4);
	shader.setInt("roughnessTexture", 5);
	shader.setInt("metallicTexture", 6);

	if (baseColorTexture != nullptr) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, baseColorTexture->id);
	}

	if (normalTexture != nullptr) {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, normalTexture->id);
	}

	if (occlusionTexture != nullptr) {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, occlusionTexture->id);
	}

	if (emissiveTexture != nullptr) {
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, emissiveTexture->id);
	}

	if (metallicRoughnessTexture != nullptr) {
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, metallicRoughnessTexture->id);
	}

	if (roughnessTexture != nullptr) {
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, roughnessTexture->id);
	}

	if (metallicTexture != nullptr) {
		glActiveTexture(GL_TEXTURE6);
		glBindTexture(GL_TEXTURE_2D, metallicTexture->id);
	}
}
