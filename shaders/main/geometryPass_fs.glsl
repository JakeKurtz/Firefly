#version 330 core
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec3 gAlbedo;
layout(location = 3) out vec3 gMetallicRoughAO;
layout(location = 4) out vec3 gEmissive;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;
in mat3 TBN;

uniform vec3 baseColorFactor;
uniform vec3 emissiveColorFactor;
uniform float roughnessFactor;
uniform float metallicFactor;

uniform sampler2D baseColorTexture;
uniform sampler2D normalTexture;
uniform sampler2D occlusionTexture;
uniform sampler2D emissiveTexture;
uniform sampler2D metallicRoughnessTexture;
uniform sampler2D roughnessTexture;
uniform sampler2D metallicTexture;

uniform bool baseColorTexture_sample;
uniform bool normalTexture_sample;
uniform bool occlusionTexture_sample;
uniform bool emissiveTexture_sample;
uniform bool metallicRughnessTexture_sample;
uniform bool roughnessTexture_sample;
uniform bool metallicTexture_sample;

void main()
{
    gPosition = FragPos;
    
    // BASE COLOR
    if (baseColorTexture_sample)
        gAlbedo = texture(baseColorTexture, TexCoords).rgb;
    else
        gAlbedo = baseColorFactor;

    // NORMAL
    if (normalTexture_sample) {
        gNormal = texture(normalTexture, TexCoords).rgb;
        gNormal = gNormal * 2.0 - 1.0;
        gNormal = normalize(TBN * gNormal);
    }
    else
        gNormal = normalize(Normal);

    // EMISSIVE
    if (emissiveTexture_sample)
        gEmissive = texture(emissiveTexture, TexCoords).rgb;
    else
        gEmissive = vec3(0.f);

    if (metallicRughnessTexture_sample) {
        gMetallicRoughAO = texture(metallicRoughnessTexture, TexCoords).rgb;
    } else {
        // OCCLUSION
        if (occlusionTexture_sample)
            gMetallicRoughAO.r = texture(occlusionTexture, TexCoords).r;
        else
            gMetallicRoughAO.r = 1.f;

        // ROUGHNESS
        if (roughnessTexture_sample)
            gMetallicRoughAO.g = texture(roughnessTexture, TexCoords).r;
        else
            gMetallicRoughAO.g = roughnessFactor;

        // METALLIC
        if (metallicTexture_sample)
            gMetallicRoughAO.b = texture(metallicTexture, TexCoords).r;
        else
            gMetallicRoughAO.b = metallicFactor;
    }
}