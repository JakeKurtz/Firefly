#pragma once

#include <cmath>

#include "Light.h"
#include "Shader.h"
#include "FrameBuffer.h"
#include "Scene.h"
#include "EnvironmentLight.h"
#include "TextureArray.h"

class Rasterizer {

public:
    TextureArray* cascadeShadowMapTexArray;
    TextureArray* cascadeShadowMapTexArray_tmptarget;

    unsigned int depthMap;
    unsigned int depthMapFBO;

    FrameBuffer* fbo;
    FrameBuffer* fbo_sdwmap_tmptarget;
    FrameBuffer* fbo_hdr_tmptarget;
    FrameBuffer* fbo_sdwmap;
    FrameBuffer* fbo_test;
    FrameBuffer* fbo_test2;
    FrameBuffer* gBuffer;

    Shader shaderGeometryPass;
    Shader shaderLightingPass;
    Shader shaderShadowPass;
    Shader shaderShadowDepth;
    Shader gaussFilter;
    Shader bilateralFilter;

    unsigned int uboExampleBlock;

    int shadow_cube_size = 1024;
    int shadow_cascade_size = 4096;

    float cascade_shadow_offset = 250.f;
    float lookat_point_offset = 1.f;
    float lightSize = 250.f;
    float searchAreaSize = 15.f;
    float shadowScaler = 0.05f;
    int kernel_size = 9;

    float scale = 10.f;
    glm::mat4 cascade_proj_mat = glm::ortho(-10.f, 10.f, -10.f, 10.f, 0.1f, 1000.f);
    //glm::mat4 cascade_proj_mat = glm::perspective(glm::radians(45.f), 1.f, 0.01f, 10000.f);
    glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);

    unsigned int screenVAO = 0;
    unsigned int screenVBO;
    float screenVerts[20] = {
        // positions        // texture Coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };

    int SCR_WIDTH;
    int SCR_HEIGHT;

    Rasterizer(int _SCR_WIDTH = 1000, int _SCR_HEIGHT = 1000);

    void draw(Scene* scene);
    void drawGeometry(Scene* scene);
    void drawLighting(Scene* scene);
    void drawShadows(Scene* scene);
    void drawShadowMaps(Scene* scene);

    void setScreenSize(int _SCR_WIDTH, int _SCR_HEIGHT);
    void setShadowCubeSize(int _size);
    void setShadowCascadeSize(int _size);

private:
    void initFBOs();
    void applyFilter(Scene* scene);
    void applyToneMapping(FrameBuffer* targetFB, aType targetAttachment, unsigned int texId);
    void drawScreen();
};