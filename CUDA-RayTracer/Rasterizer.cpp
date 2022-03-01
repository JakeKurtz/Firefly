#include "Rasterizer.h"

Rasterizer::Rasterizer(int _SCR_WIDTH, int _SCR_HEIGHT) :
    shaderGeometryPass("../shaders/main/geometryPass_vs.glsl", "../shaders/main/geometryPass_fs.glsl"),
    shaderLightingPass("../shaders/main/lightingPass_vs.glsl", "../shaders/main/lightingPass_fs.glsl"),
    shaderShadowPass("../shaders/main/shadowPass_vs.glsl", "../shaders/main/shadowPass_fs.glsl"),
    shaderShadowDepth("../shaders/main/shadowDepth_vs.glsl", "../shaders/main/shadowDepth_fs.glsl"),
    gaussFilter("../shaders/main/gaussFilter_vs.glsl", "../shaders/main/gaussFilter_fs.glsl"),
    bilateralFilter("../shaders/main/bilateralFilter_vs.glsl", "../shaders/main/bilateralFilter_fs.glsl")
{
    SCR_WIDTH = _SCR_WIDTH;
    SCR_HEIGHT = _SCR_HEIGHT;

    //blueNoise = new Texture("LDR_RG01_0.png", "../textures/", "noise", GL_TEXTURE_2D);
    cascadeShadowMapTexArray = new TextureArray(shadow_cascade_size, shadow_cascade_size, 4, false);
    cascadeShadowMapTexArray_tmptarget = new TextureArray(shadow_cascade_size, shadow_cascade_size, 4, false);

    uboExampleBlock;
    glGenBuffers(1, &uboExampleBlock);
    glBindBuffer(GL_UNIFORM_BUFFER, uboExampleBlock);
    glBufferData(GL_UNIFORM_BUFFER, 32 * sizeof(glm::mat4), NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferRange(GL_UNIFORM_BUFFER, 0, uboExampleBlock, 0, 32 * sizeof(glm::mat4));

    unsigned int uniformBlockIndexRed = glGetUniformBlockIndex(shaderLightingPass.ID, "ExampleBlock");

    glUniformBlockBinding(shaderLightingPass.ID, uniformBlockIndexRed, 0);

    initFBOs();
}

float lerp(float v0, float v1, float t)
{
    return (1 - t) * v0 + t * v1;
}

void Rasterizer::draw(Scene* scene)
{
    //for (auto obj : scene->models)
        //obj->updateTRS();

    // ------ GEOMETRY PASS ------ //

    gBuffer->bind();

    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    drawGeometry(scene);

    gBuffer->unbind();
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    // ------ SHADOW MAP PASS ------ //

    glCullFace(GL_FRONT);
    drawShadowMaps(scene);
    glCullFace(GL_BACK);

    fbo_test2->bind();
    drawShadows(scene);
    fbo_test2->unbind();
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    // ------ LIGHTING PASS ------ //

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    drawLighting(scene);

    glEnable(GL_DEPTH_TEST);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer->id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, gBuffer->width, gBuffer->height, 0, 0, gBuffer->width, gBuffer->height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ------ BACKGROUND ------ //

    scene->environment_light->draw_background(scene->camera);
}

void Rasterizer::drawGeometry(Scene* scene)
{
    shaderGeometryPass.use();
    for (auto m : scene->models) {
        shaderGeometryPass.setMat4("model", m->model_mat);
        scene->camera->sendUniforms(shaderGeometryPass);
        m->draw(shaderGeometryPass);
    }
}

void Rasterizer::drawLighting(Scene* scene)
{
    shaderLightingPass.use();

    shaderLightingPass.setInt(gBuffer->color_attachments[0]->name, 0);
    shaderLightingPass.setInt(gBuffer->color_attachments[1]->name, 1);
    shaderLightingPass.setInt(gBuffer->color_attachments[2]->name, 2);
    shaderLightingPass.setInt(gBuffer->color_attachments[3]->name, 3);
    shaderLightingPass.setInt(gBuffer->color_attachments[4]->name, 4);
    shaderLightingPass.setInt("gShadows", 5);
    shaderLightingPass.setInt("irradianceMap", 6);
    shaderLightingPass.setInt("prefilterMap", 7);
    shaderLightingPass.setInt("brdfLUT", 8);

    shaderLightingPass.setInt("shadowMaps", 9);

    shaderLightingPass.setFloat("searchAreaSize", searchAreaSize);
    shaderLightingPass.setFloat("lightSize", lightSize);
    shaderLightingPass.setInt("kernel_size", kernel_size);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gBuffer->color_attachments[0]->id);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gBuffer->color_attachments[1]->id);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, gBuffer->color_attachments[2]->id);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, gBuffer->color_attachments[3]->id);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, gBuffer->color_attachments[4]->id);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, fbo_test->color_attachments[0]->id);

    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_CUBE_MAP, scene->environment_light->getIrradianceMapID());

    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_CUBE_MAP, scene->environment_light->getPrefilterMapID());

    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, scene->environment_light->get_brdfLUT_ID());

    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray->getID());

    scene->send_uniforms(shaderLightingPass);

    glm::vec3 cam_lookat_pos = (scene->camera->front * lookat_point_offset) + scene->camera->position;
    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glm::vec3 dir = -scene->dir_lights[i]->getDirection();
        glm::vec3 pos = cam_lookat_pos + (-dir * cascade_shadow_offset);
        glm::mat4 lookat = glm::lookAt(pos, cam_lookat_pos, up);
        glm::mat4 LSM = cascade_proj_mat * lookat;

        glBindBuffer(GL_UNIFORM_BUFFER, uboExampleBlock);
        glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(LSM));
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
    drawScreen();
}

void Rasterizer::drawShadows(Scene* scene)
{
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_test2->color_attachments[0]->id, 0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    shaderShadowPass.use();

    shaderShadowPass.setInt(gBuffer->color_attachments[0]->name, 0);
    shaderShadowPass.setInt("shadowMaps", 1);

    shaderShadowPass.setFloat("searchAreaSize", searchAreaSize);
    shaderShadowPass.setFloat("lightSize", lightSize);
    shaderShadowPass.setFloat("shadowScaler", shadowScaler);
    shaderShadowPass.setInt("kernel_size", kernel_size);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gBuffer->color_attachments[0]->id);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray->getID());

    scene->send_uniforms(shaderShadowPass);

    glm::vec3 cam_lookat_pos = (scene->camera->front * lookat_point_offset) + scene->camera->position;
    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glm::vec3 dir = -scene->dir_lights[i]->getDirection();
        glm::vec3 pos = cam_lookat_pos + (-dir * cascade_shadow_offset);
        glm::mat4 lookat = glm::lookAt(pos, cam_lookat_pos, up);
        glm::mat4 LSM = cascade_proj_mat * lookat;

        glBindBuffer(GL_UNIFORM_BUFFER, uboExampleBlock);
        glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(LSM));
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
    drawScreen();

    glDisable(GL_DEPTH_TEST);

    // ----- First pass ----- //

    fbo_test->bind();
    bilateralFilter.use();

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_test->color_attachments[0]->id, 0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fbo_test2->color_attachments[0]->id);

    bilateralFilter.setBool("horizontal", true);
    bilateralFilter.setInt("image", 0);
    bilateralFilter.setVec3("scale", glm::vec3(1.f / (SCR_WIDTH * 0.1f), 0.f, 0.f));
    bilateralFilter.setFloat("r", 16.f);
    drawScreen();

    fbo_test->unbind();
    /*
    // ----- Second pass ----- //

    fbo_test2->bind();
    gaussFilter.use();

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_test2->color_attachments[0]->id, 0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fbo_test->color_attachments[0]->id);

    gaussFilter.setBool("horizontal", false);
    gaussFilter.setInt("image", 0);
    gaussFilter.setVec3("scale", vec3(0.f, 1.f / (SCR_HEIGHT * 0.1f), 0.f));
    gaussFilter.setFloat("r", 16.f);
    drawScreen();

    fbo_test2->unbind();

    glEnable(GL_DEPTH_TEST);
    */
}

void Rasterizer::drawShadowMaps(Scene* scene)
{
    glm::vec3 cam_lookat_pos = (scene->camera->front * lookat_point_offset) + scene->camera->position;

    fbo_sdwmap->bind();
    shaderShadowDepth.use();

    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cascadeShadowMapTexArray->getID(), 0, i);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glm::vec3 dir = -scene->dir_lights[i]->getDirection();
        glm::vec3 pos = cam_lookat_pos + (-dir * cascade_shadow_offset);
        glm::mat4 lookat = glm::lookAt(pos, cam_lookat_pos, up);
        //glm::mat4 lookat = glm::lookAt(glm::vec3(15.f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 LSM = cascade_proj_mat * lookat;

        shaderShadowDepth.setMat4("lsm", LSM);
        for (auto m : scene->models) {
            shaderShadowDepth.setMat4("model", m->model_mat);
            m->draw(shaderShadowDepth);
        }
    }
    fbo_sdwmap->unbind();

    glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray->getID());
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
}

void Rasterizer::setScreenSize(int _SCR_WIDTH, int _SCR_HEIGHT)
{
    SCR_WIDTH = _SCR_WIDTH;
    SCR_HEIGHT = _SCR_HEIGHT;
}

void Rasterizer::setShadowCubeSize(int _size)
{
}

void Rasterizer::setShadowCascadeSize(int _size)
{
}

void Rasterizer::initFBOs()
{
    fbo = new FrameBuffer(SCR_WIDTH, SCR_HEIGHT);
    fbo->attach(GL_COLOR_ATTACHMENT0, GL_RGBA16F, GL_RGBA, GL_FLOAT);
    fbo->attach(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo->construct();

    fbo_sdwmap_tmptarget = new FrameBuffer(shadow_cascade_size, shadow_cascade_size);
    fbo_sdwmap_tmptarget->attach(GL_COLOR_ATTACHMENT0, GL_RG32F, GL_RG, GL_FLOAT);
    fbo_sdwmap_tmptarget->attach(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_sdwmap_tmptarget->construct();

    fbo_sdwmap = new FrameBuffer(shadow_cascade_size, shadow_cascade_size);
    fbo_sdwmap->attach(GL_COLOR_ATTACHMENT0, GL_RG32F, GL_RG, GL_FLOAT);
    fbo_sdwmap->attach(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_sdwmap->construct();

    fbo_test = new FrameBuffer(SCR_WIDTH, SCR_HEIGHT);
    fbo_test->attach(GL_COLOR_ATTACHMENT0, GL_RED, GL_RED, GL_UNSIGNED_BYTE);
    fbo_test->attach(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_test->construct();

    fbo_test2 = new FrameBuffer(SCR_WIDTH, SCR_HEIGHT);
    fbo_test2->attach(GL_COLOR_ATTACHMENT0, GL_RED, GL_RED, GL_UNSIGNED_BYTE);
    fbo_test2->attach(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    fbo_test2->construct();

    // Setup defferred shading
    gBuffer = new FrameBuffer(SCR_WIDTH, SCR_HEIGHT);
    gBuffer->attach(GL_COLOR_ATTACHMENT0, GL_RGBA16F, GL_RGBA, GL_FLOAT, "gPosition");
    gBuffer->attach(GL_COLOR_ATTACHMENT1, GL_RGBA16F, GL_RGBA, GL_FLOAT, "gNormal");
    gBuffer->attach(GL_COLOR_ATTACHMENT2, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, "gAlbedo");
    gBuffer->attach(GL_COLOR_ATTACHMENT3, GL_RGBA16F, GL_RGBA, GL_FLOAT, "gMetallicRoughAO");
    gBuffer->attach(GL_COLOR_ATTACHMENT4, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, "gEmissive");
    gBuffer->attach(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
    gBuffer->construct();
}

void Rasterizer::applyFilter(Scene* scene)
{
    glDisable(GL_DEPTH_TEST);

    fbo_sdwmap_tmptarget->bind();
    gaussFilter.use();

    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cascadeShadowMapTexArray_tmptarget->getID(), 0, i);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray->getID());

        gaussFilter.setBool("horizontal", true);
        gaussFilter.setInt("shadowMaps", 0);
        gaussFilter.setInt("index", i);
        gaussFilter.setVec3("scale", glm::vec3(1.f / (2048 * 0.25f), 0.f, 0.f));
        drawScreen();

    }
    fbo_sdwmap_tmptarget->unbind();

    // ----- Second pass ----- //

    fbo_sdwmap->bind();
    gaussFilter.use();

    for (int i = 0; i < scene->dir_lights.size(); i++)
    {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cascadeShadowMapTexArray->getID(), 0, i);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, cascadeShadowMapTexArray_tmptarget->getID());

        gaussFilter.setBool("horizontal", false);
        gaussFilter.setInt("shadowMaps", 0);
        gaussFilter.setInt("index", i);
        gaussFilter.setVec3("scale", glm::vec3(0.f, 1.f / (2048 * 0.25f), 0.f));
        drawScreen();
    }
    fbo_sdwmap->unbind();

    glEnable(GL_DEPTH_TEST);
}

void Rasterizer::applyToneMapping(FrameBuffer* targetFB, aType targetAttachment, unsigned int texId)
{
    /*glDisable(GL_DEPTH_TEST);

    targetFB->bind(targetAttachment);
    hdrShader.use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texId);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    hdrShader.setFloat("exposure", camera->Exposure);
    //hdrShader.setFloat("lum", lum);
    screen->model.Draw(hdrShader);

    glEnable(GL_DEPTH_TEST);
    */
}

void Rasterizer::drawScreen()
{
    if (screenVAO == 0)
    {
        glGenVertexArrays(1, &screenVAO);
        glGenBuffers(1, &screenVBO);
        glBindVertexArray(screenVAO);
        glBindBuffer(GL_ARRAY_BUFFER, screenVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(screenVerts), &screenVerts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(screenVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}
