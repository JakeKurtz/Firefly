#pragma once
#include "GLCommon.h"

#include "Camera.h"
#include "CubeMap.h"
#include "FrameBuffer.h"

class EnvironmentLight
{
protected:
    unsigned int skyboxVAO, skyboxVBO;
    float skyboxVerts[108] =
    {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };

    unsigned int screenVAO, screenVBO;
    float screenVerts[20] = {
        // positions        // texture Coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };

    glm::mat4 captureViews[6] =
    {
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };

    glm::vec3 captureDir[6] =
    {
        glm::vec3(1.0f,  0.0f,  0.0f),
        glm::vec3(-1.0f, 0.0f,  0.0f),
        glm::vec3(0.0f,  1.0f,  0.0f),
        glm::vec3(0.0f, -1.0f,  0.0f),
        glm::vec3(0.0f,  0.0f, 1.0f),
        glm::vec3(0.0f,  0.0f,  -1.0f)
    };

    glm::vec3 captureUp[6] =
    {
        glm::vec3(0.0f,  1.0f,  0.0f),
        glm::vec3(0.0f,  1.0f,  0.0f),
        glm::vec3(0.0f,  0.0f,  1.0f),
        glm::vec3(0.0f,  0.0f,  -1.0f),
        glm::vec3(0.0f,  1.0f,  0.0f),
        glm::vec3(0.0f,  1.0f,  0.0f)
    };

    glm::vec3 captureRight[6] =
    {
        glm::vec3(0.0f,  0.0f,  1.0f),
        glm::vec3(0.0f,  0.0f,  -1.0f),
        glm::vec3(1.0f,  0.0f,  0.0f),
        glm::vec3(1.0f,  0.0f,  0.0f),
        glm::vec3(-1.0f,  0.0f,  0.0f),
        glm::vec3(1.0f,  0.0f,  0.0f)
    };

    glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);

    FrameBuffer* fbo;

    Texture* hdri_enviromentMap;
    glm::vec3 color = glm::vec3(0.8);

    CubeMap* environmentMap;
    CubeMap* irradianceMap;
    CubeMap* prefilterMap;

    Shader atmosphereShader;
    Shader basicBackgroundShader;
    Shader equirectangularToCubemapShader;
    Shader backgroundShader;
    Shader irradianceShader;
    Shader prefilterShader;
    Shader brdfShader;

    int size;

    void init_buffers()
    {
        // skybox VAO
        glGenVertexArrays(1, &skyboxVAO);
        glGenBuffers(1, &skyboxVBO);
        glBindVertexArray(skyboxVAO);
        glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVerts), &skyboxVerts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

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

    void init_fbo()
    {
        fbo = new FrameBuffer(size, size);
        fbo->attach(GL_COLOR_ATTACHMENT0, GL_RG16F, GL_RG, GL_FLOAT); // used for brdfLUT
        fbo->attach(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT);
        fbo->attach_rbo(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24, size, size);
        fbo->construct();
    }

    void build_environmentMap_color()
    {
        basicBackgroundShader.use();
        basicBackgroundShader.setMat4("projection", captureProjection);
        basicBackgroundShader.setVec3("color", color);

        fbo->bind();
        for (unsigned int i = 0; i < 6; ++i)
        {
            equirectangularToCubemapShader.setMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, environmentMap->getID(), 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            draw_skybox();
        }
        fbo->unbind();

        glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    }

    void build_environmentMap_texture()
    {
        equirectangularToCubemapShader.use();
        equirectangularToCubemapShader.setInt("equirectangularMap", 0);
        equirectangularToCubemapShader.setMat4("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, hdri_enviromentMap->getID());

        fbo->bind();
        for (unsigned int i = 0; i < 6; ++i)
        {
            equirectangularToCubemapShader.setMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, environmentMap->getID(), 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            draw_skybox();
        }
        fbo->unbind();

        glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    }

    void build_irradianceMap()
    {
        irradianceShader.use();
        irradianceShader.setInt("equirectangularMap", 0);
        irradianceShader.setMat4("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());

        fbo->bind(32, 32);
        for (unsigned int i = 0; i < 6; ++i)
        {
            irradianceShader.setMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap->getID(), 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            draw_skybox();
        }
        fbo->unbind();
    }

    void build_prefilterMap()
    {
        prefilterShader.use();
        prefilterShader.setInt("environmentMap", 0);
        prefilterShader.setMat4("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());

        fbo->bind();
        unsigned int maxMipLevels = 5;
        for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
        {
            // reisze framebuffer according to mip-level size.
            unsigned int mipWidth = 128 * std::pow(0.5, mip);
            unsigned int mipHeight = 128 * std::pow(0.5, mip);

            fbo->bind_rbo(mipWidth, mipHeight);

            float roughness = (float)mip / (float)(maxMipLevels - 1);
            prefilterShader.setFloat("roughness", roughness);
            for (unsigned int i = 0; i < 6; ++i)
            {
                prefilterShader.setMat4("view", captureViews[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterMap->getID(), mip);

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                draw_skybox();
            }
        }
        fbo->unbind();
    }

    void init_brdfLUT()
    {
        fbo->bind();
        fbo->bind_rbo(fbo->width, fbo->height);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo->color_attachments[0]->id, 0);

        brdfShader.use();
        draw_screen();

        fbo->unbind();
    }

    void draw_skybox()
    {
        glBindVertexArray(skyboxVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
    }

    void draw_screen()
    {
        glBindVertexArray(screenVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }

public:
    EnvironmentLight() :
        backgroundShader("../shaders/environment_map/background_vs.glsl", "../shaders/environment_map/background_fs.glsl"),
        irradianceShader("../shaders/environment_map/irradiance_convolution_vs.glsl", "../shaders/environment_map/irradiance_convolution_fs.glsl"),
        prefilterShader("../shaders/environment_map/prefilter_vs.glsl", "../shaders/environment_map/prefilter_fs.glsl"),
        equirectangularToCubemapShader("../shaders/environment_map/equirectangularToCubemap_vs.glsl", "../shaders/environment_map/equirectangularToCubemap_fs.glsl"),
        brdfShader("../shaders/environment_map/brdfLUT_vs.glsl", "../shaders/environment_map/brdfLUT_fs.glsl"),
        basicBackgroundShader("../shaders/environment_map/basicBackground_vs.glsl", "../shaders/environment_map/basicBackground_fs.glsl"),
        atmosphereShader("../shaders/environment_map/atmo_vs.glsl", "../shaders/environment_map/atmo_fs.glsl")
    {
        size = 4;

        init_buffers();
        init_fbo();

        environmentMap = new CubeMap(4, true, false, "environment_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
        build_environmentMap_color();

        irradianceMap = environmentMap;
        prefilterMap = environmentMap;

        init_brdfLUT();
    }

    EnvironmentLight(glm::vec3 _color) :
        backgroundShader("../shaders/environment_map/background_vs.glsl", "../shaders/environment_map/background_fs.glsl"),
        irradianceShader("../shaders/environment_map/irradiance_convolution_vs.glsl", "../shaders/environment_map/irradiance_convolution_fs.glsl"),
        prefilterShader("../shaders/environment_map/prefilter_vs.glsl", "../shaders/environment_map/prefilter_fs.glsl"),
        equirectangularToCubemapShader("../shaders/environment_map/equirectangularToCubemap_vs.glsl", "../shaders/environment_map/equirectangularToCubemap_fs.glsl"),
        brdfShader("../shaders/environment_map/brdfLUT_vs.glsl", "../shaders/environment_map/brdfLUT_fs.glsl"),
        basicBackgroundShader("../shaders/environment_map/basicBackground_vs.glsl", "../shaders/environment_map/basicBackground_fs.glsl"),
        atmosphereShader("../shaders/environment_map/atmo_vs.glsl", "../shaders/environment_map/atmo_fs.glsl")
    {
        size = 4;
        color = _color;

        init_buffers();
        init_fbo();

        environmentMap = new CubeMap(4, true, false, "environment_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
        build_environmentMap_color();

        irradianceMap = environmentMap;
        prefilterMap = environmentMap;

        init_brdfLUT();
    }

    EnvironmentLight(string const& path) :
        backgroundShader("../shaders/environment_map/background_vs.glsl", "../shaders/environment_map/background_fs.glsl"),
        irradianceShader("../shaders/environment_map/irradiance_convolution_vs.glsl", "../shaders/environment_map/irradiance_convolution_fs.glsl"),
        prefilterShader("../shaders/environment_map/prefilter_vs.glsl", "../shaders/environment_map/prefilter_fs.glsl"),
        equirectangularToCubemapShader("../shaders/environment_map/equirectangularToCubemap_vs.glsl", "../shaders/environment_map/equirectangularToCubemap_fs.glsl"),
        brdfShader("../shaders/environment_map/brdfLUT_vs.glsl", "../shaders/environment_map/brdfLUT_fs.glsl"),
        basicBackgroundShader("../shaders/environment_map/basicBackground_vs.glsl", "../shaders/environment_map/basicBackground_fs.glsl"),
        atmosphereShader("../shaders/environment_map/atmo_vs.glsl", "../shaders/environment_map/atmo_fs.glsl")
    {
        size = 512;
        hdri_enviromentMap = new Texture(path, GL_TEXTURE_2D, true);

        init_buffers();
        init_fbo();

        environmentMap = new CubeMap(512, true, false, "environment_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
        build_environmentMap_texture();

        irradianceMap = new CubeMap(32, false, false, "irradiance_map", GL_CLAMP_TO_EDGE);
        build_irradianceMap();

        prefilterMap = new CubeMap(128, true, false, "pre_filter_map", GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
        build_prefilterMap();

        init_brdfLUT();
    }

    unsigned int getCubeMapID()
    {
        return environmentMap->getID();
    }

    unsigned int getIrradianceMapID()
    {
        return irradianceMap->getID();
    }

    unsigned int getPrefilterMapID()
    {
        return prefilterMap->getID();
    }

    unsigned int get_brdfLUT_ID()
    {
        return fbo->color_attachments[0]->id;
    }

    void draw_background(Camera* camera)
    {
        glDepthFunc(GL_LEQUAL);

        backgroundShader.use();
        camera->sendUniforms2(backgroundShader);

        // skybox cube
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap->getID());
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS);
    }

    void update(glm::vec3 _color)
    {
        color = _color;
        build_environmentMap_color();
    }

    void update(string const& path)
    {
        hdri_enviromentMap = new Texture(path, GL_TEXTURE_2D, true);
        build_environmentMap_color();
        build_irradianceMap();
        build_prefilterMap();
    }

    void update(Texture* texture)
    {
        hdri_enviromentMap = texture;
        build_environmentMap_color();
        build_irradianceMap();
        build_prefilterMap();
    }

    std::string get_texture_filepath() {
        return hdri_enviromentMap->filepath;
    }

    glm::vec3 get_color() {
        return color;
    }

    int get_tex_width() 
    {
        return hdri_enviromentMap->width;
    }

    int get_tex_height() 
    {
        return hdri_enviromentMap->height;
    }
};

