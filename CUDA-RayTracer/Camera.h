#pragma once
#include "GLCommon.h"

#include <string>
#include "Shader.h"

enum Camera_Movement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Camera
{
public:
    std::string name;
    int id;

    float zfar;
    float znear;

    // camera Attributes
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp = glm::vec3(0.f, 1.f, 0.f);

    glm::mat4 lookat_mat;
    glm::mat4 proj_mat;

    // euler Angles
    float yaw = 844.499390;
    float pitch = -7.30003309;

    // camera options
    float movementSpeed = 2.5f;
    float lookSensitivity = 0.1f;
    float zoom = 53.f;
    float exposure = 1.f;

    float lens_radius = 0.0001f;
    float focal_distance = 35.f;
    float d = 50.f;

    glm::mat4 getViewMatrix();
    glm::mat4 getProjMatrix();

    void sendUniforms(Shader& shader);
    void sendUniforms2(Shader& shader);

    void processKeyboard(Camera_Movement direction, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);
    void processMouseScroll(float yoffset);

protected:
    void updateCameraVectors();
};

