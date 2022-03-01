#include "Camera.h"

glm::mat4 Camera::getViewMatrix()
{
    return lookat_mat;
}

glm::mat4 Camera::getProjMatrix()
{
    return proj_mat;
}

void Camera::sendUniforms(Shader& shader) {
    shader.setMat4("projection", proj_mat);
    shader.setMat4("view", lookat_mat);
    shader.setVec3("camPos", position);
    shader.setVec3("camDir", front);
    shader.setVec3("camUp", up);
    shader.setVec3("camRight", right);
    shader.setFloat("cam_exposure", exposure);
}
void Camera::sendUniforms2(Shader& shader) {
    shader.setMat4("projection", proj_mat);
    shader.setMat4("view", glm::mat4(glm::mat3(lookat_mat)));
    shader.setVec3("camPos", position);
    shader.setVec3("camDir", front);
    shader.setVec3("camUp", up);
    shader.setVec3("camRight", right);
    shader.setFloat("cam_exposure", exposure);
}

void Camera::processKeyboard(Camera_Movement direction, float deltaTime)
{
    float velocity = movementSpeed * deltaTime;
    if (direction == FORWARD)
        position += front * velocity;
    if (direction == BACKWARD)
        position -= front * velocity;
    if (direction == LEFT)
        position -= right * velocity;
    if (direction == RIGHT)
        position += right * velocity;
    if (direction == UP)
        position += up * velocity;
    if (direction == DOWN)
        position -= up * velocity;
    updateCameraVectors();
}
void Camera::processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
{
    xoffset *= lookSensitivity;
    yoffset *= lookSensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch)
    {
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
    }

    // update Front, Right and Up Vectors using the updated Euler angles
    updateCameraVectors();
}
void Camera::processMouseScroll(float yoffset)
{
    zoom -= (float)yoffset;
    if (zoom < 1.0f)
        zoom = 1.0f;
    if (zoom > 120.0f)
        zoom = 120.0f;
    updateCameraVectors();
}

void Camera::updateCameraVectors()
{
    // calculate the new Front vector
    glm::vec3 _front;
    _front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    _front.y = sin(glm::radians(pitch));
    _front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(_front);

    // also re-calculate the Right and Up vector
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));

    lookat_mat = glm::lookAt(position, position + front, up);
}
