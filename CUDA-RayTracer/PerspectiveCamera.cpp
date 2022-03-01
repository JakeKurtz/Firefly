#include "PerspectiveCamera.h"

PerspectiveCamera::PerspectiveCamera()
{
    aspectRatio = 1.6;
    yfov = 90.f;
    znear = 1.f;
    zfar = 100.f;

    proj_mat = glm::perspective(yfov, aspectRatio, znear, zfar);
    updateCameraVectors();
}

PerspectiveCamera::PerspectiveCamera(float _aspectRatio, float _yfov, float _znear, float _zfar)
{
    aspectRatio = _aspectRatio;
    yfov = _yfov;
    znear = _znear;
    zfar = _zfar;

    proj_mat = glm::perspective(yfov, aspectRatio, znear, zfar);
    updateCameraVectors();
}

PerspectiveCamera::PerspectiveCamera(glm::vec3 _position, float _aspectRatio, float _yfov, float _znear, float _zfar)
{
    position = _position;
    aspectRatio = _aspectRatio;
    yfov = _yfov;
    znear = _znear;
    zfar = _zfar;

    proj_mat = glm::perspective(yfov, aspectRatio, znear, zfar);
    updateCameraVectors();
}