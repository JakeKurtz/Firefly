#ifndef _RAYTRACER_CAMERA_RAY_RATRACECAMERA_H_
#define _RAYTRACER_CAMERA_RAY_RATRACECAMERA_H_

#include <glm/glm.hpp>
#include <cuda_runtime.h>

class Camera
{
public:
    glm::vec3 position;
    glm::vec3 lookat;
    glm::vec3 direction;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 world_up = glm::vec3(0, 1, 0);

    float exposure_time = 1.f;

    __host__ __device__ Camera(glm::vec3 position);
    __host__ __device__ Camera(glm::vec3 position, glm::vec3 lookat);
    __host__ __device__ void update_camera_vectors(void);
    __host__ __device__ virtual ~Camera() {};
};

__host__ __device__ Camera::Camera(glm::vec3 _position) {
    position = _position;
    update_camera_vectors();
}

__host__ __device__ Camera::Camera(glm::vec3 _position, glm::vec3 _lookat) {
    position = _position;
    lookat = _lookat;
    update_camera_vectors();
}

__host__ __device__ void Camera::update_camera_vectors(void) {
    direction = normalize(position - lookat);
    right = normalize(cross(direction, world_up));
    up = normalize(cross(right, direction));
}


#endif // _RAYTRACER_CAMERA_RAY_RATRACECAMERA_H_

