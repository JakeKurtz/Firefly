#ifndef _RAYTRACER_CAMERA_RAY_RATRACECAMERA_H_
#define _RAYTRACER_CAMERA_RAY_RATRACECAMERA_H_

#include <cuda_runtime.h>

class Camera
{
public:
    float3 position;
    float3 lookat;
    float3 direction;
    float3 up;
    float3 right;
    float3 world_up = make_float3(0, 1, 0);

    float exposure_time = 1.f;

    __host__ __device__ Camera(float3 position);
    __host__ __device__ Camera(float3 position, float3 lookat);
    __host__ __device__ void update_camera_vectors(void);
    __host__ __device__ virtual ~Camera() {};
};

__host__ __device__ Camera::Camera(float3 _position) {
    position = _position;
    update_camera_vectors();
}

__host__ __device__ Camera::Camera(float3 _position, float3 _lookat) {
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

