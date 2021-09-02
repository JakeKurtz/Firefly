#pragma once

#include <cuda_runtime.h>
#include "cutil_math.h"

class Camera
{
public:
    float3 position;
    float3 lookat;
    float3 direction;
    float3 up;
    float3 right;
    float3 world_up = make_float3(0, 1, 0);
    float lens_radius = 1.f;	// lens radius
    float d = 100.f;			// view plane distance
    float f = 1.f;				// focal plane distance
    float zoom = 1.f;			// zoom factor
    float exposure_time = 1.f;

    float yaw = -90;
    float pitch = 0.f;

    __device__ Camera(float3 position);
    __device__ Camera(float3 position, float3 lookat);
    __device__ void update_camera_vectors(void) {

        float3 front;
        front.x = cos(yaw) * cos(pitch);
        front.y = sin(pitch);
        front.z = sin(yaw) * cos(pitch);
        direction = normalize(front);

        //direction = normalize(position - lookat);
        right = normalize(cross(direction, world_up));
        up = normalize(cross(right, direction));
        lookat = position + direction;
    };

    __device__ void process_mouse_movement(float xoffset, float yoffset, bool constrainPitch = true)
    {
        float MouseSensitivity = 0.01;

        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

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
        update_camera_vectors();
    }

    __device__ float3 ray_direction(const float2& pixel_point, const float2& lens_point) {
        float2 p;
        p.x = pixel_point.x * (float)f / (float)d;
        p.y = pixel_point.y * (float)f / (float)d;

        float3 dir = normalize((p.x - lens_point.x) * right + (p.y - lens_point.y) * up - f * direction);
        return dir;
    };
    __device__ virtual ~Camera() {};
};
