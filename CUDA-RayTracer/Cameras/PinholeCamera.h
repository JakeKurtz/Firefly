#ifndef _RAYTRACER_CAMERA_RAY_PINHOLECAMERA_H_
#define _RAYTRACER_CAMERA_RAY_PINHOLECAMERA_H_

#include "Camera.h"

class PinholeCamera : public Camera
{
public:
	Camera::Camera;

	__host__ __device__ float3 ray_direction(const float2& p) const {
		return normalize(p.x * right + p.y * up - d * direction);
	};

	__host__ __device__ void set_view_distance(float _d) {
		d = _d;
	};

	__host__ __device__ void set_zoom(float _zoom) {
		zoom = 1.f / _zoom;
	};

	__host__ __device__ float get_zoom(void) {
		return zoom;
	};

private:
	float d = 100.f;		// view-plane distance
	float zoom = 1.f;			// zoom factor
};

#endif // _RAYTRACER_CAMERA_RAY_PINHOLECAMERA_H_