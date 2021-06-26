#ifndef _RAYTRACER_CAMERA_RAY_THINLENSCAMERA_H_
#define _RAYTRACER_CAMERA_RAY_THINLENSCAMERA_H_

#include "Camera.h"
#include "../Samplers/Sampler.h"

class Sampler;

class ThinLensCamera : public Camera
{
public:
	Camera::Camera;

	__device__ void set_sampler(Sampler* sp)
	{
		if (sampler_ptr) {
			delete sampler_ptr;
			sampler_ptr = nullptr;
		}

		sampler_ptr = sp;
		sampler_ptr->generate_samples();
		sampler_ptr->map_to_unit_disk();
	};

	__host__ __device__ Sampler* get_sampler(void) {
		return sampler_ptr;
	};

	__host__ __device__ glm::vec3 ray_direction(const glm::vec2& pixel_point, const glm::vec2& lens_point) const
	{
		glm::vec2 p;
		p.x = pixel_point.x * (float) f / (float) d;
		p.y = pixel_point.y * (float) f / (float) d;

		glm::vec3 dir = normalize((p.x - lens_point.x) * right + (p.y - lens_point.y) * up - f * direction);
		return dir;
	};

	__host__ __device__ void set_view_distance(float _d)
	{
		d = _d;
	};

	__host__ __device__ void set_focal_distance(float _f)
	{
		f = _f;
	};

	__host__ __device__ float get_focal_distance(void)
	{
		return f;
	};

	__host__ __device__ void set_lens_radius(float _lens_radius)
	{
		lens_radius = _lens_radius;
	};

	__host__ __device__ float get_lens_radius(void)
	{
		return lens_radius;
	};

	__host__ __device__ void set_zoom(float _zoom)
	{
		zoom = 1.f / _zoom;
	};

	__host__ __device__ float get_zoom(void)
	{
		return zoom;
	};

private:
	float lens_radius = 1.f;	// lens radius
	float d = 100.f;			// view plane distance
	float f = 1.f;				// focal plane distance
	float zoom = 1.f;			// zoom factor
	Sampler* sampler_ptr;	// sampler obj
};

#endif // _RAYTRACER_CAMERA_RAY_THINLENSCAMERA_H_
