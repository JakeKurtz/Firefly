#pragma once

#include <cuda_runtime.h>

#include "dRay.cuh"
#include "dFilm.cuh"
#include "dMatrix.cuh"

struct dCamera
{
    float3 position = make_float3(0, 0, 0);
    Matrix4x4 inv_view_proj_mat;
    Matrix4x4 inv_view_mat;

    float lens_radius = 1.f;	// lens radius
    float d = 100.f;			// view plane distance
    float f = 1.f;				// focal plane distance
    float zoom = 1.f;			// zoom factor
    float exposure_time = 1.f;

    __device__ dCamera();
    __device__ dRay gen_ray(dFilm* film, float2 p);
};
