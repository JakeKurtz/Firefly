#pragma once

#include "dMatrix.cuh"

struct dTransform
{
	float3 centroid;
	Matrix4x4 matrix = Matrix4x4::unit();
	Matrix4x4 inv_matrix = Matrix4x4::unit();
};