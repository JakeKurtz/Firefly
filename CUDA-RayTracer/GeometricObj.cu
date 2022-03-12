#include "GeometricObj.cuh"

__device__ float3 GeometricObj::sample(void)
{
	return make_float3(0, 0, 0);
};

__device__ float3 GeometricObj::get_normal(const float3 p)
{
	return make_float3(0, 0, 0);
};

__device__ void GeometricObj::set_color(float r, float g, float b)
{
	color = make_float3(r, g, b);
};

__device__ float3 GeometricObj::get_color()
{
	return color;
};

__device__ dMaterial* GeometricObj::get_material(void)
{
	return material;
};

__device__ void GeometricObj::set_material(dMaterial* _material)
{
	material = _material;
};

__device__ void GeometricObj::enable_shadows(bool b)
{
	shadows = b;
};

__device__ float GeometricObj::pdf(const Isect& isect)
{
	return inv_area;
};