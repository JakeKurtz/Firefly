#include "dDirectionalLight.cuh" 
#include "dMaterial.cuh"
#include "dTriangle.cuh"

dDirectionalLight::dDirectionalLight(void)
{
	delta = true;
}

dDirectionalLight::dDirectionalLight(float3 dir)
{
	(this)->dir = normalize(dir);
	delta = true;
}

__device__ void dDirectionalLight::get_direction(const Isect& isect, float3& wi, float3& sample_point)
{
	wi = dir;
}

__device__ float3 dDirectionalLight::L(const Isect& isect)
{
	return (emissive_L(material));
}

__device__ float3 dDirectionalLight::L(const Isect& isect, float3 wi, float3 sample_point)
{
	return (emissive_L(material));
}

__device__ float dDirectionalLight::get_pdf(const Isect& isect) const
{
	return 1.f;
}

__device__ float dDirectionalLight::get_pdf(const Isect& isect, const float3& wi) const
{
	return 1.f;
}

__device__ float dDirectionalLight::get_pdf(const Isect& isect, const dRay& ray) const
{
	return 1.f;
}

__device__ bool dDirectionalLight::visible(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const
{
	float ts = K_HUGE;
	return (!intersect_shadows(nodes, triangles, ray, ts));
}

__device__ void dDirectionalLight::set_direction(const float x, const float y, const float z)
{
	dir = make_float3(x, y, z);
}

__device__ void dDirectionalLight::set_direction(const float3 dir)
{
	(this)->dir = dir;
}
