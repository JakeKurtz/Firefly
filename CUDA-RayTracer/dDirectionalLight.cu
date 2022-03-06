#include "dDirectionalLight.cuh" 
#include "dMaterial.cuh"
#include "dTriangle.cuh"

dDirectionalLight::dDirectionalLight(void)
{
}

dDirectionalLight::dDirectionalLight(float3 dir)
{
	(this)->dir = normalize(dir);
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

__device__ float dDirectionalLight::G(const Isect& isect) const
{
	return 1.f;
}

__device__ bool dDirectionalLight::visible(const dRay& ray) const
{
	return false;
}

__device__ bool dDirectionalLight::visible(const dRay& ray, float& tmin, Isect& isect) const
{
	return false;
}

__device__ float dDirectionalLight::get_pdf(const Isect& isect) const
{
	return 5000.f;
}

__device__ float dDirectionalLight::get_pdf(const Isect& isect, const dRay& ray) const
{
	return 5000.f;
}

__device__ bool dDirectionalLight::in_shadow(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const
{
	float ts = K_HUGE;// = dot((object_ptr->sample() - ray.o), ray.d);
	return (intersect_shadows(nodes, triangles, ray, ts));
}

__device__ void dDirectionalLight::set_direction(const float x, const float y, const float z)
{
	dir = make_float3(x, y, z);
}

__device__ void dDirectionalLight::set_direction(const float3 dir)
{
	(this)->dir = dir;
}
