#include "dAreaLight.cuh"
#include "GeometricObj.cuh"
#include "Isect.cuh"
#include "dRay.cuh"

__device__ dAreaLight::dAreaLight(void) :
	dLight(),
	object_ptr(nullptr)//,
	//material_ptr(nullptr)
{
	ls = 1.f;
	position = make_float3(0, 0, 0);
	color = make_float3(1, 1, 1);
	enable_shadows(true);
}

__device__ void dAreaLight::get_direction(const Isect& isect, float3& wi, float3& sample_point)
{
	sample_point = object_ptr->sample();
	//float3 light_normal = object_ptr->get_normal(sample_point);
	wi = normalize(sample_point - isect.position);
};

__device__ float3 dAreaLight::L(const Isect& isect, float3 wi, float3 sample_point)
{
	float3 light_normal = object_ptr->get_normal(sample_point);
	float n_dot_d = dot(-light_normal, wi);

	//if (n_dot_d > 0.0)
	//	return (emissive_L(material_ptr));
	//else
	//	return (make_float3(0, 0, 0));
};

__device__ float dAreaLight::G(const Isect& isect) const
{
	float3 sample_point = object_ptr->sample();
	float3 light_normal = object_ptr->get_normal(sample_point);
	float3 wi = normalize(sample_point - isect.position);

	float n_dot_d = dot(-light_normal, wi);
	float d2 = pow(distance(sample_point, isect.position), 2);

	return (n_dot_d / d2);
};

__device__ bool dAreaLight::visible(const dRay& ray) const
{
	return object_ptr->hit(ray);
};

__device__ bool dAreaLight::visible(const dRay& ray, float& tmin, Isect& isect) const
{
	return object_ptr->hit(ray, tmin, isect);
}

__device__ float dAreaLight::get_pdf(const Isect& isect) const
{
	float3 sample_point = object_ptr->sample();
	float3 light_normal = object_ptr->get_normal(sample_point);
	float3 wi = normalize(sample_point - isect.position);

	float n_dot_d = abs(dot(light_normal, -wi));
	float d2 = pow(distance(sample_point, isect.position), 2);

	return ((d2 / n_dot_d) * object_ptr->pdf(isect));
};

__device__ float dAreaLight::get_pdf(const Isect& isect, const dRay& ray) const
{
	float3 wi = normalize(ray.o - isect.position);

	dRay visibility_ray(isect.position, wi);
	if (!object_ptr->hit(visibility_ray)) return 0.f;

	float n_dot_d = abs(dot(isect.normal, -wi));
	float d2 = pow(distance(ray.o, isect.position), 2);

	return ((d2 / n_dot_d) * object_ptr->pdf(isect));
};

__device__ bool dAreaLight::in_shadow(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const
{
	float ts = dot((object_ptr->sample() - ray.o), ray.d);
	return false;//(intersect_shadows(ray, ts));
};

__device__ void dAreaLight::set_position(const float x, const float y, const float z)
{
	position = make_float3(x, y, z);
};

__device__ void dAreaLight::set_position(const float3 pos)
{
	position = pos;
}

__device__ void dAreaLight::set_object(GeometricObj* obj_ptr)
{
	object_ptr = obj_ptr;
};
/*
__device__ void dAreaLight::set_material(cudaMaterial* mat_ptr)
{
	material_ptr = mat_ptr;
};
*/