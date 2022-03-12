#include "dLight.cuh"

__device__ void dLight::get_direction(const Isect& isect, float3& wi, float3& sample_point)
{
	return void();
}

__device__ float3 dLight::L(const Isect& isect, float3 wi, float3 sample_point) { return make_float3(1, 1, 1); };

__device__ float3 dLight::L(const Isect& isect) { return make_float3(1, 1, 1); };

__device__ float dLight::G(const Isect& isect) const
{
	return 1.0f;
};

__device__ float dLight::get_pdf(const Isect& isect, const dRay& ray) const
{
	return 1.0f;
};

__device__ float dLight::get_pdf(const Isect& isect) const
{
	return 1.0f;
};

__device__ void dLight::set_color(const float x, const float y, const float z)
{
	color = make_float3(x, y, z);
};

__device__ void dLight::set_color(const float s)
{
	color = make_float3(s, s, s);
};

__device__ void dLight::set_color(const float3 col)
{
	color = col;
};

__device__ void dLight::scale_radiance(const float _ls)
{
	ls = _ls;
};

__device__ bool dLight::casts_shadows(void)
{
	return shadows;
};

__device__ void dLight::enable_shadows(bool b)
{
	shadows = b;
}
__device__ bool dLight::is_delta()
{
	return delta;
}
;