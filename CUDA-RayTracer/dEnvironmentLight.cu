#include "dEnvironmentLight.cuh"
#include "dMath.cuh"
#include "dRandom.cuh"
#include "dTriangle.cuh"
#include "Helpers.cuh"

__device__ dEnvironmentLight::dEnvironmentLight(void)
{
	color = make_float3(1.f, 1.f, 1.f);
	hrd_texture = -1;
	delta = false;

	// NOTE: setting the environment light radius too large will result in precision errors.
	radius = 10000;
}

__device__ dEnvironmentLight::dEnvironmentLight(float3 color)
{
	(this)->color = color;
	(this)->ls = ls;
	hrd_texture = -1;
	delta = false;

	// NOTE: setting the environment light radius too large will result in precision errors.
	radius = 10000;
}

dEnvironmentLight::dEnvironmentLight(cudaTextureObject_t hrd_texture, cudaSurfaceObject_t pdf_texture, double* marginal_y, double** conds_y, int tex_width, int tex_height)
{
	(this)->hrd_texture = hrd_texture;
	(this)->tex_width = tex_width;
	(this)->tex_height = tex_height;
	(this)->pdf_texture = pdf_texture;
	(this)->conds_y = conds_y;
	(this)->marginal_y = marginal_y;

	delta = false;
	radius = 10000;
}

__device__ void dEnvironmentLight::get_direction(const Isect& isect, float3& wi, float3& sample_point)
{
	float ex = random();
	float ey = random();

	int y = upper_bound(marginal_y, tex_height, ey) - 1.f;
	int x = upper_bound(conds_y[y], tex_width, ex) - 1.f;

	float u = (float)x / (float)tex_width;
	float v = (float)y / (float)tex_height;

	float3 dir = sample_spherical_direction(make_float2(u, v));

	sample_point = make_float3(0.f) + dir * radius;
	wi = normalize(sample_point - isect.position);

	//wi = sample_spherical_direction(make_float2(u, v));
}

__device__ bool dEnvironmentLight::visible(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const
{
	float ts = K_HUGE;
	return (!intersect_shadows(nodes, triangles, ray, ts));
}

__device__ float3 dEnvironmentLight::L(const Isect& isect, float3 wi, float3 sample_point)
{
	if (hrd_texture == -1) {
		return color;
	}
	else {
		float2 uv = sample_spherical_map(wi);
		float4 s = tex2DLod<float4>(hrd_texture, uv.x, uv.y, 0);
		return make_float3(s.x, s.y, s.z);
	}
}

__device__ float dEnvironmentLight::get_pdf(const Isect& isect) const
{
	return M_1_4PI;
}

__device__ float dEnvironmentLight::get_pdf(const Isect& isect, const float3& wi) const
{
	//float3 sample_point = make_float3(0.f) + wi * radius;
	//float3 dir = normalize(sample_point - isect.position);

	float2 uv = sample_spherical_map(wi);
	/*
	float4 sample = tex2DLod<float4>(hrd_texture, uv.x, uv.y, 0);

	float lum = luminance(make_float3(sample.x, sample.y, sample.z));

	float pdf = (lum * sin(M_PI * uv.y)) / pdf_denom;
	float foo = (tex_width * tex_height) / (2.f * M_PI * M_PI * sin(acos(clamp(wi.z, -1.f, 1.f))));

	return pdf * foo;
	*/
	float pdf;
	surf2Dread(&pdf, pdf_texture, (int)(uv.x * (tex_width - 1)) * sizeof(float), (int)(uv.y * (tex_height - 1)));

	return pdf / sin(acos(clamp(wi.z, -1.f, 1.f)));
}

__device__ cudaTextureObject_t dEnvironmentLight::get_hrd_tex()
{
	return hrd_texture;
}

__device__ cudaSurfaceObject_t dEnvironmentLight::get_pdf_surf()
{
	return pdf_texture;
}

__device__ int dEnvironmentLight::get_tex_width()
{
	return tex_width;
}

__device__ int dEnvironmentLight::get_tex_height()
{
	return tex_height;
}

__device__ double dEnvironmentLight::get_marginal_y(int y)
{
	return marginal_y[y];
}

__device__ double dEnvironmentLight::get_conds_y(int x, int y)
{
	return conds_y[y][x];
}
