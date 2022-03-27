#pragma once

#include "dLight.cuh"
#include <string>
#include "dTexture.cuh"

class dEnvironmentLight : public dLight
{
public:
	__device__ dEnvironmentLight(void);
	__device__ dEnvironmentLight(float3 color);
	__device__ dEnvironmentLight(cudaTextureObject_t hrd_texture, cudaSurfaceObject_t pdf_texture, double* marginal_y, double** conds_y, int tex_width, int tex_height);
	__device__ virtual void get_direction(const Isect& isect, float3& wi, float3& sample_point);
	__device__ virtual bool visible(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& ray) const;
	__device__ virtual float3 L(const Isect& isect, float3 wi, float3 sample_point);
	__device__ virtual float get_pdf(const Isect& isect) const;
	__device__ virtual float get_pdf(const Isect& isect, const float3& wi) const;
	__device__ cudaTextureObject_t get_hrd_tex();
	__device__ cudaSurfaceObject_t get_pdf_surf();
	__device__ int get_tex_width();
	__device__ int get_tex_height();
	__device__ double get_marginal_y(int y);
	__device__ double get_conds_y(int x, int y);

private:
	cudaTextureObject_t hrd_texture = -1;
	cudaSurfaceObject_t pdf_texture = -1;

	double* marginal_y;
	double* marginal_p;
	double** conds_y;

	float radius;
	int tex_width = 1;
	int tex_height = 1;
	double pdf_denom;
};
