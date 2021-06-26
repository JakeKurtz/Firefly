#pragma once
#include <cuda_runtime.h>

#include "../Samplers/MultiJittered.h"
#include "../Samplers/Regular.h"

class Sampler;

class ViewPlane
{
public:
	int			hres;			// horizontal image resolution
	int			vres;			// vertical image resolution
	float		s;				// pixel size;
	float		gamma;			// monitor gamma factor
	float		inv_gamma;		// one over gamma
	int			num_samples;	// one over gamma
	int			max_depth;
	Sampler*	sampler_ptr = nullptr; 


	__device__ void set_hres(int _hres) {
		hres = _hres;
	};
	__device__ void set_vres(int _vres) {
		vres = _vres;
	};
	__device__ void set_pixel_size(float _s) {
		s = _s;
	};
	__device__ void set_gamma(float _gamma) {
		gamma = _gamma;
		inv_gamma = 1.f / _gamma;
	};
	__device__ void set_samples(const int n);
	//__device__ void set_sampler(Sampler* sp);
	__device__ void set_max_depth(const int _max_depth) { max_depth = _max_depth; };

	//__device__ ~ViewPlane() = default;
};

__device__ void ViewPlane::set_samples(const int n) {
	
	num_samples = n;

	if (sampler_ptr) {
		delete sampler_ptr;
		sampler_ptr = nullptr;
	}

	if (num_samples > 1) {
		sampler_ptr = new MultiJittered(num_samples);
	}
	else {
		sampler_ptr = new Regular(1);
	}
	sampler_ptr->generate_samples();
};

/*__device__ void ViewPlane::set_sampler(Sampler* sp) {
	
	if (sampler_ptr) {
		delete sampler_ptr;
		sampler_ptr = nullptr;
	}

	num_samples = sp->get_num_samples();
	sampler_ptr = sp;

	sp->generate_samples();
	
}*/