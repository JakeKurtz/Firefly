#pragma once
#include <cuda_runtime.h>

struct ViewPlane {
	int			hres;			// horizontal image resolution
	int			vres;			// vertical image resolution
	float		s;				// pixel size;
	float		gamma;			// monitor gamma factor
	float		inv_gamma;		// one over gamma
	int			num_samples;	// one over gamma
	int			max_depth;
};