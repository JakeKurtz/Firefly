#pragma once

#include <cuda_runtime.h>

struct dFilm {
	int			hres = 0;			// horizontal image resolution
	int			vres = 0;			// vertical image resolution
	float		gamma = 1.f;			// monitor gamma factor
	float		inv_gamma = 1.f;		// one over gamma
};