#pragma once
#include "Sampler.h"

class MultiJittered : public Sampler {
public:
	using Sampler::Sampler;

	__device__ virtual void generate_samples(void) {
		
		samples = CudaList<glm::vec2>(num_samples);
		
		// num_samples needs to be a perfect square
		int n = (int)sqrt((float)num_samples);
		float subcell_width = 1.0 / ((float)num_samples);

		// fill the samples array with dummy points to allow us to use the [ ] notation when we set the 
		// initial patterns

		glm::vec2 fill_point;
		for (int j = 0; j < num_samples; j++)
			samples.add(glm::vec2(fill_point));

	// distribute points in the initial patterns
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				samples[i * n + j].x = (i * n + j) * subcell_width + rand_float(0, subcell_width);
				samples[i * n + j].y = (j * n + i) * subcell_width + rand_float(0, subcell_width);
			}
		}
		
	// shuffle x coordinates
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				int k = rand_int(j, n - 1);
				float t = samples[i * n + j].x;
				samples[i * n + j].x = samples[i * n + k].x;
				samples[i * n + k].x = t;
			}
		}

	// shuffle y coordinates
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				int k = rand_int(j, n - 1);
				float t = samples[j * n + i].y;
				samples[j * n + i].y = samples[k * n + i].y;
				samples[k * n + i].y = t;
			}
		}
	};
};