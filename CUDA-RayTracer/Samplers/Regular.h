#pragma once
#include "Sampler.h"

class Regular : public Sampler {
public:
	using Sampler::Sampler;
private:
	__device__ virtual void generate_samples(void) {

		samples = CudaList<glm::vec2>(num_samples);

		for (int p = 0; p < num_samples; p++) {
			glm::vec2 sp(0.5, 0.5);
			samples.add(sp);
		}
	};
};