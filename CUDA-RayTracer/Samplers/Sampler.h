#pragma once

#include <glm/glm.hpp>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "../Utilities/CudaList.h"
#include "../Utilities/Random.h"
#include "../Utilities/Math.h"

using namespace glm;
using namespace std;

using namespace glm;

class Sampler
{
public:

	unsigned long		count;					// the current number of sample points used
	int					jump;					// random index jump

	__device__ Sampler(int n) {
		num_samples = n;
		jump = 0;
		count = 0;
	};

	__device__ virtual void generate_samples(void) = 0;

	__device__ void shuffle_samples(void) 
	{
		// nothing 
	};

	__device__ glm::vec2 sample_unit_square(void) 
	{
		return (samples[rand_int() % num_samples]);
	};

	__device__ glm::vec2 sample_unit_disk(void) 
	{
		return (disk_samples[rand_int() % num_samples]);
	};

	__device__ glm::vec3 sample_hemisphere(void)
	{
		return (hemisphere_samples[rand_int() % num_samples]);
	};

	__device__ glm::vec3 sample_sphere(void) 
	{
		return (sphere_samples[rand_int() % num_samples]);
	};

	__device__ int get_num_samples(void)
	{
		return num_samples;
	}

	__device__ void map_to_unit_disk(void)
	{
		int size = samples.size();
		float r, phi;
		glm::vec2 sp;

		disk_samples = CudaList<glm::vec2>(size);

		for (int i = 0; i < size; i++) {

			// map sample point to [-1, 1] [-1, 1]
			sp.x = 2.f * (samples[i]).x - 1.f;
			sp.y = 2.f * (samples[i]).y - 1.f;

			if (sp.x > -sp.y) {
				if (sp.x > sp.y) {
					r = sp.x;
					phi = sp.y / sp.x;
				}
				else {
					r = sp.y;
					phi = 2 - sp.x / sp.y;
				}
			}
			else {
				if (sp.x < sp.y) {
					r = -sp.x;
					phi = 4 + sp.y / sp.x;
				}
				else {
					r = -sp.y;
					if (sp.y != 0.f)
						phi = 6 - sp.x / sp.y;
					else
						phi = 0.f;
				}
			}

			phi *= M_PI / 4.f;

			disk_samples.add(glm::vec2(r * cos(phi), r * sin(phi)));
		}
	};

	__device__ void map_to_hemisphere(const float e)
	{
		int size = samples.size();
		hemisphere_samples = CudaList<glm::vec3>(num_samples);;

		for (int i = 0; i < size; i++) {
			float cos_phi = cos(2.f * M_PI * (samples[i]).x);
			float sin_phi = sin(2.f * M_PI * (samples[i]).x);

			float cos_theta = pow((1.f - (samples[i]).y), 1.f / (e + 1.f));
			float sin_theta = sqrt(1.f - cos_theta * cos_theta);

			float pu = sin_theta * cos_phi;
			float pv = sin_theta * sin_phi;
			float pw = cos_theta;

			hemisphere_samples.add(glm::vec3(pu, pv, pw));
		}
	};

	__device__ void map_to_sphere(void) 
	{
		float r1, r2;
		float x, y, z;
		float r, phi;

		sphere_samples = CudaList<glm::vec3>(num_samples);

		for (int j = 0; j < num_samples; j++) {
			r1 = (samples[j]).x;
			r2 = (samples[j]).y;
			z = 1.0 - 2.0 * r1;
			r = sqrt(1.0 - z * z);
			phi = 2 * M_PI * r2;
			x = r * cos(phi);
			y = r * sin(phi);

			sphere_samples.add(glm::vec3(x, y, z));
		}
	};

	__device__ virtual ~Sampler() = default;

protected:
	int					num_samples;			// number of sample points in a pattern
	CudaList<glm::vec2>		samples;				// sample points on a unit square
	CudaList<glm::vec2>		disk_samples;			// sample points on a unit disk
	CudaList<glm::vec3>		hemisphere_samples;		// sample points on a hemisphere
	CudaList<glm::vec3>		sphere_samples;			// sample points on a hemisphere
	CudaList<int>		shuffled_indices;		// shuffled samples array indices

};

