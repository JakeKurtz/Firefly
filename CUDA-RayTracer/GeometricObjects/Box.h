#ifndef _RAYTRACER_GEOMETRICOBJECTS_BOX_H_
#define _RAYTRACER_GEOMETRICOBJECTS_BOX_H_

#include "GeometricObj.h"
#include "../Samplers/Sampler.h"

class Box : public GeometricObj
{
public:

	__device__ Box(void)
	{

	};

	__device__ Box* clone(void) const
	{
		return (new Box(*this));
	};

	__device__ virtual bool hit(const Ray& ray, double& tmin, ShadeRec& sr) const
	{

	};

	__device__ virtual bool hit(const Ray& ray) const
	{

	};

	__device__ virtual bool shadow_hit(const Ray& ray, double& tmin) const
	{

	};

	__device__ virtual void set_sampler(Sampler* sp)
	{
		if (sampler_ptr) {
			delete sampler_ptr;
			sampler_ptr = nullptr;
		}

		sampler_ptr = sp;
		sampler_ptr->generate_samples();
	};

	__device__ virtual vec3 sample(void)
	{

	};

	__device__ virtual glm::dvec3 get_normal(const glm::dvec3 p)
	{
		return normal;
	};

private:
	glm::dvec3	c, u, v;
	glm::dvec3	normal;
	Sampler*	sampler_ptr;
};
#endif // _RAYTRACER_GEOMETRICOBJECTS_BOX_H_