#ifndef _RAYTRACER_GEOMETRICOBJECTS_TORUS_H_
#define _RAYTRACER_GEOMETRICOBJECTS_TORUS_H_

#include "GeometricObj.h"
#include "../Samplers/Sampler.h"

class Torus : public GeometricObj
{
public:

    __device__ Torus(void) :
        major(1.0), minor(0.5)
    {
        inv_area = 1.0 / (double)(2.0*M_PI*major)*(2.0*M_PI*minor);
    };

    __device__ Torus(const double _major, const double _minor) :
        major(_major), minor(_minor)
    {
        inv_area = 1.0 / (double)(2.0 * M_PI * major) * (2.0 * M_PI * minor);
    };

    __device__ Torus* clone(void) const
    {
        return (new Torus(*this));
    }

    __device__ virtual bool hit(const Ray& ray, double& tmin, ShadeRec& sr) const
    {
        double x1 = ray.o.x; double y1 = ray.o.y; double z1 = ray.o.z;
        double d1 = ray.d.x; double d2 = ray.d.y; double d3 = ray.d.z;

        double coeffs[5];	// coefficient array for the quartic equation
        double roots[4];	// solution array for the quartic equation

        // define the coefficients of the quartic equation

        double sum_d_sqrd = d1 * d1 + d2 * d2 + d3 * d3;
        double e = x1 * x1 + y1 * y1 + z1 * z1 - major * major - minor * minor;
        double f = x1 * d1 + y1 * d2 + z1 * d3;
        double four_a_sqrd = 4.0 * major * major;

        coeffs[0] = e * e - four_a_sqrd * (minor * minor - y1 * y1); 	// constant term
        coeffs[1] = 4.0 * f * e + 2.0 * four_a_sqrd * y1 * d2;
        coeffs[2] = 2.0 * sum_d_sqrd * e + 4.0 * f * f + four_a_sqrd * d2 * d2;
        coeffs[3] = 4.0 * sum_d_sqrd * f;
        coeffs[4] = sum_d_sqrd * sum_d_sqrd;  					// coefficient of t^4

        // find roots of the quartic equation

        int num_real_roots = solve_quartic(coeffs, roots);

        bool	intersected = false;
        double 	t = K_HUGE;

        if (num_real_roots == 0)  // ray misses the torus
            return(false);

        // find the smallest root greater than kEpsilon, if any
        // the roots array is not sorted

        for (int j = 0; j < num_real_roots; j++)
            if (roots[j] > K_EPSILON) {
                intersected = true;
                if (roots[j] < t)
                    t = roots[j];
            }

        if (!intersected)
            return (false);

        tmin = t;
        sr.local_hit_point = ray.o + t * ray.d;
        sr.normal = compute_normal(sr.local_hit_point);

        return (true);
    };

    __device__ virtual bool hit(const Ray& ray) const
    {
        double x1 = ray.o.x; double y1 = ray.o.y; double z1 = ray.o.z;
        double d1 = ray.d.x; double d2 = ray.d.y; double d3 = ray.d.z;

        double coeffs[5];	// coefficient array for the quartic equation
        double roots[4];	// solution array for the quartic equation

        // define the coefficients of the quartic equation

        double sum_d_sqrd = d1 * d1 + d2 * d2 + d3 * d3;
        double e = x1 * x1 + y1 * y1 + z1 * z1 - major * major - minor * minor;
        double f = x1 * d1 + y1 * d2 + z1 * d3;
        double four_a_sqrd = 4.0 * major * major;

        coeffs[0] = e * e - four_a_sqrd * (minor * minor - y1 * y1); 	// constant term
        coeffs[1] = 4.0 * f * e + 2.0 * four_a_sqrd * y1 * d2;
        coeffs[2] = 2.0 * sum_d_sqrd * e + 4.0 * f * f + four_a_sqrd * d2 * d2;
        coeffs[3] = 4.0 * sum_d_sqrd * f;
        coeffs[4] = sum_d_sqrd * sum_d_sqrd;  					// coefficient of t^4

        // find roots of the quartic equation

        int num_real_roots = solve_quartic(coeffs, roots);

        bool	intersected = false;
        double 	t = K_HUGE;

        if (num_real_roots == 0)  // ray misses the torus
            return(false);

        // find the smallest root greater than kEpsilon, if any
        // the roots array is not sorted

        for (int j = 0; j < num_real_roots; j++)
            if (roots[j] > K_EPSILON) {
                intersected = true;
                if (roots[j] < t)
                    t = roots[j];
            }

        if (!intersected)
            return (false);

        return (true);
    };

    __device__ virtual bool shadow_hit(const Ray& ray, double& tmin) const
    {
        double x1 = ray.o.x; double y1 = ray.o.y; double z1 = ray.o.z;
        double d1 = ray.d.x; double d2 = ray.d.y; double d3 = ray.d.z;

        double coeffs[5];	// coefficient array for the quartic equation
        double roots[4];	// solution array for the quartic equation

        // define the coefficients of the quartic equation

        double sum_d_sqrd = d1 * d1 + d2 * d2 + d3 * d3;
        double e = x1 * x1 + y1 * y1 + z1 * z1 - major * major - minor * minor;
        double f = x1 * d1 + y1 * d2 + z1 * d3;
        double four_a_sqrd = 4.0 * major * major;

        coeffs[0] = e * e - four_a_sqrd * (minor * minor - y1 * y1); 	// constant term
        coeffs[1] = 4.0 * f * e + 2.0 * four_a_sqrd * y1 * d2;
        coeffs[2] = 2.0 * sum_d_sqrd * e + 4.0 * f * f + four_a_sqrd * d2 * d2;
        coeffs[3] = 4.0 * sum_d_sqrd * f;
        coeffs[4] = sum_d_sqrd * sum_d_sqrd;  					// coefficient of t^4

        // find roots of the quartic equation

        int num_real_roots = solve_quartic(coeffs, roots);

        bool	intersected = false;
        double 	t = K_HUGE;

        if (num_real_roots == 0)  // ray misses the torus
            return(false);

        // find the smallest root greater than kEpsilon, if any
        // the roots array is not sorted

        for (int j = 0; j < num_real_roots; j++)
            if (roots[j] > K_EPSILON) {
                intersected = true;
                if (roots[j] < t)
                    t = roots[j];
            }

        if (!intersected)
            return (false);

        tmin = t;

        return (true);
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

    __device__ virtual glm::vec3 sample(void)
    {

    };

    __device__ virtual glm::dvec3 get_normal(const glm::dvec3 p)
    {
        return compute_normal(p);
    };

    __device__ virtual glm::dvec3 compute_normal(const glm::dvec3 p) const
    {
        glm::dvec3 normal;
        double param_squared = major * major + minor * minor;

        double x = p.x;
        double y = p.y;
        double z = p.z;
        double sum_squared = x * x + y * y + z * z;

        normal.x = 4.0 * x * (sum_squared - param_squared);
        normal.y = 4.0 * y * (sum_squared - param_squared + 2.0 * major * major);
        normal.z = 4.0 * z * (sum_squared - param_squared);
        normalize(normal);

        return (normal);
    };

private:
    double      major, minor;
    Sampler*    sampler_ptr;
};

#endif // _RAYTRACER_GEOMETRICOBJECTS_TORUS_H_