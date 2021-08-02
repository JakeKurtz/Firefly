#ifndef _RAYTRACER_MATERIALS_EMISSIVE_H_
#define _RAYTRACER_MATERIALS_EMISSIVE_H_

#include "Material.cuh"

class Emissive : public Material
{
public:
    __device__ void scale_radiance(const float _ls)
    {
        ls = _ls;
    };

    __device__ void set_ce(const float r, const float g, const float b)
    {
        ce = make_float3(r, g, b);
    };

    __device__ virtual float3 get_Le(ShadeRec& sr) const
    {
        return (ls * ce);
    };

     __device__ virtual float3 get_Le(Isect& isect) const
    {
        return (ls * ce);
    };

    __device__ virtual float3 shade(ShadeRec& sr)
    {
        if (dot(-sr.normal, sr.ray.d) > 0.0)
            return (ls * ce);
        else
            return (make_float3(0, 0, 0));
    };

    __device__ virtual bool is_emissive(void)
    {
        return (true);
    };

private:
    float ls;
    float3 ce;
};

#endif // _RAYTRACER_MATERIALS_EMISSIVE_H_