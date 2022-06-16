#include "dCamera.cuh"
#include "dMath.cuh"
#include "dMatrix.cuh"
#include "dRandom.cuh"

__device__ dCamera::dCamera()
{
}

__device__ dRay dCamera::gen_ray(dFilm* film, float2 p)
{
    float2 pNDC; // [-1, 1] x [-1, 1]
    pNDC.x = 2.f * ((p.x + 0.5) / (float)film->hres) - 1.f;
    pNDC.y = 1.f - 2.f * ((p.y + 0.5) / (float)film->vres);

    float4 pNearNDC = inv_view_proj_mat * make_float4(pNDC.x, pNDC.y, -1.f, 1.f);
    float4 pFarNDC = inv_view_proj_mat * make_float4(pNDC.x, pNDC.y, 1.f, 1.f);

    float3 pNear = make_float3(pNearNDC.x, pNearNDC.y, pNearNDC.z) / pNearNDC.w;
    float3 pFar = make_float3(pFarNDC.x, pFarNDC.y, pFarNDC.z) / pFarNDC.w;

    dRay ray;
    ray.o = pNear;
    ray.d = normalize(pFar - pNear);

    if (lens_radius > 0.f) {
        float3 pFocal = ray.o + ray.d * f;
        float2 pLensNorm = ConcentricSampleDisk() * lens_radius;

        float4 pLensTrans = inv_view_proj_mat * make_float4(pLensNorm.x, pLensNorm.y, 0.f, 1.f);
        float3 pLens = make_float3(pLensTrans.x, pLensTrans.y, pLensTrans.z) / pLensTrans.w;

        ray.o = pLens;
        ray.d = normalize(pFocal - ray.o);
    }

    return ray;
}