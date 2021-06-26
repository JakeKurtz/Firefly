#pragma once

#include "../Utilities/Ray.h"
#include "../Utilities/Math.h"

class BBox
{
public:
    double x0, x1;
    double y0, y1;
    double z0, z1;

    __device__ BBox() :
        x0(0), x1(0),
        y0(0), y1(0),
        z0(0), z1(0)
    {};

    __device__ BBox(double const _x0, double const _x1, double const _y0, double const _y1, double const _z0, double const _z1) :
        x0(_x0), x1(_x1),
        y0(_y0), y1(_y1),
        z0(_z0), z1(_z1)
    {};

    __device__ bool hit(const Ray& ray) const
    {
        double ox = ray.o.x; double oy = ray.o.y; double oz = ray.o.z;
        double dx = ray.d.x; double dy = ray.d.y; double dz = ray.d.z;

        double tx_min, ty_min, tz_min;
        double tx_max, ty_max, tz_max;

        double a = 1.0 / dx;
        if (a >= 0) {
            tx_min = (x0 - ox) * a;
            tx_max = (x1 - ox) * a;
        }
        else
        {
            tx_min = (x1 - ox) * a;
            tx_max = (x0 - ox) * a;
        }

        double b = 1.0 / dy;
        if (b >= 0) {
            ty_min = (y0 - oy) * b;
            ty_max = (y1 - oy) * b;
        }
        else
        {
            ty_min = (y1 - oy) * b;
            ty_max = (y0 - oy) * b;
        }

        double c = 1.0 / dy;
        if (c >= 0) {
            tz_min = (z0 - oz) * c;
            tz_max = (z1 - oz) * c;
        }
        else
        {
            tz_min = (z1 - oz) * c;
            tz_max = (z0 - oz) * c;
        }

        double t0, t1;

        if (tx_min > ty_min)
            t0 = tx_min;
        else
            t0 = ty_min;

        if (tz_min > t0)
            t0 = tz_min;

        if (tx_max > ty_max)
            t1 = tx_max;
        else
            t1 = ty_max;

        if (tz_max < t1)
            t1 = tz_max;

        return (t0 < t1 && t1 > K_EPSILON);
    };

    __device__ bool inside(const vec3& p) const
    {
        return ((p.x > x0 && p.x < x1) && (p.y > y0 && p.y < y1) && (p.z > z0 && p.z < z1));
    };
};
