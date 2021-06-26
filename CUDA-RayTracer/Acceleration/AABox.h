#pragma once

#include "../Utilities/Ray.h"
#include "../Utilities/Math.h"
#include "../Utilities/ShadeRec.h"

#include <glm/glm.hpp>

class BBox
{
public:
    __device__ bool hit(const Ray& ray, double& tmin, ShadeRec& sr) const {
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
        int face_in, face_out;

        if (tx_min > ty_min) {
            t0 = tx_min;
            face_in = (a >= 0.0) ? 0 : 3;
        }
        else {
            t0 = ty_min;
            face_in = (b >= 0.0) ? 1 : 4;
        }
        if (tz_min > t0) {
            t0 = tz_min;
            face_in = (c >= 0.0) ? 2 : 5;
        }
        if (tx_max > ty_max) {
            t1 = tx_max;
            face_out = (a >= 0.0) ? 3 : 0;
        }
        else {
            t1 = ty_max;
            face_out = (b >= 0.0) ? 4 : 1;
        }
        if (tz_max < t1){
            t1 = tz_max;
            face_out = (c >= 0.0) ? 5 : 2;
        }

        if (t0 < t1 && t1 > K_EPSILON) {   // condition for a hit
            if (t0 > K_EPSILON) {
                tmin = t0;                  // ray hits outside surface
                sr.normal = get_normal(face_in);
            }
            else {
                tmin = t1;
                sr.normal = get_normal(face_out);
            }
            sr.local_hit_point = ray.o + tmin * ray.d;
            return (true);
        }
        else {
            return (false);
        }
    };
private:
    float x0, x1;
    float y0, y1;
    float z0, z1;

    __device__ glm::dvec3 get_normal(const int face_hit) const {
        switch (face_hit)
        {
            case 0:     return (glm::dvec3(-1, 0, 0));
            case 1:     return (glm::dvec3(0, -1, 0));
            case 2:     return (glm::dvec3(0, 0, -1));
            case 3:     return (glm::dvec3(1, 0, 0));
            case 4:     return (glm::dvec3(0, 1, 0));
            case 5:     return (glm::dvec3(0, 0, 1));
        }
    }
};
