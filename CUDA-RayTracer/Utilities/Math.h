#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

__constant__ const int DEFAULT_SET_SIZE = 181;

__constant__ const float K_EPSILON = 1e-4;

__constant__ const float K_HUGE = 1e32;

// The base of natural logarithms (e)
__constant__ const float M_E = 2.71828182845904523536028747135266250;

// The logarithm to base 2 of M_E (log2(e))
__constant__ const float M_LOG2E = 1.44269504088896340735992468100189214;

// The logarithm to base 10 of M_E (log10(e))
__constant__ const float M_LOG10E = 0.434294481903251827651128918916605082;

// The natural logarithm of 2 (loge(2))
__constant__ const float M_LN2 = 0.693147180559945309417232121458176568;

// The natural logarithm of 10 (loge(10))
__constant__ const float M_LN10 = 2.30258509299404568401799145468436421;

// Pi, the ratio of a circle's circumference to its diameter.
__constant__ const float M_PI = 3.14159265358979323846264338327950288;

// Pi divided by two (pi/2)
__constant__ const float M_PI_2 = 1.57079632679489661923132169163975144;

// Pi divided by four  (pi/4)
__constant__ const float M_PI_4 = 0.785398163397448309615660845819875721;

// The reciprocal of pi (1/pi)
__constant__ const float M_1_PI = 0.318309886183790671537767526745028724;

// Two times the reciprocal of pi (2/pi)
__constant__ const float M_2_PI = 0.636619772367581343075535053490057448;

// Two times the reciprocal of the square root of pi (2/sqrt(pi))
__constant__ const float M_2_SQRTPI = 1.12837916709551257389615890312154517;

// The square root of two (sqrt(2))
__constant__ const float M_SQRT2 = 1.41421356237309504880168872420969808;

// The reciprocal of the square root of two (1/sqrt(2))
__constant__ const float M_SQRT1_2 = 0.707106781186547524400844362104849039;

static constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5;

#define EQN_EPS     1e-90

#define	IsZero(x)	((x) > -EQN_EPS && (x) < EQN_EPS)

inline float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

__device__ glm::vec3 fresnel(glm::vec3 f0, glm::vec3 h, glm::vec3 wo)
{
    return f0 + (glm::vec3(1.f) - f0) * pow(glm::max(1.f - glm::max(0.f, dot(h, wo)), 0.f), 5.f);
};

template <typename T> __host__ __device__ T lerp(const T a, const T b, const T w)
{
    return a + w * (b - a);
}

template <typename T> __host__ __device__ T remap(const T x, const T low1, const T high1, const T low2, const T high2)
{
    return low2 + (x - low1) * (high2 - low2) / (high1 - low1);
}

__host__ __device__ int solve_quadric(double c[3], double s[2])
{
    double p, q, D;

    /* normal form: x^2 + px + q = 0 */

    p = c[1] / (2 * c[2]);
    q = c[0] / c[2];

    D = p * p - q;

    if (IsZero(D)) {
        s[0] = -p;
        return 1;
    }
    else if (D > 0) {
        double sqrt_D = sqrt(D);

        s[0] = sqrt_D - p;
        s[1] = -sqrt_D - p;
        return 2;
    }
    else /* if (D < 0) */
        return 0;
}

__host__ __device__ int solve_cubic(double c[4], double s[3])
{
    int     i, num;
    double  sub;
    double  A, B, C;
    double  sq_A, p, q;
    double  cb_p, D;

    /* normal form: x^3 + Ax^2 + Bx + C = 0 */

    A = c[2] / c[3];
    B = c[1] / c[3];
    C = c[0] / c[3];

    /*  substitute x = y - A/3 to eliminate quadric term:
    x^3 +px + q = 0 */

    sq_A = A * A;
    p = 1.0 / 3 * (-1.0 / 3 * sq_A + B);
    q = 1.0 / 2 * (2.0 / 27 * A * sq_A - 1.0 / 3 * A * B + C);

    /* use Cardano's formula */

    cb_p = p * p * p;
    D = q * q + cb_p;

    if (IsZero(D)) {
        if (IsZero(q)) { /* one triple solution */
            s[0] = 0;
            num = 1;
        }
        else { /* one single and one double solution */
            double u = cbrt(-q);
            s[0] = 2 * u;
            s[1] = -u;
            num = 2;
        }
    }
    else if (D < 0) { /* Casus irreducibilis: three real solutions */
        double phi = 1.0 / 3 * acos(-q / sqrt(-cb_p));
        double t = 2 * sqrt(-p);

        s[0] = t * cos(phi);
        s[1] = -t * cos(phi + M_PI / 3);
        s[2] = -t * cos(phi - M_PI / 3);
        num = 3;
    }
    else { /* one real solution */
        double sqrt_D = sqrt(D);
        double u = cbrt(sqrt_D - q);
        double v = -cbrt(sqrt_D + q);

        s[0] = u + v;
        num = 1;
    }

    /* resubstitute */

    sub = 1.0 / 3 * A;

    for (i = 0; i < num; ++i)
        s[i] -= sub;

    return num;
}

__host__ __device__ int solve_quartic(double c[5], double s[4])
{
    double  coeffs[4];
    double  z, u, v, sub;
    double  A, B, C, D;
    double  sq_A, p, q, r;
    int     i, num;

    /* normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0 */

    A = c[3] / c[4];
    B = c[2] / c[4];
    C = c[1] / c[4];
    D = c[0] / c[4];

    /*  substitute x = y - A/4 to eliminate cubic term:
    x^4 + px^2 + qx + r = 0 */

    sq_A = A * A;
    p = -3.0 / 8 * sq_A + B;
    q = 1.0 / 8 * sq_A * A - 1.0 / 2 * A * B + C;
    r = -3.0 / 256 * sq_A * sq_A + 1.0 / 16 * sq_A * B - 1.0 / 4 * A * C + D;

    if (IsZero(r)) {
        /* no absolute term: y(y^3 + py + q) = 0 */

        coeffs[0] = q;
        coeffs[1] = p;
        coeffs[2] = 0;
        coeffs[3] = 1;

        num = solve_cubic(coeffs, s);

        s[num++] = 0;
    }
    else {
        /* solve the resolvent cubic ... */

        coeffs[0] = 1.0 / 2 * r * p - 1.0 / 8 * q * q;
        coeffs[1] = -r;
        coeffs[2] = -1.0 / 2 * p;
        coeffs[3] = 1;

        (void)solve_cubic(coeffs, s);

        /* ... and take the one real solution ... */

        z = s[0];

        /* ... to build two quadric equations */

        u = z * z - r;
        v = 2 * z - p;

        if (IsZero(u))
            u = 0;
        else if (u > 0)
            u = sqrt(u);
        else
            return 0;

        if (IsZero(v))
            v = 0;
        else if (v > 0)
            v = sqrt(v);
        else
            return 0;

        coeffs[0] = z - u;
        coeffs[1] = q < 0 ? -v : v;
        coeffs[2] = 1;

        num = solve_quadric(coeffs, s);

        coeffs[0] = z + u;
        coeffs[1] = q < 0 ? v : -v;
        coeffs[2] = 1;

        num += solve_quadric(coeffs, s + num);
    }

    /* resubstitute */

    sub = 1.0 / 4 * A;

    for (i = 0; i < num; ++i)
        s[i] -= sub;

    return num;
}

__device__ glm::vec3 get_orthogonal_vec(glm::vec3 in)
{
    glm::vec3 majorAxis;
    if (abs(in.x) < 0.57735026919f /* 1 / sqrt(3) */) {
        majorAxis = glm::vec3(1, 0, 0);
    }
    else if (abs(in.y) < 0.57735026919f /* 1 / sqrt(3) */) {
        majorAxis = glm::vec3(0, 1, 0);
    }
    else {
        majorAxis = glm::vec3(0, 0, 1);
    }
    return majorAxis;
}

__device__ glm::mat4 get_TBN(glm::vec3 normal)
{
    glm::vec3 N = normal;
    glm::vec3 T = get_orthogonal_vec(N);
    glm::vec3 B = normalize(cross(N, T));

    return glm::mat3(T, N, B);
}
