/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.
    This file is part of pbrt.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
#include <vector>
#include <cuda_runtime.h>

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_SPECTRUM_H
#define PBRT_CORE_SPECTRUM_H

 // core/spectrum.h*
//#include "pbrt.h"
//#include "stringprint.h"

    // Spectrum Utility Declarations
    static const int sampledLambdaStart = 400;
    static const int sampledLambdaEnd = 700;
    static const int nSpectralSamples = 60;
    extern __host__ __device__ bool SpectrumSamplesSorted(const float* lambda, const float* vals, int n);
    extern __host__ __device__ void SortSpectrumSamples(float* lambda, float* vals, int n);
    extern __host__ __device__ float AverageSpectrumSamples(const float* lambda, const float* vals, int n, float lambdaStart, float lambdaEnd);

    inline __host__ __device__ void XYZToRGB(const float xyz[3], float rgb[3]) {
        rgb[0] = 3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
        rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
        rgb[2] = 0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
    }

    inline __host__ __device__ void RGBToXYZ(const float rgb[3], float xyz[3]) {
        xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
        xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
        xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
    }

    enum class SpectrumType { Reflectance, Illuminant };
    extern __host__ __device__ float InterpolateSpectrumSamples(const float* lambda, const float* vals, int n, float l);
    extern __host__ __device__ void Blackbody(const float* lambda, int n, float T, float* Le);
    extern __host__ __device__ void BlackbodyNormalized(const float* lambda, int n, float T, float* vals);

    // Spectral Data Declarations
    static const int nCIESamples = 471;
    extern const float CIE_X[nCIESamples];
    extern const float CIE_Y[nCIESamples];
    extern const float CIE_Z[nCIESamples];
    extern const float CIE_lambda[nCIESamples];
    static const float CIE_Y_integral = 106.856895;
    static const int nRGB2SpectSamples = 32;
    extern const float RGB2SpectLambda[nRGB2SpectSamples];
    extern const float RGBRefl2SpectWhite[nRGB2SpectSamples];
    extern const float RGBRefl2SpectCyan[nRGB2SpectSamples];
    extern const float RGBRefl2SpectMagenta[nRGB2SpectSamples];
    extern const float RGBRefl2SpectYellow[nRGB2SpectSamples];
    extern const float RGBRefl2SpectRed[nRGB2SpectSamples];
    extern const float RGBRefl2SpectGreen[nRGB2SpectSamples];
    extern const float RGBRefl2SpectBlue[nRGB2SpectSamples];
    extern const float RGBIllum2SpectWhite[nRGB2SpectSamples];
    extern const float RGBIllum2SpectCyan[nRGB2SpectSamples];
    extern const float RGBIllum2SpectMagenta[nRGB2SpectSamples];
    extern const float RGBIllum2SpectYellow[nRGB2SpectSamples];
    extern const float RGBIllum2SpectRed[nRGB2SpectSamples];
    extern const float RGBIllum2SpectGreen[nRGB2SpectSamples];
    extern const float RGBIllum2SpectBlue[nRGB2SpectSamples];

    // Spectrum Declarations
    template <int nSpectrumSamples>
    class CoefficientSpectrum {
    public:
        // CoefficientSpectrum Public Methods
        __host__ __device__ CoefficientSpectrum(float v = 0.f) {
            for (int i = 0; i < nSpectrumSamples; ++i) c[i] = v;
            //DCHECK(!HasNaNs());
        }

        __host__ __device__ CoefficientSpectrum& operator+=(const CoefficientSpectrum& s2) {
            //DCHECK(!s2.HasNaNs());
            for (int i = 0; i < nSpectrumSamples; ++i) c[i] += s2.c[i];
            return *this;
        }
        __host__ __device__ CoefficientSpectrum operator+(const CoefficientSpectrum& s2) const {
            //DCHECK(!s2.HasNaNs());
            CoefficientSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] += s2.c[i];
            return ret;
        }
        __host__ __device__ CoefficientSpectrum operator-(const CoefficientSpectrum& s2) const {
            //DCHECK(!s2.HasNaNs());
            CoefficientSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] -= s2.c[i];
            return ret;
        }
        __host__ __device__ CoefficientSpectrum operator/(const CoefficientSpectrum& s2) const {
            //DCHECK(!s2.HasNaNs());
            CoefficientSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; ++i) {
                //CHECK_NE(s2.c[i], 0);
                ret.c[i] /= s2.c[i];
            }
            return ret;
        }
        __host__ __device__ CoefficientSpectrum operator*(const CoefficientSpectrum& sp) const {
            //DCHECK(!sp.HasNaNs());
            CoefficientSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] *= sp.c[i];
            return ret;
        }
        __host__ __device__ CoefficientSpectrum& operator*=(const CoefficientSpectrum& sp) {
            //DCHECK(!sp.HasNaNs());
            for (int i = 0; i < nSpectrumSamples; ++i) c[i] *= sp.c[i];
            return *this;
        }
        __host__ __device__ CoefficientSpectrum operator*(float a) const {
            CoefficientSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] *= a;
            //DCHECK(!ret.HasNaNs());
            return ret;
        }
        __host__ __device__ CoefficientSpectrum& operator*=(float a) {
            for (int i = 0; i < nSpectrumSamples; ++i) c[i] *= a;
            //DCHECK(!HasNaNs());
            return *this;
        }
        friend inline __host__ __device__  CoefficientSpectrum operator*(float a,
            const __host__ __device__ CoefficientSpectrum& s) {
            //DCHECK(!std::isnan(a) && !s.HasNaNs());
            return s * a;
        }
        __host__ __device__ CoefficientSpectrum operator/(float a) const {
            //CHECK_NE(a, 0);
            //DCHECK(!std::isnan(a));
            CoefficientSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] /= a;
            //DCHECK(!ret.HasNaNs());
            return ret;
        }
        __host__ __device__ CoefficientSpectrum& operator/=(float a) {
            //CHECK_NE(a, 0);
            //DCHECK(!std::isnan(a));
            for (int i = 0; i < nSpectrumSamples; ++i) c[i] /= a;
            return *this;
        }
        __host__ __device__ bool operator==(const CoefficientSpectrum& sp) const {
            for (int i = 0; i < nSpectrumSamples; ++i)
                if (c[i] != sp.c[i]) return false;
            return true;
        }
        __host__ __device__ bool operator!=(const CoefficientSpectrum& sp) const {
            return !(*this == sp);
        }
        __host__ __device__ bool IsBlack() const {
            for (int i = 0; i < nSpectrumSamples; ++i)
                if (c[i] != 0.) return false;
            return true;
        }
        friend __host__ __device__ CoefficientSpectrum Sqrt(const CoefficientSpectrum& s) {
            CoefficientSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] = std::sqrt(s.c[i]);
            //DCHECK(!ret.HasNaNs());
            return ret;
        }
        template <int n>
        friend inline __host__ __device__ CoefficientSpectrum<n> Pow(const CoefficientSpectrum<n>& s,
            float e);
        __host__ __device__ CoefficientSpectrum operator-() const {
            CoefficientSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] = -c[i];
            return ret;
        }
        friend __host__ __device__ CoefficientSpectrum Exp(const CoefficientSpectrum& s) {
            CoefficientSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] = std::exp(s.c[i]);
            //DCHECK(!ret.HasNaNs());
            return ret;
        }
        __host__ __device__ CoefficientSpectrum Clamp(float low = 0, float high = Infinity) const {
            CoefficientSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; ++i)
                ret.c[i] = pbrt::Clamp(c[i], low, high);
            //DCHECK(!ret.HasNaNs());
            return ret;
        }
        __host__ __device__ float MaxComponentValue() const {
            float m = c[0];
            for (int i = 1; i < nSpectrumSamples; ++i)
                m = std::max(m, c[i]);
            return m;
        }
        __host__ __device__ bool HasNaNs() const {
            for (int i = 0; i < nSpectrumSamples; ++i)
                if (std::isnan(c[i])) return true;
            return false;
        }
        __host__ __device__ float& operator[](int i) {
            //DCHECK(i >= 0 && i < nSpectrumSamples);
            return c[i];
        }
        __host__ __device__ float operator[](int i) const {
            //DCHECK(i >= 0 && i < nSpectrumSamples);
            return c[i];
        }

        // CoefficientSpectrum Public Data
        static const int nSamples = nSpectrumSamples;

    protected:
        // CoefficientSpectrum Protected Data
        float c[nSpectrumSamples];
    };

    class SampledSpectrum : public CoefficientSpectrum<nSpectralSamples> {
    public:
        // SampledSpectrum Public Methods
        __host__ __device__ SampledSpectrum(float v = 0.f) : CoefficientSpectrum(v) {}
        __host__ __device__ SampledSpectrum(const CoefficientSpectrum<nSpectralSamples>& v)
            : CoefficientSpectrum<nSpectralSamples>(v) {}
        __host__ __device__ static SampledSpectrum FromSampled(const float* lambda, const float* v, int n)
        {
            // Sort samples if unordered, use sorted for returned spectrum
            if (!SpectrumSamplesSorted(lambda, v, n)) {
                std::vector<float> slambda(&lambda[0], &lambda[n]);
                std::vector<float> sv(&v[0], &v[n]);
                SortSpectrumSamples(&slambda[0], &sv[0], n);
                return FromSampled(&slambda[0], &sv[0], n);
            }
            SampledSpectrum r;
            for (int i = 0; i < nSpectralSamples; ++i) {
                // Compute average value of given SPD over $i$th sample's range
                float lambda0 = lerp(float(i) / float(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);
                float lambda1 = lerp(float(i + 1) / float(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);
                r.c[i] = AverageSpectrumSamples(lambda, v, n, lambda0, lambda1);
            }
            return r;
        }
        __host__ __device__ static void Init() {
            // Compute XYZ matching functions for _SampledSpectrum_
            for (int i = 0; i < nSpectralSamples; ++i) {
                float wl0 = lerp(float(i) / float(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);
                float wl1 = lerp(float(i + 1) / float(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);
                X.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_X, nCIESamples, wl0, wl1);
                Y.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_Y, nCIESamples, wl0, wl1);
                Z.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_Z, nCIESamples, wl0, wl1);
            }

            // Compute RGB to spectrum functions for _SampledSpectrum_
            for (int i = 0; i < nSpectralSamples; ++i) {
                float wl0 = lerp(float(i) / float(nSpectralSamples),sampledLambdaStart, sampledLambdaEnd);
                float wl1 = lerp(float(i + 1) / float(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);

                rgbRefl2SpectWhite.c[i] = 
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectWhite, 
                        nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectCyan.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectCyan,
                        nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectMagenta.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectMagenta,
                        nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectYellow.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectYellow,
                        nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectRed.c[i] = 
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectRed,
                        nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectGreen.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectGreen,
                        nRGB2SpectSamples, wl0, wl1);
                rgbRefl2SpectBlue.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBRefl2SpectBlue,
                        nRGB2SpectSamples, wl0, wl1);

                rgbIllum2SpectWhite.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectWhite,
                        nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectCyan.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectCyan,
                        nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectMagenta.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectMagenta,
                        nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectYellow.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectYellow,
                        nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectRed.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectRed,
                        nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectGreen.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectGreen,
                        nRGB2SpectSamples, wl0, wl1);
                rgbIllum2SpectBlue.c[i] =
                    AverageSpectrumSamples(RGB2SpectLambda, RGBIllum2SpectBlue,
                        nRGB2SpectSamples, wl0, wl1);
            }
        }
        __host__ __device__ void ToXYZ(float xyz[3]) const {
            xyz[0] = xyz[1] = xyz[2] = 0.f;
            for (int i = 0; i < nSpectralSamples; ++i) {
                xyz[0] += X.c[i] * c[i];
                xyz[1] += Y.c[i] * c[i];
                xyz[2] += Z.c[i] * c[i];
            }
            float scale = float(sampledLambdaEnd - sampledLambdaStart) /
                float(CIE_Y_integral * nSpectralSamples);
            xyz[0] *= scale;
            xyz[1] *= scale;
            xyz[2] *= scale;
        }
        __host__ __device__ float y() const {
            float yy = 0.f;
            for (int i = 0; i < nSpectralSamples; ++i) yy += Y.c[i] * c[i];
            return yy * float(sampledLambdaEnd - sampledLambdaStart) /
                float(CIE_Y_integral * nSpectralSamples);
        }
        __host__ __device__ void ToRGB(float rgb[3]) const {
            float xyz[3];
            ToXYZ(xyz);
            XYZToRGB(xyz, rgb);
        }
        __host__ __device__ RGBSpectrum ToRGBSpectrum() const;
        __host__ __device__ static SampledSpectrum FromRGB(const float rgb[3], SpectrumType type = SpectrumType::Illuminant);
        __host__ __device__ static SampledSpectrum FromXYZ(const float xyz[3], SpectrumType type = SpectrumType::Reflectance)
        {
            float rgb[3];
            XYZToRGB(xyz, rgb);
            return FromRGB(rgb, type);
        }
        __host__ __device__ SampledSpectrum(const RGBSpectrum& r, SpectrumType type = SpectrumType::Reflectance);

    private:
        // SampledSpectrum Private Data
        static SampledSpectrum X, Y, Z;
        static SampledSpectrum rgbRefl2SpectWhite, rgbRefl2SpectCyan;
        static SampledSpectrum rgbRefl2SpectMagenta, rgbRefl2SpectYellow;
        static SampledSpectrum rgbRefl2SpectRed, rgbRefl2SpectGreen;
        static SampledSpectrum rgbRefl2SpectBlue;
        static SampledSpectrum rgbIllum2SpectWhite, rgbIllum2SpectCyan;
        static SampledSpectrum rgbIllum2SpectMagenta, rgbIllum2SpectYellow;
        static SampledSpectrum rgbIllum2SpectRed, rgbIllum2SpectGreen;
        static SampledSpectrum rgbIllum2SpectBlue;
    };

    class RGBSpectrum : public CoefficientSpectrum<3> {
        using CoefficientSpectrum<3>::c;

    public:
        // RGBSpectrum Public Methods
        __host__ __device__ RGBSpectrum(float v = 0.f) : CoefficientSpectrum<3>(v) {}
        __host__ __device__ RGBSpectrum(const CoefficientSpectrum<3>& v) : CoefficientSpectrum<3>(v) {}
        __host__ __device__ RGBSpectrum(const RGBSpectrum& s, SpectrumType type = SpectrumType::Reflectance)
        {
            *this = s;
        }
        __host__ __device__ static RGBSpectrum FromRGB(const float rgb[3], SpectrumType type = SpectrumType::Reflectance)
        {
            RGBSpectrum s;
            s.c[0] = rgb[0];
            s.c[1] = rgb[1];
            s.c[2] = rgb[2];
            //DCHECK(!s.HasNaNs());
            return s;
        }
        __host__ __device__ void ToRGB(float* rgb) const 
        {
            rgb[0] = c[0];
            rgb[1] = c[1];
            rgb[2] = c[2];
        }
        __host__ __device__ const RGBSpectrum& ToRGBSpectrum() const { return *this; }
        __host__ __device__ void ToXYZ(float xyz[3]) const { RGBToXYZ(c, xyz); }
        __host__ __device__ static RGBSpectrum FromXYZ(const float xyz[3], SpectrumType type = SpectrumType::Reflectance) {
            RGBSpectrum r;
            XYZToRGB(xyz, r.c);
            return r;
        }
        __host__ __device__ float y() const {
            const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
            return YWeight[0] * c[0] + YWeight[1] * c[1] + YWeight[2] * c[2];
        }
        __host__ __device__ static RGBSpectrum FromSampled(const float* lambda, const float* v, int n) {
            // Sort samples if unordered, use sorted for returned spectrum
            if (!SpectrumSamplesSorted(lambda, v, n)) {
                std::vector<float> slambda(&lambda[0], &lambda[n]);
                std::vector<float> sv(&v[0], &v[n]);
                SortSpectrumSamples(&slambda[0], &sv[0], n);
                return FromSampled(&slambda[0], &sv[0], n);
            }
            float xyz[3] = { 0, 0, 0 };
            for (int i = 0; i < nCIESamples; ++i) {
                float val = InterpolateSpectrumSamples(lambda, v, n, CIE_lambda[i]);
                xyz[0] += val * CIE_X[i];
                xyz[1] += val * CIE_Y[i];
                xyz[2] += val * CIE_Z[i];
            }
            float scale = float(CIE_lambda[nCIESamples - 1] - CIE_lambda[0]) /
                float(CIE_Y_integral * nCIESamples);
            xyz[0] *= scale;
            xyz[1] *= scale;
            xyz[2] *= scale;
            return FromXYZ(xyz);
        }
    };

    // Spectrum Inline Functions
    template <int nSpectrumSamples>
    inline __host__ __device__ CoefficientSpectrum<nSpectrumSamples> Pow(const CoefficientSpectrum<nSpectrumSamples>& s, float e)
    {
        CoefficientSpectrum<nSpectrumSamples> ret;
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] = std::pow(s.c[i], e);
        //DCHECK(!ret.HasNaNs());
        return ret;
    }

    inline __host__ __device__ RGBSpectrum lerp(float t, const RGBSpectrum& s1, const RGBSpectrum& s2)
    {
        return (1 - t) * s1 + t * s2;
    }

    inline __host__ __device__ SampledSpectrum lerp(float t, const SampledSpectrum& s1, const SampledSpectrum& s2) 
    {
        return (1 - t) * s1 + t * s2;
    }

    inline __host__ __device__ float lerp(float t, const int& s1, const int& s2)
    {
        return (1 - t) * s1 + t * s2;
    }

    __host__ __device__ void ResampleLinearSpectrum(const float* lambdaIn, const float* vIn, int nIn, float lambdaMin, float lambdaMax, int nOut, float* vOut);

    template <typename Predicate> __host__ __device__ int FindInterval(int size,
        const Predicate& pred) {
        int first = 0, len = size;
        while (len > 0) {
            int half = len >> 1, middle = first + half;
            // Bisect range based on value of pred at middle
            if (pred(middle)) {
                first = middle + 1;
                len -= half + 1;
            }
            else
                len = half;

        }
        return glm::clamp(first - 1, 0, size - 2);
    }

#endif  // PBRT_CORE_SPECTRUM_H
*/