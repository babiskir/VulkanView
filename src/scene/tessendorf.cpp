/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tessendorf.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

WSTessendorf::WSTessendorf(uint32_t tileSize, float tileLength)
    : m_TileLength(tileLength)
    , m_WindDir(glm::normalize(s_kDefaultWindDir))
    , m_WindSpeed(s_kDefaultWindSpeed)
    , m_A(s_kDefaultPhillipsConst)
    , m_Damping(s_kDefaultPhillipsDamping)
    , m_AnimationPeriod(s_kDefaultAnimPeriod)
{
    SetTileSize(tileSize);
    SetAnimationPeriod(s_kDefaultAnimPeriod);
}

WSTessendorf::~WSTessendorf()
{
    DestroyFFTW();
    fftwf_cleanup();
}

void WSTessendorf::SetTileSize(uint32_t size)
{
    const bool isPow2 = (size > 0) && ((size & (size - 1)) == 0);
    if (!isPow2) {
        std::cerr << "[Tessendorf] Tile size must be power of two, got " << size << std::endl;
        return;
    }
    m_TileSize = size;
}

void WSTessendorf::SetAnimationPeriod(float T)
{
    m_AnimationPeriod = T;
    m_BaseFreq = 2.0f * static_cast<float>(M_PI) / T;
}

void WSTessendorf::Prepare()
{
    std::cout << "[Tessendorf] Preparing " << m_TileSize << "x" << m_TileSize << " simulation" << std::endl;

    m_WaveVectors    = ComputeWaveVectors();
    auto gaussRandom = ComputeGaussRandomArray();
    m_BaseWaveHeights = ComputeBaseWaveHeightField(gaussRandom);

    const uint32_t N = m_TileSize;
    m_Displacements.assign(N * N, glm::vec4(0.0f));
    m_Normals.assign(N * N, glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

    DestroyFFTW();
    SetupFFTW();
}

std::vector<WSTessendorf::WaveVector> WSTessendorf::ComputeWaveVectors() const
{
    const int32_t N = static_cast<int32_t>(m_TileSize);
    const float   L = m_TileLength;

    std::vector<WaveVector> wv;
    wv.reserve(N * N);

    for (int32_t m = 0; m < N; ++m)
        for (int32_t n = 0; n < N; ++n)
            wv.emplace_back(glm::vec2(
                static_cast<float>(M_PI) * (2.0f * n - N) / L,
                static_cast<float>(M_PI) * (2.0f * m - N) / L
            ));

    return wv;
}

std::vector<WSTessendorf::Complex> WSTessendorf::ComputeGaussRandomArray() const
{
    const uint32_t N = m_TileSize;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<Complex> arr(N * N);
    for (auto& c : arr)
        c = Complex(dist(rng), dist(rng));

    return arr;
}

std::vector<WSTessendorf::BaseWaveHeight> WSTessendorf::ComputeBaseWaveHeightField(
    const std::vector<Complex>& gaussRandom
) const
{
    const uint32_t N = m_TileSize;
    std::vector<BaseWaveHeight> bwh(N * N);

    for (uint32_t m = 0; m < N; ++m)
    for (uint32_t n = 0; n < N; ++n)
    {
        const uint32_t idx = m * N + n;
        const auto& kv = m_WaveVectors[idx];
        const float k = glm::length(kv.vec);

        auto& h0 = bwh[idx];
        if (k > 0.00001f) {
            const auto gr = gaussRandom[idx];
            h0.heightAmp      = BaseWaveHeightFT(gr, kv.unit, k);
            h0.heightAmp_conj = std::conj(BaseWaveHeightFT(gr, -kv.unit, k));
            h0.dispersion     = QDispersion(k);
        } else {
            h0.heightAmp      = Complex(0);
            h0.heightAmp_conj = Complex(0);
            h0.dispersion     = 0.0f;
        }
    }

    return bwh;
}

void WSTessendorf::SetupFFTW()
{
    const uint32_t N  = m_TileSize;
    const uint32_t N2 = N * N;

    // Allocate all 7 arrays in one contiguous block
    m_Height = reinterpret_cast<Complex*>(fftwf_alloc_complex(7 * N2));
    m_SlopeX          = m_Height          + N2;
    m_SlopeZ          = m_SlopeX          + N2;
    m_DisplacementX   = m_SlopeZ          + N2;
    m_DisplacementZ   = m_DisplacementX   + N2;
    m_dxDisplacementX = m_DisplacementZ   + N2;
    m_dzDisplacementZ = m_dxDisplacementX + N2;

    auto mkPlan = [&](Complex* ptr) {
        return fftwf_plan_dft_2d(N, N,
            reinterpret_cast<fftwf_complex*>(ptr),
            reinterpret_cast<fftwf_complex*>(ptr),
            FFTW_BACKWARD, FFTW_MEASURE);
    };

    m_PlanHeight          = mkPlan(m_Height);
    m_PlanSlopeX          = mkPlan(m_SlopeX);
    m_PlanSlopeZ          = mkPlan(m_SlopeZ);
    m_PlanDisplacementX   = mkPlan(m_DisplacementX);
    m_PlanDisplacementZ   = mkPlan(m_DisplacementZ);
    m_PlandxDisplacementX = mkPlan(m_dxDisplacementX);
    m_PlandzDisplacementZ = mkPlan(m_dzDisplacementZ);
}

void WSTessendorf::DestroyFFTW()
{
    if (!m_PlanHeight) return;

    fftwf_destroy_plan(m_PlanHeight);         m_PlanHeight = nullptr;
    fftwf_destroy_plan(m_PlanSlopeX);
    fftwf_destroy_plan(m_PlanSlopeZ);
    fftwf_destroy_plan(m_PlanDisplacementX);
    fftwf_destroy_plan(m_PlanDisplacementZ);
    fftwf_destroy_plan(m_PlandxDisplacementX);
    fftwf_destroy_plan(m_PlandzDisplacementZ);

    fftwf_free(reinterpret_cast<fftwf_complex*>(m_Height));
    m_Height = nullptr;
}

float WSTessendorf::ComputeWaves(float t)
{
    const uint32_t N = m_TileSize;

    // Fill frequency-domain arrays
    for (uint32_t m = 0; m < N; ++m)
    for (uint32_t n = 0; n < N; ++n)
    {
        const uint32_t idx = m * N + n;
        const auto& kv = m_WaveVectors[idx];
        const Complex h = WaveHeightFT(m_BaseWaveHeights[idx], t);

        m_Height[idx]          = h;
        m_SlopeX[idx]          = Complex(0, kv.vec.x)  * h;
        m_SlopeZ[idx]          = Complex(0, kv.vec.y)  * h;
        m_DisplacementX[idx]   = Complex(0, -kv.unit.x) * h;
        m_DisplacementZ[idx]   = Complex(0, -kv.unit.y) * h;
        m_dxDisplacementX[idx] = Complex(0, kv.vec.x)  * m_DisplacementX[idx];
        m_dzDisplacementZ[idx] = Complex(0, kv.vec.y)  * m_DisplacementZ[idx];
    }

    // Execute all FFTs
    fftwf_execute(m_PlanHeight);
    fftwf_execute(m_PlanSlopeX);
    fftwf_execute(m_PlanSlopeZ);
    fftwf_execute(m_PlanDisplacementX);
    fftwf_execute(m_PlanDisplacementZ);
    fftwf_execute(m_PlandxDisplacementX);
    fftwf_execute(m_PlandzDisplacementZ);

    float maxH = std::numeric_limits<float>::lowest();
    float minH = std::numeric_limits<float>::max();

    const float kSigns[2] = { 1.0f, -1.0f };

    for (uint32_t m = 0; m < N; ++m)
    for (uint32_t n = 0; n < N; ++n)
    {
        const uint32_t idx  = m * N + n;
        const float    sign = kSigns[(n + m) & 1];

        const float hReal = m_Height[idx].real() * sign;
        maxH = std::max(hReal, maxH);
        minH = std::min(hReal, minH);

        m_Displacements[idx] = glm::vec4(
            sign * m_Lambda * m_DisplacementX[idx].real(),
            hReal,
            sign * m_Lambda * m_DisplacementZ[idx].real(),
            1.0f
        );

        m_Normals[idx] = glm::vec4(
            sign * m_SlopeX[idx].real(),
            sign * m_SlopeZ[idx].real(),
            sign * m_dxDisplacementX[idx].real(),
            sign * m_dzDisplacementZ[idx].real()
        );
    }

    return NormalizeHeights(minH, maxH);
}

float WSTessendorf::NormalizeHeights(float minH, float maxH)
{
    m_MinHeight = minH;
    m_MaxHeight = maxH;

    const float A = std::max(std::abs(minH), std::abs(maxH));
    if (A < 1e-7f) return A;

    const float invA = 1.0f / A;
    for (auto& d : m_Displacements)
        d.y *= invA;

    return A;
}
