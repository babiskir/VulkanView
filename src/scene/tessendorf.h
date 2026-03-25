/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <complex>
#include <vector>
#include <random>
#include <memory>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <fftw3.h>

/**
 * @brief Generates data used for rendering the water surface.
 *  Ported from WaterSurfaceRendering demo (MIT License).
 *  Based on: Jerry Tessendorf. Simulating Ocean Water. 1999.
 */
class WSTessendorf
{
public:
    static constexpr uint32_t     s_kDefaultTileSize  { 256 };
    static constexpr float        s_kDefaultTileLength{ 1000.0f };

    static inline const glm::vec2 s_kDefaultWindDir{ 1.0f, 1.0f };
    static constexpr float        s_kDefaultWindSpeed{ 30.0f };
    static constexpr float        s_kDefaultAnimPeriod{ 200.0f };
    static constexpr float        s_kDefaultPhillipsConst{ 3e-7f };
    static constexpr float        s_kDefaultPhillipsDamping{ 0.1f };

    using Displacement = glm::vec4; // vec4(dx, dy/height, dz, jacobian)
    using Normal       = glm::vec4; // vec4(slopeX, slopeZ, dDxdx, dDzdz)

public:
    WSTessendorf(uint32_t tileSize = WSTessendorf::s_kDefaultTileSize,
                 float tileLength  = WSTessendorf::s_kDefaultTileLength);
    ~WSTessendorf();

    void Prepare();
    float ComputeWaves(float time);

    auto GetTileSize() const { return m_TileSize; }
    auto GetTileLength() const { return m_TileLength; }
    auto GetWindDir() const { return m_WindDir; }
    auto GetWindSpeed() const { return m_WindSpeed; }
    float GetMinHeight() const { return m_MinHeight; }
    float GetMaxHeight() const { return m_MaxHeight; }
    float GetDisplacementLambda() const { return m_Lambda; }

    const std::vector<Displacement>& GetDisplacements() const { return m_Displacements; }
    const std::vector<Normal>&       GetNormals() const { return m_Normals; }

    void SetTileSize(uint32_t size);
    void SetTileLength(float length) { m_TileLength = length; }
    void SetWindDirection(const glm::vec2& w) { m_WindDir = glm::normalize(w); }
    void SetWindSpeed(float v) { m_WindSpeed = glm::max(0.0001f, v); }
    void SetAnimationPeriod(float T);
    void SetPhillipsConst(float A) { m_A = A; }
    void SetLambda(float lambda) { m_Lambda = lambda; }
    void SetDamping(float damping) { m_Damping = damping; }

private:
    using Complex = std::complex<float>;

    struct WaveVector {
        glm::vec2 vec;
        glm::vec2 unit;
        WaveVector(glm::vec2 v)
            : vec(v), unit(glm::length(v) > 0.00001f ? glm::normalize(v) : glm::vec2(0)) {}
    };

    struct BaseWaveHeight {
        Complex heightAmp;
        Complex heightAmp_conj;
        float   dispersion;
    };

    std::vector<WaveVector>    ComputeWaveVectors() const;
    std::vector<Complex>       ComputeGaussRandomArray() const;
    std::vector<BaseWaveHeight> ComputeBaseWaveHeightField(const std::vector<Complex>& gaussRandom) const;
    float                      NormalizeHeights(float minH, float maxH);
    void                       SetupFFTW();
    void                       DestroyFFTW();

    Complex BaseWaveHeightFT(Complex gaussRandom, glm::vec2 unitWaveVec, float k) const
    {
        static constexpr float kOneOver2sqrt = 1.0f / 1.41421356f;
        return kOneOver2sqrt * gaussRandom * std::sqrt(PhillipsSpectrum(unitWaveVec, k));
    }

    float PhillipsSpectrum(glm::vec2 unitWaveVec, float k) const
    {
        const float k2 = k * k, k4 = k2 * k2;
        float cosFact = glm::dot(unitWaveVec, m_WindDir);
        cosFact *= cosFact;
        const float L = m_WindSpeed * m_WindSpeed / 9.81f;
        return m_A * std::exp(-1.0f / (k2 * L * L)) / k4 * cosFact
               * std::exp(-k2 * m_Damping * m_Damping);
    }

    static Complex WaveHeightFT(const BaseWaveHeight& wh, float t)
    {
        const float omega_t = wh.dispersion * t;
        const float pc = std::cos(omega_t), ps = std::sin(omega_t);
        return wh.heightAmp * Complex(pc, ps) + wh.heightAmp_conj * Complex(pc, -ps);
    }

    float QDispersion(float k) const
    {
        const float w = std::sqrt(9.81f * k);
        return std::floor(w / m_BaseFreq) * m_BaseFreq;
    }

private:
    uint32_t  m_TileSize;
    float     m_TileLength;
    glm::vec2 m_WindDir;
    float     m_WindSpeed;
    float     m_A;
    float     m_Damping;
    float     m_AnimationPeriod;
    float     m_BaseFreq{ 1.0f };
    float     m_Lambda{ -1.0f };

    std::vector<Displacement>    m_Displacements;
    std::vector<Normal>          m_Normals;
    std::vector<WaveVector>      m_WaveVectors;
    std::vector<BaseWaveHeight>  m_BaseWaveHeights;

    Complex* m_Height        { nullptr };
    Complex* m_SlopeX        { nullptr };
    Complex* m_SlopeZ        { nullptr };
    Complex* m_DisplacementX { nullptr };
    Complex* m_DisplacementZ { nullptr };
    Complex* m_dxDisplacementX{ nullptr };
    Complex* m_dzDisplacementZ{ nullptr };

    fftwf_plan m_PlanHeight          { nullptr };
    fftwf_plan m_PlanSlopeX          { nullptr };
    fftwf_plan m_PlanSlopeZ          { nullptr };
    fftwf_plan m_PlanDisplacementX   { nullptr };
    fftwf_plan m_PlanDisplacementZ   { nullptr };
    fftwf_plan m_PlandxDisplacementX { nullptr };
    fftwf_plan m_PlandzDisplacementZ { nullptr };

    float m_MinHeight{ -1.0f };
    float m_MaxHeight{  1.0f };
};
