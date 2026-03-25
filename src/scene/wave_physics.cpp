#include "wave_physics.h"

#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// WaveComponent
// =============================================================================

WaveComponent::WaveComponent(float height, float period,
                             float phase, float directionDeg) {
    constexpr float g = 9.81f;

    m_height    = height;
    m_period    = period;
    m_phase     = phase;
    m_direction = directionDeg * static_cast<float>(M_PI) / 180.0f;
    m_amplitude = m_height * 0.5f;
    m_omega     = 2.0f * static_cast<float>(M_PI) / m_period;
    m_wavelength = g * m_period * m_period / (2.0f * static_cast<float>(M_PI));
    m_waveNumber = 2.0f * static_cast<float>(M_PI) / m_wavelength;
    m_kx = m_waveNumber * std::cos(m_direction);
    m_kz = m_waveNumber * std::sin(m_direction);
}

float WaveComponent::getDirectionDeg() const {
    return m_direction * 180.0f / static_cast<float>(M_PI);
}

float WaveComponent::getElevation(float x, float z, float t) const {
    return m_amplitude * std::cos(m_kx * x + m_kz * z - m_omega * t + m_phase);
}

// =============================================================================
// Spectrum
// =============================================================================

float Spectrum::piersonMoskowitz(float omega, float Hs, float Tp) {
    if (omega <= 0.0f) return 0.0f;
    float omegaP = 2.0f * static_cast<float>(M_PI) / Tp;
    return (5.0f / 16.0f) * Hs * Hs
         * std::pow(omegaP, 4.0f)
         * std::pow(omega,  -5.0f)
         * std::exp(-1.25f * std::pow(omega / omegaP, -4.0f));
}

float Spectrum::jonswap(float omega, float Hs, float Tp, float gamma) {
    if (omega <= 0.0f) return 0.0f;
    float spm    = piersonMoskowitz(omega, Hs, Tp);
    float omegaP = 2.0f * static_cast<float>(M_PI) / Tp;
    float sigma  = (omega <= omegaP) ? 0.07f : 0.09f;
    float r      = std::exp(-0.5f * std::pow((omega - omegaP) / (sigma * omegaP), 2.0f));
    float aGamma = 1.0f - 0.287f * std::log(gamma);
    return aGamma * spm * std::pow(gamma, r);
}

std::vector<WaveComponent> Spectrum::generateJONSWAP(
        float Hs, float Tp, int numComponents,
        float directionDeg, float gamma) {
    std::vector<WaveComponent> waves;
    waves.reserve(numComponents);

    float omegaP     = 2.0f * static_cast<float>(M_PI) / Tp;
    float omegaMin   = 0.3f * omegaP;
    float omegaMax   = 3.0f * omegaP;
    float deltaOmega = (omegaMax - omegaMin) / numComponents;

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> phaseDist(0.0f, 2.0f * static_cast<float>(M_PI));

    for (int i = 0; i < numComponents; ++i) {
        float omega   = omegaMin + (i + 0.5f) * deltaOmega;
        float S       = jonswap(omega, Hs, Tp, gamma);
        float height  = 2.0f * std::sqrt(2.0f * S * deltaOmega);
        float period  = 2.0f * static_cast<float>(M_PI) / omega;
        waves.emplace_back(height, period, phaseDist(gen), directionDeg);
    }
    return waves;
}

std::vector<WaveComponent> Spectrum::generatePiersonMoskowitz(
        float Hs, float Tp, int numComponents, float directionDeg) {
    std::vector<WaveComponent> waves;
    waves.reserve(numComponents);

    float omegaP     = 2.0f * static_cast<float>(M_PI) / Tp;
    float omegaMin   = 0.3f * omegaP;
    float omegaMax   = 3.0f * omegaP;
    float deltaOmega = (omegaMax - omegaMin) / numComponents;

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> phaseDist(0.0f, 2.0f * static_cast<float>(M_PI));

    for (int i = 0; i < numComponents; ++i) {
        float omega   = omegaMin + (i + 0.5f) * deltaOmega;
        float S       = piersonMoskowitz(omega, Hs, Tp);
        float height  = 2.0f * std::sqrt(2.0f * S * deltaOmega);
        float period  = 2.0f * static_cast<float>(M_PI) / omega;
        waves.emplace_back(height, period, phaseDist(gen), directionDeg);
    }
    return waves;
}

// =============================================================================
// SeaState
// =============================================================================

void SeaState::addJONSWAP(const JONSWAPConfig& cfg) {
    auto w = Spectrum::generateJONSWAP(cfg.Hs, cfg.Tp, cfg.numComponents,
                                       cfg.directionDeg, cfg.gamma);
    m_components.insert(m_components.end(), w.begin(), w.end());
}

void SeaState::addPiersonMoskowitz(float Hs, float Tp,
                                   int numComponents, float directionDeg) {
    auto w = Spectrum::generatePiersonMoskowitz(Hs, Tp, numComponents, directionDeg);
    m_components.insert(m_components.end(), w.begin(), w.end());
}

void SeaState::addSwell(const SwellConfig& cfg) {
    m_components.emplace_back(cfg.height, cfg.period, cfg.phase, cfg.directionDeg);
}

float SeaState::getElevation(float x, float z, float t) const {
    float eta = 0.0f;
    for (const auto& w : m_components)
        eta += w.getElevation(x, z, t);
    return eta;
}

float SeaState::distanceToWaterline(float x, float y, float z, float t) const {
    return y - getElevation(x, z, t);
}
