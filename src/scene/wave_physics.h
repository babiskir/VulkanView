#pragma once
//
// WavePhysics — Analytical ocean wave simulation.
//
// Ported from EnoSea-Engine (hydrobehaviour/WavePhysics):
//   - WaveComponent: single regular Airy wave
//   - Spectrum: JONSWAP / Pierson-Moskowitz spectral generation
//   - SeaState: superposition of multiple wave systems
//
// Self-contained: no engine dependencies, only <vector> and <cstdint>.
//

#include <vector>
#include <cstdint>

// =============================================================================
// WaveComponent — a single regular wave (Airy wave theory)
// =============================================================================
class WaveComponent {
public:
    WaveComponent(float height, float period,
                  float phase = 0.0f, float directionDeg = 0.0f);

    /// Surface elevation at world position (x, z) and time t.
    float getElevation(float x, float z, float t) const;

    float getHeight()       const { return m_height; }
    float getPeriod()       const { return m_period; }
    float getAmplitude()    const { return m_amplitude; }
    float getWavelength()   const { return m_wavelength; }
    float getWaveNumber()   const { return m_waveNumber; }
    float getOmega()        const { return m_omega; }
    float getPhase()        const { return m_phase; }
    float getDirectionRad() const { return m_direction; }
    float getDirectionDeg() const;
    float getKx()           const { return m_kx; }
    float getKz()           const { return m_kz; }

private:
    float m_height;
    float m_period;
    float m_amplitude;
    float m_omega;
    float m_wavelength;
    float m_waveNumber;
    float m_phase;
    float m_direction;   // radians
    float m_kx;
    float m_kz;
};

// =============================================================================
// Spectrum generation functions
// =============================================================================
namespace Spectrum {

/// Pierson-Moskowitz spectral energy density S(ω).
float piersonMoskowitz(float omega, float Hs, float Tp);

/// JONSWAP spectral energy density S(ω).
float jonswap(float omega, float Hs, float Tp, float gamma = 3.3f);

/// Generate WaveComponents from a JONSWAP spectrum.
std::vector<WaveComponent> generateJONSWAP(
    float Hs, float Tp,
    int   numComponents = 50,
    float directionDeg  = 0.0f,
    float gamma         = 3.3f);

/// Generate WaveComponents from a Pierson-Moskowitz spectrum.
std::vector<WaveComponent> generatePiersonMoskowitz(
    float Hs, float Tp,
    int   numComponents = 50,
    float directionDeg  = 0.0f);

} // namespace Spectrum

// =============================================================================
// Sea-state configuration helpers
// =============================================================================
enum class SpectrumType { JONSWAP, PiersonMoskowitz };

struct JONSWAPConfig {
    float Hs            = 0.0f;
    float Tp            = 8.0f;
    float directionDeg  = 0.0f;
    float gamma         = 3.3f;
    int   numComponents = 50;
};

struct SwellConfig {
    float height       = 0.0f;
    float period       = 12.0f;
    float directionDeg = 0.0f;
    float phase        = 0.0f;
};

// =============================================================================
// SeaState — superposition of multiple wave systems
// =============================================================================
class SeaState {
public:
    /// Add a JONSWAP wind-sea system.
    void addJONSWAP(const JONSWAPConfig& config);

    /// Add a Pierson-Moskowitz wind-sea system.
    void addPiersonMoskowitz(float Hs, float Tp,
                             int numComponents = 50,
                             float directionDeg = 0.0f);

    /// Add a single regular swell component.
    void addSwell(const SwellConfig& config);

    /// Combined surface elevation at (x, z, t).
    float getElevation(float x, float z, float t) const;

    /// Signed distance from point (x,y,z) to the water surface.
    /// Positive = above water, negative = submerged.
    float distanceToWaterline(float x, float y, float z, float t) const;

    const std::vector<WaveComponent>& getComponents() const { return m_components; }
    int getComponentCount() const { return static_cast<int>(m_components.size()); }

    void clear() { m_components.clear(); }

private:
    std::vector<WaveComponent> m_components;
};
