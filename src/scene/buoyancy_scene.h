#pragma once

class Engine;

/// Load the EnoSea-inspired buoyancy demo scene.
/// Features: JONSWAP/Pierson-Moskowitz ocean spectra, analytical mesh-based
/// buoyancy forces (PhysX), floating Box/Sphere/Capsule objects, and a
/// real-time wave-parameter editor.
void LoadBuoyancyScene(Engine* engine);
