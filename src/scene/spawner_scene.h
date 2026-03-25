#pragma once

class Engine;

/// Load the EnoSea-style object-spawning demo scene.
/// Features: 6 primitive shapes (Box, Sphere, Capsule, Cylinder, Cone, Cone-as-Ellipsoid)
/// placed over a ground plane with PhysX dynamics and JONSWAP ocean buoyancy.
/// Press SPACE (or use the ImGui panel) to spawn additional random objects.
void LoadSpawnerScene(Engine* engine);
