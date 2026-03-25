#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <cstdint>

/**
 * @brief Simple procedural mesh primitive generator.
 *
 * Returns meshes in the same vertex layout as MeshComponent (position, normal, texCoord, tangent),
 * but stored separately so they can be used before committing to a MeshComponent.
 */
struct PrimVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 tangent{1, 0, 0, 1};
};

struct PrimMesh {
    std::vector<PrimVertex> vertices;
    std::vector<uint32_t>   indices;
};

PrimMesh MakeBox(float halfExtent = 0.5f);
PrimMesh MakeSphere(float radius = 0.5f, int rings = 16, int sectors = 32);
PrimMesh MakeCapsule(float radius = 0.25f, float halfHeight = 0.5f, int rings = 8, int sectors = 16);
PrimMesh MakeGround(float halfSize = 20.0f);
PrimMesh MakeCylinder(float radius = 0.5f, float halfHeight = 0.5f, int sectors = 24);
PrimMesh MakeCone(float baseRadius = 0.5f, float height = 1.0f, int sectors = 24);
