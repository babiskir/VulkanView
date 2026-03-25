#include "primitive_meshes.h"
#include <glm/gtc/constants.hpp>
#include <cmath>

// ---------------------------------------------------------------------------
// Box (unit cube, 6 faces, 4 verts each, correct normals and UVs)
// ---------------------------------------------------------------------------
PrimMesh MakeBox(float h) {
    PrimMesh mesh;

    struct FaceDef { glm::vec3 normal; glm::vec3 up; glm::vec3 right; };
    static const FaceDef faces[6] = {
        {{ 0, 0, 1},{ 0, 1, 0},{ 1, 0, 0}},  // +Z
        {{ 0, 0,-1},{ 0, 1, 0},{-1, 0, 0}},  // -Z
        {{ 0, 1, 0},{ 0, 0,-1},{ 1, 0, 0}},  // +Y
        {{ 0,-1, 0},{ 0, 0, 1},{ 1, 0, 0}},  // -Y
        {{ 1, 0, 0},{ 0, 1, 0},{ 0, 0,-1}},  // +X
        {{-1, 0, 0},{ 0, 1, 0},{ 0, 0, 1}},  // -X
    };

    static const glm::vec2 uvs[4] = {{0,0},{1,0},{1,1},{0,1}};

    for (auto& f : faces) {
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        // Quad corners: (-right-up), (+right-up), (+right+up), (-right+up)
        glm::vec3 corners[4] = {
            (f.normal - f.right - f.up) * h,
            (f.normal + f.right - f.up) * h,
            (f.normal + f.right + f.up) * h,
            (f.normal - f.right + f.up) * h,
        };
        glm::vec4 tangent = glm::vec4(f.right, 1.0f);
        for (int i = 0; i < 4; ++i) {
            PrimVertex v;
            v.pos     = corners[i];
            v.normal  = f.normal;
            v.uv      = uvs[i];
            v.tangent = tangent;
            mesh.vertices.push_back(v);
        }
        // Two triangles: 0,1,2 and 0,2,3
        mesh.indices.insert(mesh.indices.end(),
            {base, base+1, base+2, base, base+2, base+3});
    }
    return mesh;
}

// ---------------------------------------------------------------------------
// Sphere (UV sphere)
// ---------------------------------------------------------------------------
PrimMesh MakeSphere(float radius, int rings, int sectors) {
    PrimMesh mesh;
    const float pi  = glm::pi<float>();
    const float pi2 = 2.f * pi;

    for (int r = 0; r <= rings; ++r) {
        float phi = pi * static_cast<float>(r) / static_cast<float>(rings);
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);

        for (int s = 0; s <= sectors; ++s) {
            float theta = pi2 * static_cast<float>(s) / static_cast<float>(sectors);
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);

            glm::vec3 n{sinPhi * cosTheta, cosPhi, sinPhi * sinTheta};
            glm::vec2 uv{static_cast<float>(s) / static_cast<float>(sectors),
                         static_cast<float>(r) / static_cast<float>(rings)};
            // Tangent along increasing theta
            glm::vec3 t{-sinTheta, 0.f, cosTheta};

            PrimVertex v;
            v.pos     = n * radius;
            v.normal  = n;
            v.uv      = uv;
            v.tangent = glm::vec4(t, 1.0f);
            mesh.vertices.push_back(v);
        }
    }

    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            uint32_t a = static_cast<uint32_t>(r * (sectors + 1) + s);
            uint32_t b = a + static_cast<uint32_t>(sectors + 1);
            mesh.indices.insert(mesh.indices.end(),
                {a, b, a + 1, b, b + 1, a + 1});
        }
    }
    return mesh;
}

// ---------------------------------------------------------------------------
// Capsule (cylinder body + hemisphere caps)
// ---------------------------------------------------------------------------
PrimMesh MakeCapsule(float radius, float halfHeight, int rings, int sectors) {
    PrimMesh mesh;
    const float pi  = glm::pi<float>();
    const float pi2 = 2.f * pi;

    // Top hemisphere (phi from 0 to pi/2)
    int topRings = rings;
    for (int r = 0; r <= topRings; ++r) {
        float phi = (pi / 2.f) * static_cast<float>(r) / static_cast<float>(topRings);
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        for (int s = 0; s <= sectors; ++s) {
            float theta = pi2 * static_cast<float>(s) / static_cast<float>(sectors);
            glm::vec3 n{sinPhi * std::cos(theta), cosPhi, sinPhi * std::sin(theta)};
            PrimVertex v;
            v.pos    = n * radius + glm::vec3(0.f, halfHeight, 0.f);
            v.normal = n;
            v.uv     = {static_cast<float>(s)/static_cast<float>(sectors),
                        static_cast<float>(r)/(2.f*static_cast<float>(topRings))};
            v.tangent= glm::vec4(-std::sin(theta), 0.f, std::cos(theta), 1.0f);
            mesh.vertices.push_back(v);
        }
    }

    // Bottom hemisphere (phi from pi/2 to pi)
    int botRings = rings;
    for (int r = 0; r <= botRings; ++r) {
        float phi = (pi / 2.f) + (pi / 2.f) * static_cast<float>(r) / static_cast<float>(botRings);
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        for (int s = 0; s <= sectors; ++s) {
            float theta = pi2 * static_cast<float>(s) / static_cast<float>(sectors);
            glm::vec3 n{sinPhi * std::cos(theta), cosPhi, sinPhi * std::sin(theta)};
            PrimVertex v;
            v.pos    = n * radius + glm::vec3(0.f, -halfHeight, 0.f);
            v.normal = n;
            v.uv     = {static_cast<float>(s)/static_cast<float>(sectors),
                        0.5f + static_cast<float>(r)/(2.f*static_cast<float>(botRings))};
            v.tangent= glm::vec4(-std::sin(theta), 0.f, std::cos(theta), 1.0f);
            mesh.vertices.push_back(v);
        }
    }

    // Generate indices for both hemispheres
    int totalRings = topRings + botRings + 1; // +1 for the seam ring counted in both
    int rowVerts   = sectors + 1;
    for (int r = 0; r < totalRings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            uint32_t a = static_cast<uint32_t>(r * rowVerts + s);
            uint32_t b = a + static_cast<uint32_t>(rowVerts);
            mesh.indices.insert(mesh.indices.end(),
                {a, b, a + 1, b, b + 1, a + 1});
        }
    }
    return mesh;
}

// ---------------------------------------------------------------------------
// Ground plane (large flat quad)
// ---------------------------------------------------------------------------
PrimMesh MakeGround(float halfSize) {
    PrimMesh mesh;
    glm::vec3 n{0, 1, 0};
    glm::vec4 t{1, 0, 0, 1};
    float s = halfSize;

    mesh.vertices = {
        {glm::vec3(-s, 0.f,-s), n, {0,0}, t},
        {glm::vec3( s, 0.f,-s), n, {1,0}, t},
        {glm::vec3( s, 0.f, s), n, {1,1}, t},
        {glm::vec3(-s, 0.f, s), n, {0,1}, t},
    };
    mesh.indices = {0, 1, 2, 0, 2, 3};
    return mesh;
}

// ---------------------------------------------------------------------------
// Cylinder (closed caps, smooth normals on the barrel)
// ---------------------------------------------------------------------------
PrimMesh MakeCylinder(float radius, float halfHeight, int sectors) {
    PrimMesh mesh;
    const float pi = glm::pi<float>();
    const float step = 2.f * pi / static_cast<float>(sectors);

    // Side vertices (two rings)
    for (int i = 0; i <= sectors; ++i) {
        float a = static_cast<float>(i) * step;
        float cx = std::cos(a), cz = std::sin(a);
        glm::vec3 sideN(cx, 0.f, cz);
        float u = static_cast<float>(i) / static_cast<float>(sectors);
        glm::vec4 tang(cz, 0.f, -cx, 1.f);
        mesh.vertices.push_back({glm::vec3(cx * radius, -halfHeight, cz * radius), sideN, {u, 1.f}, tang});
        mesh.vertices.push_back({glm::vec3(cx * radius,  halfHeight, cz * radius), sideN, {u, 0.f}, tang});
    }
    // Side quads
    for (int i = 0; i < sectors; ++i) {
        uint32_t b = static_cast<uint32_t>(i * 2);
        mesh.indices.insert(mesh.indices.end(), {b, b+2, b+1,  b+1, b+2, b+3});
    }

    // Caps — bottom then top
    for (int cap = 0; cap < 2; ++cap) {
        float y    = (cap == 0) ? -halfHeight : halfHeight;
        glm::vec3 capN(0.f, (cap == 0) ? -1.f : 1.f, 0.f);
        glm::vec4 tang(1.f, 0.f, 0.f, 1.f);
        uint32_t centerIdx = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({glm::vec3(0.f, y, 0.f), capN, {0.5f, 0.5f}, tang});
        uint32_t ringStart = static_cast<uint32_t>(mesh.vertices.size());
        for (int i = 0; i <= sectors; ++i) {
            float a = static_cast<float>(i) * step;
            float cx = std::cos(a), cz = std::sin(a);
            mesh.vertices.push_back({glm::vec3(cx * radius, y, cz * radius), capN,
                                     {0.5f + 0.5f * cx, 0.5f + 0.5f * cz}, tang});
        }
        for (int i = 0; i < sectors; ++i) {
            uint32_t a = ringStart + static_cast<uint32_t>(i);
            uint32_t b = ringStart + static_cast<uint32_t>(i + 1);
            if (cap == 0)
                mesh.indices.insert(mesh.indices.end(), {centerIdx, b, a});
            else
                mesh.indices.insert(mesh.indices.end(), {centerIdx, a, b});
        }
    }
    return mesh;
}

// ---------------------------------------------------------------------------
// Cone (closed base cap, apex at top)
// ---------------------------------------------------------------------------
PrimMesh MakeCone(float baseRadius, float height, int sectors) {
    PrimMesh mesh;
    const float pi = glm::pi<float>();
    const float step = 2.f * pi / static_cast<float>(sectors);
    const float slopeLen = std::sqrt(baseRadius * baseRadius + height * height);
    const float sinA = baseRadius / slopeLen;   // for sidewall normals
    const float cosA = height    / slopeLen;

    // Sidewall: one ring of base vertices + one apex per sector strip
    uint32_t apexBase = static_cast<uint32_t>(mesh.vertices.size());
    for (int i = 0; i < sectors; ++i) {
        float a0 = static_cast<float>(i)     * step;
        float a1 = static_cast<float>(i + 1) * step;
        float aMid = 0.5f * (a0 + a1);

        glm::vec3 n0(std::cos(a0) * cosA, sinA, std::sin(a0) * cosA);
        glm::vec3 n1(std::cos(a1) * cosA, sinA, std::sin(a1) * cosA);
        glm::vec3 nApex(std::cos(aMid) * cosA, sinA, std::sin(aMid) * cosA);

        uint32_t vi = static_cast<uint32_t>(mesh.vertices.size());
        float u0 = static_cast<float>(i)   / static_cast<float>(sectors);
        float u1 = static_cast<float>(i+1) / static_cast<float>(sectors);
        glm::vec4 tang(std::cos(a0 + glm::half_pi<float>()), 0.f,
                       std::sin(a0 + glm::half_pi<float>()), 1.f);
        mesh.vertices.push_back({glm::vec3(std::cos(a0)*baseRadius, 0.f, std::sin(a0)*baseRadius),
                                  n0, {u0, 1.f}, tang});
        mesh.vertices.push_back({glm::vec3(std::cos(a1)*baseRadius, 0.f, std::sin(a1)*baseRadius),
                                  n1, {u1, 1.f}, tang});
        mesh.vertices.push_back({glm::vec3(0.f, height, 0.f), nApex, {(u0+u1)*0.5f, 0.f}, tang});
        mesh.indices.insert(mesh.indices.end(), {vi, vi+2, vi+1});
    }
    (void)apexBase;

    // Base cap
    glm::vec3 capN(0.f, -1.f, 0.f);
    glm::vec4 tang(1.f, 0.f, 0.f, 1.f);
    uint32_t centerIdx = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({glm::vec3(0.f, 0.f, 0.f), capN, {0.5f, 0.5f}, tang});
    uint32_t ringStart = static_cast<uint32_t>(mesh.vertices.size());
    for (int i = 0; i <= sectors; ++i) {
        float a = static_cast<float>(i) * step;
        float cx = std::cos(a), cz = std::sin(a);
        mesh.vertices.push_back({glm::vec3(cx * baseRadius, 0.f, cz * baseRadius), capN,
                                 {0.5f + 0.5f * cx, 0.5f + 0.5f * cz}, tang});
    }
    for (int i = 0; i < sectors; ++i) {
        uint32_t a = ringStart + static_cast<uint32_t>(i);
        uint32_t b = ringStart + static_cast<uint32_t>(i + 1);
        mesh.indices.insert(mesh.indices.end(), {centerIdx, b, a});
    }
    return mesh;
}
