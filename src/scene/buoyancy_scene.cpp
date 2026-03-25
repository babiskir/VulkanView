// buoyancy_scene.cpp
//
// EnoSea-inspired buoyancy demo for VulkanView.
//
// Combines:
//   - JONSWAP / Pierson-Moskowitz ocean spectra (wave_physics.h)
//   - Analytical mesh waterline slicing (ported from EnoSea SubmergedMesh)
//   - PhysX 5 rigid body simulation (PhysicsSystem)
//   - Tessendorf visual ocean (WaterSystem)
//   - ImGui panel for real-time wave & physics controls
//

#include "buoyancy_scene.h"

#include "camera_component.h"
#include "engine.h"
#include "entity.h"
#include "mesh_component.h"
#include "physics_system.h"
#include "primitive_meshes.h"
#include "renderer.h"
#include "scene_ui.h"
#include "transform_component.h"
#include "water_system.h"
#include "wave_physics.h"

#include "imgui/imgui.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// =============================================================================
// Waterline mesh slicing (adapted from EnoSea SubmergedMesh, using glm)
// =============================================================================

struct SubTri {
    glm::vec3 v0, v1, v2;
    glm::vec3 center;
    glm::vec3 normal;
    float     area  = 0.f;
    float     depth = 0.f;  // positive = distance below waterline
};

static glm::vec3 interpWaterline(const glm::vec3& a, const glm::vec3& b,
                                  float da, float db) {
    // da < 0 (above water), db > 0 (below water)
    // solve da + t*(db-da) = 0  →  t = -da/(db-da)
    float t = -da / (db - da);
    return a + t * (b - a);
}

static SubTri buildSubTri(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2,
                           const SeaState& sea, float t) {
    SubTri tri;
    tri.v0 = p0; tri.v1 = p1; tri.v2 = p2;
    tri.center = (p0 + p1 + p2) / 3.0f;

    glm::vec3 e1    = p1 - p0;
    glm::vec3 e2    = p2 - p0;
    glm::vec3 cross = glm::cross(e1, e2);
    float     mag   = glm::length(cross);
    tri.normal = (mag > 1e-9f) ? cross / mag : glm::vec3(0.f, 1.f, 0.f);
    tri.area   = 0.5f * mag;

    float waveY = sea.getElevation(tri.center.x, tri.center.z, t);
    tri.depth   = waveY - tri.center.y;  // positive = submerged
    return tri;
}

/// Slice a mesh at the wave surface and return submerged triangles.
/// Winding is preserved from the input mesh so that cross-product normals
/// point outward (consistent with the hydrostatic pressure formula).
static std::vector<SubTri> sliceAtWaterline(
        const std::vector<glm::vec3>& localVerts,
        const std::vector<uint32_t>&  indices,
        const glm::mat4&              worldMat,
        const SeaState&               sea,
        float                         t) {

    // Transform vertices to world space and sample wave height per vertex
    const size_t N = localVerts.size();
    std::vector<glm::vec3> wv(N);
    std::vector<float>     d(N);
    for (size_t i = 0; i < N; ++i) {
        wv[i] = glm::vec3(worldMat * glm::vec4(localVerts[i], 1.f));
        d[i]  = sea.getElevation(wv[i].x, wv[i].z, t) - wv[i].y; // positive = submerged
    }

    std::vector<SubTri> result;

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const int i0 = static_cast<int>(indices[i]);
        const int i1 = static_cast<int>(indices[i + 1]);
        const int i2 = static_cast<int>(indices[i + 2]);

        const float d0 = d[i0], d1 = d[i1], d2 = d[i2];
        const int aboveCount = (d0 < 0.f ? 1 : 0)
                             + (d1 < 0.f ? 1 : 0)
                             + (d2 < 0.f ? 1 : 0);

        if (aboveCount == 3) continue;  // fully above water — skip

        if (aboveCount == 0) {
            // Fully submerged — emit with original winding preserved
            result.push_back(buildSubTri(wv[i0], wv[i1], wv[i2], sea, t));
            continue;
        }

        // Partial submersion: sort vertices by depth (most submerged first).
        // Track original position (0,1,2) within the triangle to restore winding.
        struct VD { glm::vec3 p; float dist; int origPos; };
        VD vs[3] = { {wv[i0], d0, 0}, {wv[i1], d1, 1}, {wv[i2], d2, 2} };
        if (vs[0].dist < vs[1].dist) std::swap(vs[0], vs[1]);
        if (vs[1].dist < vs[2].dist) std::swap(vs[1], vs[2]);
        if (vs[0].dist < vs[1].dist) std::swap(vs[0], vs[1]);
        // vs[0]=deepest (L), vs[1]=middle (M), vs[2]=shallowest/above (H)

        if (aboveCount == 1) {
            // vs[2] is the only vertex above water.
            // Clip edges H→M and H→L at the waterline.
            glm::vec3 iM = interpWaterline(vs[2].p, vs[1].p, vs[2].dist, vs[1].dist);
            glm::vec3 iL = interpWaterline(vs[2].p, vs[0].p, vs[2].dist, vs[0].dist);

            // Winding preservation: which vertex follows H in original CCW order?
            int nextOfH = (vs[2].origPos + 1) % 3;
            if (vs[1].origPos == nextOfH) {
                // Original winding: H → M → L
                result.push_back(buildSubTri(vs[1].p, iM, iL, sea, t));
                result.push_back(buildSubTri(vs[1].p, iL, vs[0].p, sea, t));
            } else {
                // Original winding: H → L → M
                result.push_back(buildSubTri(vs[0].p, iL, iM, sea, t));
                result.push_back(buildSubTri(vs[0].p, iM, vs[1].p, sea, t));
            }
        } else {
            // aboveCount == 2: vs[1] and vs[2] are above water.
            // Clip edges H→L and M→L at the waterline.
            glm::vec3 jH = interpWaterline(vs[2].p, vs[0].p, vs[2].dist, vs[0].dist);
            glm::vec3 jM = interpWaterline(vs[1].p, vs[0].p, vs[1].dist, vs[0].dist);

            int nextOfL = (vs[0].origPos + 1) % 3;
            if (vs[1].origPos == nextOfL) {
                // Original winding: L → M → H
                result.push_back(buildSubTri(vs[0].p, jM, jH, sea, t));
            } else {
                // Original winding: L → H → M
                result.push_back(buildSubTri(vs[0].p, jH, jM, sea, t));
            }
        }
    }

    return result;
}

// =============================================================================
// BuoyancyScene
// =============================================================================

struct FloatingObj {
    Entity*    entity    = nullptr;
    RigidBody* body      = nullptr;
    std::vector<glm::vec3> localVerts;
    std::vector<uint32_t>  localIdx;
    std::string name;
    int lastSubTris = 0;
    int totalTris   = 0;
};

class BuoyancyScene {
public:
    explicit BuoyancyScene(Engine* engine);

    /** Called from UpdatePublisher FixedUpdate — advances physics at fixed rate. */
    void ApplyBuoyancy(float dt) {
        totalTime += dt;
        applyBuoyancyAll();
    }

    void DrawUI(Engine* engine);

private:
    // -- Wave parameters (editable in ImGui) --
    float Hs       = 1.5f;
    float Tp       = 7.0f;
    float dirDeg   = 0.0f;
    float gamma    = 3.3f;
    int   numComps = 30;
    int   specIdx  = 0;     // 0 = JONSWAP, 1 = Pierson-Moskowitz
    bool  addSwell = false;
    float swellHs  = 0.3f;
    float swellTp  = 12.0f;
    float swellDir = 90.0f;

    // -- Physics parameters --
    float waterDensity = 1025.f;  // kg/m³ (seawater)
    float objectDensity = 800.f;  // kg/m³ (~wood, floats 78% submerged)

    // -- Runtime state --
    SeaState sea;
    float    totalTime   = 0.f;
    int      spawnCounter = 0;

    std::vector<FloatingObj>         objs;
    std::shared_ptr<SceneBrowserPanel>  browser;
    std::shared_ptr<InspectorPanel>     inspector;
    std::shared_ptr<SceneSelection>     selection;

    void rebuildSea();
    void spawnObject(Engine* engine, int typeIdx, glm::vec3 pos);
    void applyBuoyancyAll();
};

// ---------------------------------------------------------------------------
// Constructor — sets up camera, water, and initial objects
// ---------------------------------------------------------------------------
BuoyancyScene::BuoyancyScene(Engine* engine) {
    browser   = std::make_shared<SceneBrowserPanel>();
    inspector = std::make_shared<InspectorPanel>();
    selection = std::make_shared<SceneSelection>();

    rebuildSea();

    // Camera
    auto* camEnt = engine->CreateEntity("Camera");
    auto* xform  = camEnt->AddComponent<TransformComponent>();
    xform->SetPosition({0.f, 8.f, 18.f});
    auto* cam = camEnt->AddComponent<CameraComponent>();
    cam->SetAspectRatio(800.f / 600.f);
    cam->SetFieldOfView(60.f);
    cam->SetClipPlanes(0.1f, 1000.f);
    cam->SetTarget({0.f, 0.f, 0.f});
    engine->SetActiveCamera(cam);

    // Tessendorf water visual (WaterSystem)
    auto* renderer = engine->GetRenderer();
    if (renderer) {
        auto* ws            = new WaterSystem();
        ws->heightAmplitude = Hs;
        ws->choppiness      = 1.0f;
        ws->windSpeed       = 12.f;
        ws->windAngleDeg    = dirDeg;
        if (ws->Initialize(renderer))
            renderer->waterSystem = ws;
        else
            delete ws;
    }

    // Spawn initial 6 objects in a row
    const struct { int typeIdx; glm::vec3 pos; } spawns[] = {
        {0, {-7.5f, 3.f, 0.f}}, {1, {-4.5f, 3.f, 0.f}}, {2, {-1.5f, 3.f, 0.f}},
        {0, { 1.5f, 3.f, 0.f}}, {1, { 4.5f, 3.f, 0.f}}, {2, { 7.5f, 3.f, 0.f}},
    };
    for (const auto& s : spawns)
        spawnObject(engine, s.typeIdx, s.pos);

    std::cout << "[BuoyancyScene] Ready — "
              << sea.getComponentCount() << " wave components, "
              << objs.size() << " floating objects.\n";
}

// ---------------------------------------------------------------------------
void BuoyancyScene::rebuildSea() {
    sea.clear();
    if (specIdx == 0) {
        JONSWAPConfig cfg;
        cfg.Hs = Hs; cfg.Tp = Tp; cfg.directionDeg = dirDeg;
        cfg.gamma = gamma; cfg.numComponents = numComps;
        sea.addJONSWAP(cfg);
    } else {
        sea.addPiersonMoskowitz(Hs, Tp, numComps, dirDeg);
    }
    if (addSwell) {
        SwellConfig sw;
        sw.height = swellHs; sw.period = swellTp; sw.directionDeg = swellDir;
        sea.addSwell(sw);
    }
}

// ---------------------------------------------------------------------------
void BuoyancyScene::spawnObject(Engine* engine, int typeIdx, glm::vec3 pos) {
    auto* physics = engine->GetPhysicsSystem();
    auto* renderer = engine->GetRenderer();
    if (!physics) return;

    const char*     names[]  = {"Box", "Sphere", "Capsule"};
    CollisionShape  shapes[] = {CollisionShape::Box, CollisionShape::Sphere,
                                CollisionShape::Capsule};

    const int type = typeIdx % 3;
    std::string name = std::string(names[type]) + "_" + std::to_string(spawnCounter++);

    auto* ent   = engine->CreateEntity(name);
    auto* xform = ent->AddComponent<TransformComponent>();
    xform->SetPosition(pos);
    xform->SetScale(glm::vec3(1.f));

    PrimMesh prim;
    if      (type == 0) prim = MakeBox(0.5f);
    else if (type == 1) prim = MakeSphere(0.5f, 16, 32);
    else                prim = MakeCapsule(0.3f, 0.4f, 8, 16);

    auto* mesh = ent->AddComponent<MeshComponent>();
    {
        std::vector<Vertex> verts;
        verts.reserve(prim.vertices.size());
        for (const auto& pv : prim.vertices) {
            Vertex v{};
            v.position = pv.pos;
            v.normal   = pv.normal;
            v.texCoord = pv.uv;
            v.tangent  = pv.tangent;
            verts.push_back(v);
        }
        mesh->SetVertices(verts);
        mesh->SetIndices(prim.indices);
        mesh->SetTexturePath("__shared_default_albedo__");
    }

    // Compute volume-scaled mass so all objects float at the same depth fraction
    float volume = 0.f;
    if (type == 0) volume = 1.f * 1.f * 1.f;          // 1m box
    else if (type == 1) volume = (4.f/3.f)*3.14159f*0.5f*0.5f*0.5f;
    else volume = 3.14159f*0.3f*0.3f*(0.8f + (4.f/3.f)*0.3f); // capsule approximation
    float mass = objectDensity * volume;

    auto* body = physics->CreateRigidBody(ent, shapes[type], mass);

    FloatingObj obj;
    obj.entity    = ent;
    obj.body      = body;
    obj.name      = name;
    obj.totalTris = static_cast<int>(prim.indices.size() / 3);
    obj.localVerts.reserve(prim.vertices.size());
    for (const auto& pv : prim.vertices) obj.localVerts.push_back(pv.pos);
    obj.localIdx = prim.indices;
    objs.push_back(std::move(obj));

    if (renderer) {
        std::vector<Entity*> batch{ent};
        renderer->preAllocateEntityResourcesBatch(batch);
    }
}

// ---------------------------------------------------------------------------
void BuoyancyScene::applyBuoyancyAll() {
    constexpr float g = 9.81f;

    for (auto& obj : objs) {
        if (!obj.body) continue;

        glm::vec3 pos = obj.body->GetPosition();
        glm::quat rot = obj.body->GetRotation();
        glm::mat4 worldMat = glm::translate(glm::mat4(1.f), pos) * glm::mat4_cast(rot);

        auto subTris = sliceAtWaterline(obj.localVerts, obj.localIdx,
                                        worldMat, sea, totalTime);
        obj.lastSubTris = static_cast<int>(subTris.size());

        // Integrate hydrostatic pressure over submerged triangles.
        // F.y = -ρ g (n.y × depth × area)  per panel.
        // This is equivalent to Archimedes when normals are outward-facing.
        // We accumulate the total force and apply at the body's centre of mass.
        glm::vec3 totalForce{0.f};
        for (const auto& tri : subTris) {
            if (tri.depth <= 0.f) continue;
            float volY = tri.normal.y * tri.depth * tri.area;
            totalForce.y += -g * waterDensity * volY;
        }
        if (totalForce.y != 0.f)
            obj.body->ApplyForce(totalForce);
    }
}

// ---------------------------------------------------------------------------
void BuoyancyScene::DrawUI(Engine* engine) {
    // --- Scene browser + inspector ---
    auto* cam      = engine->GetActiveCamera();
    auto* renderer = engine->GetRenderer();
    auto* imgui    = engine->GetImGuiSystem();
    if (!imgui || imgui->panelState.showScene)
        browser->Draw(engine, *selection);
    if (!imgui || imgui->panelState.showInspector)
        inspector->Draw(engine, renderer, cam, *selection);

    // --- Ocean Environment panel ---
    ImGui::SetNextWindowSize(ImVec2(340, 480), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Ocean Environment")) {

        // Wave Spectrum
        if (ImGui::CollapsingHeader("Wave Spectrum", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* specNames[] = {"JONSWAP (wind sea)", "Pierson-Moskowitz"};
            bool changed = ImGui::Combo("Spectrum##type", &specIdx, specNames, 2);
            changed |= ImGui::SliderFloat("Hs (m)##buoy",  &Hs,  0.0f, 6.0f,  "%.2f");
            changed |= ImGui::SliderFloat("Tp (s)##buoy",  &Tp,  2.0f, 20.0f, "%.1f");
            changed |= ImGui::SliderFloat("Dir (deg)##buoy", &dirDeg, 0.f, 360.f, "%.0f");
            if (specIdx == 0)
                changed |= ImGui::SliderFloat("Gamma##buoy", &gamma, 1.0f, 7.0f, "%.1f");
            changed |= ImGui::SliderInt("Components##buoy", &numComps, 5, 80);

            ImGui::Spacing();
            changed |= ImGui::Checkbox("Add Swell##buoy", &addSwell);
            if (addSwell) {
                changed |= ImGui::SliderFloat("Swell Hs##buoy",  &swellHs,  0.f, 3.f, "%.2f");
                changed |= ImGui::SliderFloat("Swell Tp##buoy",  &swellTp,  5.f, 25.f, "%.1f");
                changed |= ImGui::SliderFloat("Swell Dir##buoy", &swellDir, 0.f, 360.f, "%.0f");
            }
            if (changed || ImGui::Button("Rebuild Sea State")) {
                rebuildSea();
                // Sync WaterSystem wind parameters
                if (renderer && renderer->waterSystem) {
                    renderer->waterSystem->heightAmplitude = Hs;
                    renderer->waterSystem->windAngleDeg    = dirDeg;
                }
            }
            ImGui::TextDisabled("Wave components: %d", sea.getComponentCount());
        }

        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Water density (kg/m3)", &waterDensity, 800.f, 1050.f, "%.0f");
        }

        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Floating Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("  %-18s  %s", "Name", "Sub.%");
            ImGui::Separator();
            for (const auto& obj : objs) {
                float pct = (obj.totalTris > 0)
                    ? 100.f * static_cast<float>(obj.lastSubTris)
                            / static_cast<float>(obj.totalTris)
                    : 0.f;
                ImGui::Text("  %-18s  %.0f%%", obj.name.c_str(), pct);
            }
            ImGui::Spacing();
            ImGui::TextDisabled("Spawn more objects:");
            if (ImGui::Button("Box"))    spawnObject(engine, 0, {0.f, 5.f, 0.f});
            ImGui::SameLine();
            if (ImGui::Button("Sphere")) spawnObject(engine, 1, {0.f, 5.f, 0.f});
            ImGui::SameLine();
            if (ImGui::Button("Capsule"))spawnObject(engine, 2, {0.f, 5.f, 0.f});
        }
    }
    ImGui::End();
}

// =============================================================================
// Factory function
// =============================================================================
void LoadBuoyancyScene(Engine* engine) {
    auto scene = std::make_shared<BuoyancyScene>(engine);

    // Buoyancy forces run at the engine's fixed timestep (50 Hz) via UpdatePublisher.
    engine->SubscribeSceneFixedUpdate([scene](float dt) {
        scene->ApplyBuoyancy(dt);
    });

    // ImGui panel drawn every frame.
    engine->SetSceneUI([scene, engine]() mutable {
        scene->DrawUI(engine);
    });
}
