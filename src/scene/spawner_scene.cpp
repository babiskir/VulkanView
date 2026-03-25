#include "spawner_scene.h"

#include "camera_component.h"
#include "engine.h"
#include "mesh_component.h"
#include "physics_system.h"
#include "primitive_meshes.h"
#include "renderer.h"
#include "transform_component.h"
#include "wave_physics.h"
#include "water_system.h"

#include "imgui/imgui.h"
#include "imgui_system.h"

#include <PxPhysicsAPI.h>
#include <extensions/PxExtensionsAPI.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace physx;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static PxPhysics*  getPx (Engine* e) { return static_cast<PxPhysics*> (e->GetPhysicsSystem()->GetRawPhysics()); }
static PxScene*    getSc (Engine* e) { return static_cast<PxScene*>   (e->GetPhysicsSystem()->GetRawScene()); }
static PxMaterial* getMat(Engine* e) { return static_cast<PxMaterial*>(e->GetPhysicsSystem()->GetRawMaterial()); }

static glm::vec3 fromPx(const PxVec3& v) { return {v.x, v.y, v.z}; }
static PxVec3    toPx  (glm::vec3 v)     { return {v.x, v.y, v.z}; }

// ---------------------------------------------------------------------------
// Spawnable shape types  (mirrors EnoSea ANGLEDemo shape list)
// ---------------------------------------------------------------------------
enum class SpawnShape { Box, Sphere, Capsule, Cylinder, Cone };

static const char* ShapeName(SpawnShape s) {
    switch (s) {
        case SpawnShape::Box:      return "Box";
        case SpawnShape::Sphere:   return "Sphere";
        case SpawnShape::Capsule:  return "Capsule";
        case SpawnShape::Cylinder: return "Cylinder";
        case SpawnShape::Cone:     return "Cone";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// SpawnedObject — holds a physics actor + entity
// ---------------------------------------------------------------------------
struct SpawnedObj {
    PxRigidDynamic* actor  = nullptr;
    Entity*         entity = nullptr;
    SpawnShape      shape  = SpawnShape::Box;
    std::string     name;
};

// ---------------------------------------------------------------------------
// SpawnerScene
// ---------------------------------------------------------------------------
class SpawnerScene {
public:
    explicit SpawnerScene(Engine* engine)
        : m_engine(engine)
    {
        m_rng.seed(42);
    }

    // ------------------------------------------------------------------
    // Public interface called by UpdatePublisher subscribers
    // ------------------------------------------------------------------
    void ApplyBuoyancy(float dt) { applyBuoyancy(dt); }
    void DrawUI()                { drawUI(); }

    bool Initialize() {
        auto* renderer = m_engine->GetRenderer();
        if (!renderer) return false;

        // Camera overlooking the spawn area
        auto* camEntity = m_engine->CreateEntity("Camera");
        auto* xform     = camEntity->AddComponent<TransformComponent>();
        xform->SetPosition({0.f, 20.f, -35.f});
        auto* cam = camEntity->AddComponent<CameraComponent>();
        cam->SetAspectRatio(800.f / 600.f);
        cam->SetFieldOfView(60.f);
        cam->SetClipPlanes(0.1f, 2000.f);
        cam->SetTarget({0.f, 0.f, 0.f});
        m_engine->SetActiveCamera(cam);

        // JONSWAP sea state for buoyancy sampling
        SeaState sea;
        JONSWAPConfig cfg;
        cfg.Hs            = 2.0f;
        cfg.Tp            = 8.0f;
        cfg.directionDeg  = 0.f;
        cfg.gamma         = 3.3f;
        cfg.numComponents = 16;
        sea.addJONSWAP(cfg);
        m_seaState = std::move(sea);

        // Visual water
        m_waterSystem = new WaterSystem();
        if (m_waterSystem->Initialize(renderer)) {
            renderer->waterSystem = m_waterSystem;
        } else {
            delete m_waterSystem;
            m_waterSystem = nullptr;
        }

        // Ground plane (static)
        auto* px  = getPx(m_engine);
        auto* sc  = getSc(m_engine);
        auto* mat = getMat(m_engine);
        PxRigidStatic* ground = PxCreatePlane(*px, PxPlane(0, 1, 0, m_groundDepth), *mat);
        sc->addActor(*ground);

        // Spawn the 5 initial shapes (one of each type)
        std::vector<Entity*> batch;
        const float startY = m_groundDepth + 10.f;   // above water surface
        spawnShape(SpawnShape::Box,      {-8.f, startY,  0.f}, batch);
        spawnShape(SpawnShape::Sphere,   {-4.f, startY,  0.f}, batch);
        spawnShape(SpawnShape::Capsule,  { 0.f, startY,  0.f}, batch);
        spawnShape(SpawnShape::Cylinder, { 4.f, startY,  0.f}, batch);
        spawnShape(SpawnShape::Cone,     { 8.f, startY,  0.f}, batch);

        renderer->EnqueueEntityPreallocationBatch(batch);

        std::cout << "[SpawnerScene] Initialized with " << m_objects.size() << " objects.\n";
        return true;
    }

private:
    // -----------------------------------------------------------------------
    // Mesh helpers
    // -----------------------------------------------------------------------
    static void fillMesh(MeshComponent* mc, const PrimMesh& prim) {
        std::vector<Vertex> verts;
        verts.reserve(prim.vertices.size());
        for (const auto& pv : prim.vertices) {
            Vertex v;
            v.position = pv.pos;
            v.normal   = pv.normal;
            v.texCoord = pv.uv;
            v.tangent  = pv.tangent;
            verts.push_back(v);
        }
        mc->SetVertices(verts);
        mc->SetIndices(prim.indices);
    }

    // -----------------------------------------------------------------------
    // Spawn a single shape at world position pos
    // -----------------------------------------------------------------------
    void spawnShape(SpawnShape shape, glm::vec3 pos, std::vector<Entity*>& outBatch) {
        static int counter = 0;
        std::string nm = std::string(ShapeName(shape)) + "_" + std::to_string(counter++);

        auto* entity = m_engine->CreateEntity(nm);
        auto* t      = entity->AddComponent<TransformComponent>();
        t->SetPosition(pos);
        t->SetScale(glm::vec3(1.f));

        auto* mc = entity->AddComponent<MeshComponent>();
        mc->SetTexturePath("__shared_default_albedo__");

        auto* px  = getPx(m_engine);
        auto* sc  = getSc(m_engine);
        auto* mat = getMat(m_engine);

        PxRigidDynamic* actor = nullptr;
        PxTransform     pose(toPx(pos));

        switch (shape) {
        case SpawnShape::Box:
            fillMesh(mc, MakeBox(0.5f));
            actor = PxCreateDynamic(*px, pose, PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat, 1.f);
            break;
        case SpawnShape::Sphere:
            fillMesh(mc, MakeSphere(0.5f));
            actor = PxCreateDynamic(*px, pose, PxSphereGeometry(0.5f), *mat, 1.f);
            break;
        case SpawnShape::Capsule: {
            fillMesh(mc, MakeCapsule(0.25f, 0.5f));
            PxTransform yUp(PxQuat(PxHalfPi, PxVec3(0, 0, 1)));
            actor = PxCreateDynamic(*px, pose, PxCapsuleGeometry(0.25f, 0.5f), *mat, 1.f, yUp);
            break;
        }
        case SpawnShape::Cylinder:
            // PhysX has no native cylinder — approximate with a convex hull mesh.
            // For simplicity use a box of same proportions as a proxy physics shape.
            fillMesh(mc, MakeCylinder(0.5f, 0.5f));
            actor = PxCreateDynamic(*px, pose, PxBoxGeometry(0.45f, 0.5f, 0.45f), *mat, 1.f);
            break;
        case SpawnShape::Cone:
            fillMesh(mc, MakeCone(0.5f, 1.0f));
            // Approximate physics with a scaled box
            actor = PxCreateDynamic(*px, pose, PxBoxGeometry(0.35f, 0.5f, 0.35f), *mat, 1.f);
            break;
        }

        if (!actor) return;
        actor->setAngularDamping(0.4f);
        actor->setLinearDamping(0.1f);
        sc->addActor(*actor);
        m_engine->GetPhysicsSystem()->RegisterActor(actor, entity, /*isDynamic=*/true);

        m_objects.push_back({actor, entity, shape, nm});
        outBatch.push_back(entity);
    }

    // -----------------------------------------------------------------------
    // Simplified buoyancy: apply upward force proportional to submersion depth
    // -----------------------------------------------------------------------
    void applyBuoyancy(float dt) {
        if (m_objects.empty()) return;
        float t = m_simTime;
        m_simTime += dt;

        constexpr float waterDensity = 1025.f;
        constexpr float g            = 9.81f;
        constexpr float volume       = 1.f;       // approximate unit volume per object

        for (auto& obj : m_objects) {
            if (!obj.actor) continue;
            PxTransform pose = obj.actor->getGlobalPose();
            float wy = m_seaState.getElevation(pose.p.x, pose.p.z, t);
            float submersion = wy - pose.p.y;   // positive = submerged
            if (submersion > 0.f) {
                float force = waterDensity * g * volume * std::min(submersion, 1.f);
                obj.actor->addForce(PxVec3(0.f, force, 0.f));
            }
        }
    }

    // -----------------------------------------------------------------------
    // ImGui panel
    // -----------------------------------------------------------------------
    void drawUI() {
        ImGui::Begin("Object Spawner");

        ImGui::TextUnformatted("EnoSea-style Spawner Demo");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Active objects: %d", (int)m_objects.size());
        ImGui::Spacing();

        // Spawn buttons
        if (ImGui::Button("Spawn Box",      ImVec2(120, 0))) spawnRandom(SpawnShape::Box);
        ImGui::SameLine();
        if (ImGui::Button("Spawn Sphere",   ImVec2(120, 0))) spawnRandom(SpawnShape::Sphere);
        if (ImGui::Button("Spawn Capsule",  ImVec2(120, 0))) spawnRandom(SpawnShape::Capsule);
        ImGui::SameLine();
        if (ImGui::Button("Spawn Cylinder", ImVec2(120, 0))) spawnRandom(SpawnShape::Cylinder);
        if (ImGui::Button("Spawn Cone",     ImVec2(120, 0))) spawnRandom(SpawnShape::Cone);
        ImGui::Spacing();

        // Random spawn
        if (ImGui::Button("Spawn Random [SPACE]", ImVec2(250, 0)))
            spawnRandomAny();

        ImGui::Separator();

        // Ocean parameters
        ImGui::TextUnformatted("Ocean (JONSWAP)");
        bool changed = false;
        changed |= ImGui::SliderFloat("Hs (m)",   &m_Hs,  0.5f, 10.f, "%.1f");
        changed |= ImGui::SliderFloat("Tp (s)",   &m_Tp,  3.f,  20.f, "%.1f");
        changed |= ImGui::SliderFloat("Dir (deg)",&m_Dir, 0.f, 360.f, "%.0f");
        if (changed) {
            m_seaState.clear();
            JONSWAPConfig cfg;
            cfg.Hs            = m_Hs;
            cfg.Tp            = m_Tp;
            cfg.directionDeg  = m_Dir;
            cfg.gamma         = 3.3f;
            cfg.numComponents = 16;
            m_seaState.addJONSWAP(cfg);
        }

        ImGui::End();
    }

    // Spawn at a random position above the water
    void spawnRandom(SpawnShape shape) {
        std::uniform_real_distribution<float> xzDist(-8.f, 8.f);
        std::uniform_real_distribution<float> yDist(m_groundDepth + 8.f, m_groundDepth + 20.f);
        glm::vec3 pos{xzDist(m_rng), yDist(m_rng), xzDist(m_rng)};
        std::vector<Entity*> batch;
        spawnShape(shape, pos, batch);
        if (auto* r = m_engine->GetRenderer())
            r->EnqueueEntityPreallocationBatch(batch);
    }

    void spawnRandomAny() {
        std::uniform_int_distribution<int> shapeDist(0, 4);
        spawnRandom(static_cast<SpawnShape>(shapeDist(m_rng)));
    }

    // -----------------------------------------------------------------------
    Engine*     m_engine      = nullptr;
    WaterSystem* m_waterSystem = nullptr;
    SeaState    m_seaState;
    float       m_groundDepth = 0.f;      // Y of the ground/water surface
    float       m_simTime     = 0.f;
    float       m_Hs          = 2.0f;
    float       m_Tp          = 8.0f;
    float       m_Dir         = 0.f;
    std::mt19937 m_rng;
    std::vector<SpawnedObj> m_objects;
};

// ---------------------------------------------------------------------------
// Scene entry point
// ---------------------------------------------------------------------------
void LoadSpawnerScene(Engine* engine) {
    auto* renderer = engine->GetRenderer();
    if (!renderer) return;

    auto scene = std::make_shared<SpawnerScene>(engine);
    if (!scene->Initialize()) {
        std::cerr << "[SpawnerScene] Initialization failed.\n";
        return;
    }

    // Physics (buoyancy forces) run at the engine's fixed timestep via UpdatePublisher.
    engine->SubscribeSceneFixedUpdate([scene](float dt) {
        scene->ApplyBuoyancy(dt);
    });

    // ImGui panel drawn every frame via SetSceneUI.
    engine->SetSceneUI([scene]() {
        scene->DrawUI();
    });
}
