#include "physics_scenes.h"

#include <PxPhysicsAPI.h>
#include <extensions/PxExtensionsAPI.h>   // for joints (PxSphericalJointCreate etc.)

#include "engine.h"
#include "imgui_system.h"
#include "physics_system.h"
#include "primitive_meshes.h"
#include "renderer.h"
#include "mesh_component.h"
#include "transform_component.h"
#include "camera_component.h"
#include "entity.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <string>
#include <random>
#include <iostream>

using namespace physx;

// ---------------------------------------------------------------------------
// Helpers shared by all snippet scenes
// ---------------------------------------------------------------------------

// Get raw PhysX objects from the engine's physics system — log and abort load if null
static PxPhysics*  getPhysics (Engine* e) {
    auto* p = static_cast<PxPhysics*>(e->GetPhysicsSystem()->GetRawPhysics());
    if (!p) ImGuiSystem::DebugLog("[PhysX] ERROR: GetRawPhysics() returned null!");
    return p;
}
static PxScene*    getScene   (Engine* e) {
    auto* s = static_cast<PxScene*>(e->GetPhysicsSystem()->GetRawScene());
    if (!s) ImGuiSystem::DebugLog("[PhysX] ERROR: GetRawScene() returned null!");
    return s;
}
static PxMaterial* getMaterial(Engine* e) {
    auto* m = static_cast<PxMaterial*>(e->GetPhysicsSystem()->GetRawMaterial());
    if (!m) ImGuiSystem::DebugLog("[PhysX] ERROR: GetRawMaterial() returned null!");
    return m;
}

// Enqueue GPU resource creation for entities (no loading screen shown).
static void endLoad(Engine* e, std::vector<Entity*>& entities) {
    if (auto* r = e->GetRenderer()) {
        // Use the deferred queue so GPU resource creation happens at the render thread's
        // safe point (after fence wait) rather than racing with the background upload workers.
        r->EnqueueEntityPreallocationBatch(entities);
    }
}

// glm ↔ PhysX conversion (defined but possibly unused in some TUs — suppress warning)
[[maybe_unused]] static PxVec3  toPx(glm::vec3 v)  { return {v.x, v.y, v.z}; }
[[maybe_unused]] static PxQuat  toPxQ(glm::quat q) { return PxQuat(q.x, q.y, q.z, q.w); }

// Create an entity with a mesh and sync it to a PhysX actor
// shapeHint: 0=box, 1=sphere, 2=capsule
static Entity* makePhysEntity(Engine* engine, const std::string& name,
                               int shapeHint, glm::vec3 scale,
                               PxRigidActor* actor, bool isDynamic,
                               std::vector<Entity*>& outEntities)
{
    auto* entity = engine->CreateEntity(name);

    auto* t = entity->AddComponent<TransformComponent>();
    // Initial position from actor pose
    PxTransform pose = actor->getGlobalPose();
    t->SetPosition({pose.p.x, pose.p.y, pose.p.z});
    t->SetScale(scale);

    auto* m = entity->AddComponent<MeshComponent>();
    PrimMesh prim;
    if (shapeHint == 1)      prim = MakeSphere(0.5f);
    else if (shapeHint == 2) prim = MakeCapsule(0.25f, 0.5f);
    else                     prim = MakeBox(0.5f);

    // Populate MeshComponent from PrimMesh (same as FillMeshFromPrim in main.cpp)
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
    m->SetVertices(verts);
    m->SetIndices(prim.indices);
    m->SetTexturePath("__shared_default_albedo__");

    // Register with physics system for pose sync
    engine->GetPhysicsSystem()->RegisterActor(actor, entity, isDynamic);

    outEntities.push_back(entity);
    return entity;
}

// Create a standard camera looking at origin
static void setCamera(Engine* engine, glm::vec3 eye, glm::vec3 target, float fov = 60.f) {
    auto* camEntity = engine->CreateEntity("Camera");
    auto* t = camEntity->AddComponent<TransformComponent>();
    t->SetPosition(eye);
    auto* cam = camEntity->AddComponent<CameraComponent>();
    cam->SetAspectRatio(800.f / 600.f);  // default window ratio
    cam->SetFieldOfView(fov);
    cam->SetClipPlanes(0.1f, 500.f);
    cam->SetTarget(target);
    engine->SetActiveCamera(cam);
}

// ---------------------------------------------------------------------------
// SCENE 1: Hello World — stacks of boxes + sphere
// Adapted from snippethelloworld/SnippetHelloWorld.cpp
// ---------------------------------------------------------------------------
static void createStack(PxPhysics* physics, PxScene* scene, PxMaterial* mat,
                        PxTransform t, PxU32 size, PxReal halfExtent,
                        Engine* engine, std::vector<Entity*>& entities)
{
    // isExclusive=false so the same shape can be shared across all stack bodies.
    PxShape* shape = physics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *mat,
                                          /*isExclusive=*/false);
    for (PxU32 i = 0; i < size; ++i) {
        for (PxU32 j = 0; j < size - i; ++j) {
            PxTransform localTm(PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) * halfExtent);
            PxRigidDynamic* body = physics->createRigidDynamic(t.transform(localTm));
            body->attachShape(*shape);
            PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
            scene->addActor(*body);

            std::string nm = "Stack_" + std::to_string(i) + "_" + std::to_string(j);
            makePhysEntity(engine, nm, 0 /*box*/, glm::vec3(halfExtent * 2.f),
                           body, true, entities);
        }
    }
    shape->release();
}

void LoadPhysicsScene_HelloWorld(Engine* engine) {

    setCamera(engine, {0, 9, 30}, {0, 4, 0});

    PxPhysics*  px  = getPhysics(engine);
    PxScene*    sc  = getScene(engine);
    PxMaterial* mat = getMaterial(engine);
    std::vector<Entity*> entities;

    // Ground plane
    PxRigidStatic* ground = PxCreatePlane(*px, PxPlane(0, 1, 0, 0), *mat);
    sc->addActor(*ground);
    // Ground doesn't need a rendered entity (it's infinite)

    // 5 stacks of increasing size
    for (int i = 0; i < 5; ++i) {
        createStack(px, sc, mat,
                    PxTransform(PxVec3(0, 0, -i * 10.f)), 10, 0.5f,
                    engine, entities);
    }

    // Sphere projectile
    PxRigidDynamic* ball = PxCreateDynamic(*px, PxTransform(PxVec3(0, 40, 100)),
                                            PxSphereGeometry(3.f), *mat, 10.f);
    ball->setLinearVelocity(PxVec3(0, -25, -100));
    sc->addActor(*ball);
    makePhysEntity(engine, "Sphere_Projectile", 1 /*sphere*/, glm::vec3(6.f), ball, true, entities);

    endLoad(engine, entities);
    std::cout << "[PhysX] HelloWorld scene loaded.\n";
}

// ---------------------------------------------------------------------------
// SCENE 2: Joints — chain of boxes connected with spherical joints
// Adapted from snippetjoint/SnippetJoint.cpp
// ---------------------------------------------------------------------------
static PxRigidDynamic* createDynBox(PxPhysics* px, PxScene* sc, PxMaterial* mat,
                                     const PxTransform& t, float halfExt,
                                     Engine* engine, const std::string& name,
                                     std::vector<Entity*>& entities)
{
    PxRigidDynamic* body = PxCreateDynamic(*px, t, PxBoxGeometry(halfExt, halfExt, halfExt), *mat, 1.f);
    body->setAngularDamping(0.5f);
    sc->addActor(*body);
    makePhysEntity(engine, name, 0, glm::vec3(halfExt * 2.f), body, true, entities);
    return body;
}

// Create a chain of N boxes with spherical joints
static void createChain(PxPhysics* px, PxScene* sc, PxMaterial* mat,
                         const PxTransform& t, int numLinks, float halfExt,
                         Engine* engine, const std::string& prefix,
                         std::vector<Entity*>& entities)
{
    PxVec3 offset(halfExt, 0, 0);

    PxRigidActor* prev = nullptr;
    for (int i = 0; i < numLinks; ++i) {
        PxTransform globalTm = PxTransform(PxVec3(PxReal(i) * halfExt * 2.0f, 0, 0)) * t;
        auto* body = createDynBox(px, sc, mat, globalTm, halfExt,
                                   engine, prefix + "_" + std::to_string(i), entities);

        if (prev) {
            // Spherical joint with cone limit
            PxTransform frame0(offset), frame1(PxVec3(-halfExt, 0, 0));
            PxSphericalJoint* joint = PxSphericalJointCreate(*px, prev, frame0, body, frame1);
            joint->setLimitCone(PxJointLimitCone(PxPi / 4.f, PxPi / 4.f, PxSpring(0, 0)));
            joint->setSphericalJointFlag(PxSphericalJointFlag::eLIMIT_ENABLED, true);
        } else {
            // Pin first link with a fixed joint to world (nullptr = world frame)
            PxTransform frame0(PxVec3(-halfExt, 0, 0));
            PxFixedJoint* joint = PxFixedJointCreate(*px, nullptr, globalTm.transform(frame0), body, frame0);
            (void)joint;
        }
        prev = body;
    }
}

void LoadPhysicsScene_Joints(Engine* engine) {
    // PxInitExtensions is needed for joint creation.
    // Call it once here — it's safe to call multiple times (internally ref-counted).
    PxPhysics* px = getPhysics(engine);
    PxInitExtensions(*px, nullptr);


    setCamera(engine, {0, 10, 30}, {0, 0, 0});

    PxScene*    sc  = getScene(engine);
    PxMaterial* mat = getMaterial(engine);
    std::vector<Entity*> entities;

    // Ground
    PxRigidStatic* ground = PxCreatePlane(*px, PxPlane(0, 1, 0, 0), *mat);
    sc->addActor(*ground);

    // 3 chains hanging at different heights
    for (int i = 0; i < 3; ++i) {
        createChain(px, sc, mat,
                    PxTransform(PxVec3(-10.f + i * 10.f, 10.f, 0)),
                    8, 0.5f, engine, "Chain_" + std::to_string(i), entities);
    }

    endLoad(engine, entities);
    std::cout << "[PhysX] Joints scene loaded.\n";
}

// ---------------------------------------------------------------------------
// SCENE 3: CCD — fast-moving sphere through a thin wall of boxes
// Adapted from snippetccd/SnippetCCD.cpp
// Note: The PhysX scene was created without eENABLE_CCD flag, so CCD will be
// approximate for this demo. The ball travels very fast and is still visually
// dramatic even without full CCD pipeline support.
// ---------------------------------------------------------------------------
void LoadPhysicsScene_CCD(Engine* engine) {

    setCamera(engine, {0, 15, 40}, {0, 0, 0});

    PxPhysics*  px  = getPhysics(engine);
    PxScene*    sc  = getScene(engine);
    PxMaterial* mat = getMaterial(engine);
    std::vector<Entity*> entities;

    // Ground
    PxRigidStatic* ground = PxCreatePlane(*px, PxPlane(0, 1, 0, 0), *mat);
    sc->addActor(*ground);

    // Thin wall of small boxes
    const float wallHalf = 0.1f;  // very thin
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            PxTransform t(PxVec3(c * 1.2f - 4.f, r * 1.2f + 0.5f, 0));
            PxRigidDynamic* box = PxCreateDynamic(*px, t, PxBoxGeometry(0.5f, 0.5f, wallHalf), *mat, 1.f);
            sc->addActor(*box);
            std::string nm = "Wall_" + std::to_string(r) + "_" + std::to_string(c);
            makePhysEntity(engine, nm, 0, glm::vec3(1.f, 1.f, wallHalf * 2.f), box, true, entities);
        }
    }

    // Fast sphere (CCD flag on actor — scene also needs eENABLE_CCD for full CCD,
    // but the visual effect is still dramatic with a very fast ball)
    PxRigidDynamic* ball = PxCreateDynamic(*px, PxTransform(PxVec3(0, 5, -30)),
                                            PxSphereGeometry(0.5f), *mat, 1.f);
    ball->setLinearVelocity(PxVec3(0, 0, 100.f));   // very fast
    ball->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
    sc->addActor(*ball);
    makePhysEntity(engine, "CCD_Ball", 1, glm::vec3(1.f), ball, true, entities);

    endLoad(engine, entities);
    std::cout << "[PhysX] CCD scene loaded.\n";
}

// ---------------------------------------------------------------------------
// SCENE 4: Gyroscopic — spinning L-shaped body demonstrating Dzhanibekov effect
// Adapted from snippetgyroscopic/SnippetGyroscopic.cpp
// ---------------------------------------------------------------------------
void LoadPhysicsScene_Gyroscopic(Engine* engine) {

    setCamera(engine, {0, 2, 15}, {0, 2, 0});

    PxPhysics*  px  = getPhysics(engine);
    PxScene*    sc  = getScene(engine);
    PxMaterial* mat = getMaterial(engine);
    std::vector<Entity*> entities;

    // No gravity for this one — pure rotational dynamics
    sc->setGravity(PxVec3(0, 0, 0));

    // L-shaped object: main rod + crossbar as two shapes on one actor
    PxRigidDynamic* actor = px->createRigidDynamic(PxTransform(PxVec3(0, 2, 0)));

    PxShape* rod = px->createShape(PxBoxGeometry(0.05f, 0.5f, 0.05f), *mat);
    PxShape* bar = px->createShape(PxBoxGeometry(0.1f, 0.05f, 0.05f), *mat);
    rod->setLocalPose(PxTransform(PxVec3(0, 0, 0)));
    bar->setLocalPose(PxTransform(PxVec3(0, 0.5f, 0)));
    actor->attachShape(*rod);
    actor->attachShape(*bar);
    rod->release();
    bar->release();
    PxRigidBodyExt::updateMassAndInertia(*actor, 1.f);
    actor->setAngularDamping(0.f);
    actor->setAngularVelocity(PxVec3(0.01f, 15.f, 0.01f));   // near intermediate axis → flip
    actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);
    sc->addActor(*actor);

    // Use a box mesh as a visual approximation of the L-shape
    makePhysEntity(engine, "Gyro_Body", 0, glm::vec3(0.2f, 1.f, 0.2f), actor, true, entities);

    endLoad(engine, entities);
    std::cout << "[PhysX] Gyroscopic scene loaded.\n";
}

// ---------------------------------------------------------------------------
// SCENE 5: Triggers — boxes falling into trigger zones
// Adapted from snippettriggers/SnippetTriggers.cpp
// (We use REAL_TRIGGERS: PxShapeFlag::eTRIGGER_SHAPE — no rendering for trigger volume)
// ---------------------------------------------------------------------------
void LoadPhysicsScene_Triggers(Engine* engine) {

    setCamera(engine, {0, 15, 30}, {0, 5, 0});

    PxPhysics*  px  = getPhysics(engine);
    PxScene*    sc  = getScene(engine);
    PxMaterial* mat = getMaterial(engine);
    std::vector<Entity*> entities;

    // Ground
    PxRigidStatic* ground = PxCreatePlane(*px, PxPlane(0, 1, 0, 0), *mat);
    sc->addActor(*ground);

    // Trigger volumes (static, trigger shape — no visual mesh, just physics)
    auto addTrigger = [&](PxVec3 pos, PxVec3 halfExt) {
        PxRigidStatic* trig = px->createRigidStatic(PxTransform(pos));
        PxShape* sh = px->createShape(PxBoxGeometry(halfExt.x, halfExt.y, halfExt.z), *mat, true);
        sh->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
        sh->setFlag(PxShapeFlag::eTRIGGER_SHAPE, true);
        trig->attachShape(*sh);
        sh->release();
        sc->addActor(*trig);
    };

    addTrigger(PxVec3(-5, 3, 0), PxVec3(2, 2, 2));
    addTrigger(PxVec3( 5, 3, 0), PxVec3(2, 2, 2));
    addTrigger(PxVec3( 0, 8, 0), PxVec3(3, 1, 3));

    // Falling dynamic boxes that will pass through trigger zones
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> xDist(-6.f, 6.f);
    std::uniform_real_distribution<float> yDist(12.f, 25.f);

    for (int i = 0; i < 20; ++i) {
        PxVec3 pos(xDist(rng), yDist(rng), xDist(rng) * 0.3f);
        PxRigidDynamic* box = PxCreateDynamic(*px, PxTransform(pos),
                                               PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat, 1.f);
        sc->addActor(*box);
        makePhysEntity(engine, "Trig_Box_" + std::to_string(i), 0, glm::vec3(1.f), box, true, entities);
    }

    endLoad(engine, entities);
    std::cout << "[PhysX] Triggers scene loaded.\n";
}
