/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "physics_system.h"
#include "entity.h"
#include "transform_component.h"

#include <PxPhysicsAPI.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <algorithm>

using namespace physx;

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------
struct PhysicsImpl {
    PxDefaultAllocator     allocator;
    PxDefaultErrorCallback errorCallback;
};

static PhysicsImpl*            impl (void* p) { return static_cast<PhysicsImpl*>(p); }
static PxFoundation*           fnd  (void* p) { return static_cast<PxFoundation*>(p); }
static PxPhysics*              phys (void* p) { return static_cast<PxPhysics*>(p); }
static PxScene*                scn  (void* p) { return static_cast<PxScene*>(p); }
static PxDefaultCpuDispatcher* disp (void* p) { return static_cast<PxDefaultCpuDispatcher*>(p); }
static PxMaterial*             mat  (void* p) { return static_cast<PxMaterial*>(p); }

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------
static PxVec3  toPx(glm::vec3 v)  { return {v.x, v.y, v.z}; }
static glm::vec3 fromPx(PxVec3 v) { return {v.x, v.y, v.z}; }
// PxQuat is {x,y,z,w}; glm::quat constructor is (w,x,y,z)
static glm::quat fromPxQ(PxQuat q) { return glm::quat(q.w, q.x, q.y, q.z); }
static PxQuat    toPxQ(glm::quat q){ return PxQuat(q.x, q.y, q.z, q.w); }

// ---------------------------------------------------------------------------
// ConcreteRigidBody — wraps a PxRigidActor / PxRigidDynamic
// ---------------------------------------------------------------------------
class ConcreteRigidBody final : public RigidBody {
  public:
    void*  pxActor  = nullptr;  // PxRigidActor*
    void*  pxDynamic= nullptr;  // PxRigidDynamic* (null if static/kinematic)
    bool   kinematic= false;

    void SetPosition(const glm::vec3& p) override {
        auto* actor = static_cast<PxRigidActor*>(pxActor);
        if (!actor) return;
        PxTransform t = actor->getGlobalPose();
        t.p = toPx(p);
        actor->setGlobalPose(t);
    }
    void SetRotation(const glm::quat& q) override {
        auto* actor = static_cast<PxRigidActor*>(pxActor);
        if (!actor) return;
        PxTransform t = actor->getGlobalPose();
        t.q = toPxQ(q);
        actor->setGlobalPose(t);
    }
    void SetScale(const glm::vec3& /*scale*/) override {
        // PhysX doesn't support runtime scale changes; no-op
    }
    void SetMass(float mass) override {
        auto* body = static_cast<PxRigidDynamic*>(pxDynamic);
        if (body) body->setMass(mass);
    }
    void SetRestitution(float restitution) override {
        // Restitution is stored on the material; for simplicity apply it
        // to the actor's shapes directly.
        auto* actor = static_cast<PxRigidActor*>(pxActor);
        if (!actor) return;
        PxU32 n = actor->getNbShapes();
        std::vector<PxShape*> shapes(n);
        actor->getShapes(shapes.data(), n);
        for (auto* s : shapes) {
            PxMaterial* mats[1];
            if (s->getNbMaterials() > 0) {
                s->getMaterials(mats, 1);
                mats[0]->setRestitution(restitution);
            }
        }
    }
    void SetFriction(float friction) override {
        auto* actor = static_cast<PxRigidActor*>(pxActor);
        if (!actor) return;
        PxU32 n = actor->getNbShapes();
        std::vector<PxShape*> shapes(n);
        actor->getShapes(shapes.data(), n);
        for (auto* s : shapes) {
            PxMaterial* mats[1];
            if (s->getNbMaterials() > 0) {
                s->getMaterials(mats, 1);
                mats[0]->setStaticFriction(friction);
                mats[0]->setDynamicFriction(friction);
            }
        }
    }
    void ApplyForce(const glm::vec3& force, const glm::vec3& /*localPos*/) override {
        auto* body = static_cast<PxRigidDynamic*>(pxDynamic);
        if (body) body->addForce(toPx(force));
    }
    void ApplyImpulse(const glm::vec3& impulse, const glm::vec3& localPos) override {
        auto* body = static_cast<PxRigidDynamic*>(pxDynamic);
        if (!body) return;
        PxVec3 worldPoint = body->getGlobalPose().p + toPx(localPos);
        PxRigidBodyExt::addForceAtPos(*body, toPx(impulse), worldPoint, PxForceMode::eIMPULSE);
    }
    void SetLinearVelocity(const glm::vec3& v) override {
        auto* body = static_cast<PxRigidDynamic*>(pxDynamic);
        if (body) body->setLinearVelocity(toPx(v));
    }
    void SetAngularVelocity(const glm::vec3& v) override {
        auto* body = static_cast<PxRigidDynamic*>(pxDynamic);
        if (body) body->setAngularVelocity(toPx(v));
    }
    [[nodiscard]] glm::vec3 GetPosition() const override {
        auto* actor = static_cast<PxRigidActor*>(pxActor);
        if (!actor) return {};
        return fromPx(actor->getGlobalPose().p);
    }
    [[nodiscard]] glm::quat GetRotation() const override {
        auto* actor = static_cast<PxRigidActor*>(pxActor);
        if (!actor) return glm::quat(1,0,0,0);
        return fromPxQ(actor->getGlobalPose().q);
    }
    [[nodiscard]] glm::vec3 GetLinearVelocity() const override {
        auto* body = static_cast<PxRigidDynamic*>(pxDynamic);
        if (!body) return {};
        return fromPx(body->getLinearVelocity());
    }
    [[nodiscard]] glm::vec3 GetAngularVelocity() const override {
        auto* body = static_cast<PxRigidDynamic*>(pxDynamic);
        if (!body) return {};
        return fromPx(body->getAngularVelocity());
    }
    void SetKinematic(bool k) override {
        kinematic = k;
        auto* body = static_cast<PxRigidDynamic*>(pxDynamic);
        if (body) body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, k);
    }
    [[nodiscard]] bool IsKinematic() const override { return kinematic; }
};

// ---------------------------------------------------------------------------
// PhysicsSystem::Initialize
// ---------------------------------------------------------------------------
bool PhysicsSystem::Initialize() {
    auto* pimpl = new PhysicsImpl();
    m_pimpl = pimpl;

    m_foundation = PxCreateFoundation(PX_PHYSICS_VERSION, pimpl->allocator, pimpl->errorCallback);
    if (!m_foundation) {
        std::cerr << "[PhysX] PxCreateFoundation failed\n";
        delete pimpl;
        m_pimpl = nullptr;
        return false;
    }

    PxTolerancesScale scale;
    m_physics = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd(m_foundation), scale, false, nullptr);
    if (!m_physics) {
        std::cerr << "[PhysX] PxCreatePhysics failed\n";
        return false;
    }

    m_dispatcher = PxDefaultCpuDispatcherCreate(2);

    PxSceneDesc desc(phys(m_physics)->getTolerancesScale());
    desc.gravity       = toPx(m_gravity);
    desc.cpuDispatcher = disp(m_dispatcher);
    desc.filterShader  = PxDefaultSimulationFilterShader;
    m_scene = phys(m_physics)->createScene(desc);
    if (!m_scene) {
        std::cerr << "[PhysX] createScene failed\n";
        return false;
    }

    m_material = phys(m_physics)->createMaterial(0.5f, 0.5f, 0.4f);

    std::cout << "[PhysX] Initialized (gravity = " << m_gravity.y << " m/s^2)\n";
    m_initialized = true;
    return true;
}

// ---------------------------------------------------------------------------
// PhysicsSystem::~PhysicsSystem
// ---------------------------------------------------------------------------
PhysicsSystem::~PhysicsSystem() {
    // Release rigid bodies and clear entries before releasing the scene
    {
        std::lock_guard<std::mutex> lk(m_bodiesMutex);
        for (auto& b : m_bodies) {
            auto* actor = static_cast<PxRigidActor*>(b.pxActor);
            if (actor && m_scene) {
                scn(m_scene)->removeActor(*actor);
                actor->release();
            }
        }
        m_bodies.clear();
        m_rigidBodies.clear();
    }

    if (m_scene)      { scn(m_scene)->release();     m_scene      = nullptr; }
    if (m_dispatcher) { disp(m_dispatcher)->release(); m_dispatcher = nullptr; }
    if (m_physics)    { phys(m_physics)->release();  m_physics    = nullptr; }
    if (m_foundation) { fnd(m_foundation)->release(); m_foundation = nullptr; }
    delete static_cast<PhysicsImpl*>(m_pimpl);
    m_pimpl = nullptr;
}

// ---------------------------------------------------------------------------
// PhysicsSystem::CreateRigidBody
// ---------------------------------------------------------------------------
RigidBody* PhysicsSystem::CreateRigidBody(Entity* entity, CollisionShape shape, float mass) {
    if (!m_initialized || !m_physics || !m_scene) return nullptr;

    // Determine starting position from entity transform
    glm::vec3 pos(0.f);
    glm::quat rot(1.f, 0.f, 0.f, 0.f);
    glm::vec3 sc(1.f);
    if (entity) {
        if (auto* t = entity->GetComponent<TransformComponent>()) {
            pos = t->GetPosition();
            sc  = t->GetScale();
            // TransformComponent stores Euler; build a quat from it
            glm::vec3 euler = t->GetRotation();
            rot = glm::quat(euler);
        }
    }

    PxTransform pxPose(toPx(pos), toPxQ(rot));

    auto* crb = new ConcreteRigidBody();

    bool isStatic = (mass <= 0.0f);

    if (isStatic) {
        // Static actor — used for the ground plane / kinematic barriers
        PxRigidStatic* staticActor = nullptr;

        if (shape == CollisionShape::Box) {
            float hx = std::max(sc.x * 0.5f, 0.01f);
            float hy = std::max(sc.y * 0.5f, 0.01f);
            float hz = std::max(sc.z * 0.5f, 0.01f);
            PxShape* sh = phys(m_physics)->createShape(PxBoxGeometry(hx, hy, hz), *mat(m_material));
            staticActor = phys(m_physics)->createRigidStatic(pxPose);
            staticActor->attachShape(*sh);
            sh->release();
        } else if (shape == CollisionShape::Sphere) {
            float r = std::max(sc.x * 0.5f, 0.01f);
            PxShape* sh = phys(m_physics)->createShape(PxSphereGeometry(r), *mat(m_material));
            staticActor = phys(m_physics)->createRigidStatic(pxPose);
            staticActor->attachShape(*sh);
            sh->release();
        } else {
            // Default: flat box
            PxShape* sh = phys(m_physics)->createShape(PxBoxGeometry(1.f, 0.05f, 1.f), *mat(m_material));
            staticActor = phys(m_physics)->createRigidStatic(pxPose);
            staticActor->attachShape(*sh);
            sh->release();
        }

        if (!staticActor) { delete crb; return nullptr; }

        scn(m_scene)->addActor(*staticActor);
        crb->pxActor   = staticActor;
        crb->pxDynamic = nullptr;
        crb->kinematic = false;

    } else {
        // Dynamic actor
        PxRigidDynamic* body = nullptr;

        if (shape == CollisionShape::Sphere) {
            float r = std::max(sc.x * 0.5f, 0.01f);
            body = PxCreateDynamic(*phys(m_physics), pxPose, PxSphereGeometry(r), *mat(m_material), mass);
        } else if (shape == CollisionShape::Capsule) {
            float r  = std::max(sc.x * 0.25f, 0.01f);
            float hh = std::max(sc.y * 0.25f, 0.01f);
            PxTransform yUp(PxQuat(PxHalfPi, PxVec3(0, 0, 1)));
            body = PxCreateDynamic(*phys(m_physics), pxPose, PxCapsuleGeometry(r, hh), *mat(m_material), mass, yUp);
        } else {
            // Box (and Mesh falls back to Box)
            float hx = std::max(sc.x * 0.5f, 0.01f);
            float hy = std::max(sc.y * 0.5f, 0.01f);
            float hz = std::max(sc.z * 0.5f, 0.01f);
            body = PxCreateDynamic(*phys(m_physics), pxPose, PxBoxGeometry(hx, hy, hz), *mat(m_material), mass);
        }

        if (!body) { delete crb; return nullptr; }
        body->setAngularDamping(0.3f);
        scn(m_scene)->addActor(*body);
        crb->pxActor   = body;
        crb->pxDynamic = body;
        crb->kinematic = false;
    }

    // Store ownership
    std::unique_ptr<ConcreteRigidBody> crbOwned(crb);
    RigidBody* rawPtr = crb;

    {
        std::lock_guard<std::mutex> lk(m_bodiesMutex);
        BodyEntry entry;
        entry.pxActor   = crb->pxActor;
        entry.isDynamic = (crb->pxDynamic != nullptr);
        entry.entity    = entity;
        entry.shape     = shape;
        entry.rigidBody = crb;
        m_bodies.push_back(entry);
        m_rigidBodies.push_back(std::move(crbOwned));
    }

    return rawPtr;
}

// ---------------------------------------------------------------------------
// PhysicsSystem::DestroyRigidBody
// ---------------------------------------------------------------------------
bool PhysicsSystem::DestroyRigidBody(RigidBody* rigidBody) {
    if (!rigidBody || !m_initialized) return false;

    std::lock_guard<std::mutex> lk(m_bodiesMutex);

    // Find the BodyEntry for this rigid body
    auto entryIt = std::find_if(m_bodies.begin(), m_bodies.end(),
        [rigidBody](const BodyEntry& b) { return b.rigidBody == rigidBody; });
    if (entryIt == m_bodies.end()) return false;

    auto* actor = static_cast<PxRigidActor*>(entryIt->pxActor);
    if (actor && m_scene) {
        scn(m_scene)->removeActor(*actor);
        actor->release();
    }
    m_bodies.erase(entryIt);

    // Remove from rigidBodies ownership list
    auto rbIt = std::find_if(m_rigidBodies.begin(), m_rigidBodies.end(),
        [rigidBody](const std::unique_ptr<RigidBody>& p) { return p.get() == rigidBody; });
    if (rbIt != m_rigidBodies.end()) {
        m_rigidBodies.erase(rbIt);
    }

    return true;
}

// ---------------------------------------------------------------------------
// PhysicsSystem::Step  — single fixed-timestep step (UpdatePublisher path)
// ---------------------------------------------------------------------------
void PhysicsSystem::Step(float fixedDt) {
    if (!m_initialized) return;

    // Drain pending body creations (from background / main threads)
    {
        std::vector<PendingCreation> pending;
        {
            std::lock_guard<std::mutex> lk(m_pendingMutex);
            pending.swap(m_pendingCreations);
        }
        for (auto& p : pending) {
            RigidBody* rb = CreateRigidBody(p.entity, p.shape, p.mass);
            if (rb) {
                rb->SetRestitution(p.restitution);
                rb->SetFriction(p.friction);
                if (p.kinematic) rb->SetKinematic(true);
            }
        }
    }

    // Single simulation step — accumulator is owned by UpdatePublisher
    scn(m_scene)->simulate(fixedDt);
    scn(m_scene)->fetchResults(true);

    // Sync actor poses back to entity transforms
    {
        std::lock_guard<std::mutex> lk(m_bodiesMutex);
        for (auto& b : m_bodies) {
            if (!b.isDynamic || !b.entity || !b.pxActor) continue;
            auto* actor = static_cast<PxRigidActor*>(b.pxActor);
            PxTransform pose = actor->getGlobalPose();
            auto* transform = b.entity->GetComponent<TransformComponent>();
            if (transform) {
                transform->SetPosition(fromPx(pose.p));
                transform->SetRotation(glm::eulerAngles(fromPxQ(pose.q)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PhysicsSystem::Update  — legacy path with internal fixed-step accumulator
// ---------------------------------------------------------------------------
void PhysicsSystem::Update(std::chrono::milliseconds deltaTime) {
    if (!m_initialized) return;

    float dt = deltaTime.count() / 1000.f;  // ms -> seconds
    m_accumulator += dt;
    if (m_accumulator > FIXED_STEP * 4.f) m_accumulator = FIXED_STEP * 4.f;

    while (m_accumulator >= FIXED_STEP) {
        Step(FIXED_STEP);
        m_accumulator -= FIXED_STEP;
    }
}

// ---------------------------------------------------------------------------
// PhysicsSystem::SetGravity / GetGravity
// ---------------------------------------------------------------------------
void PhysicsSystem::SetGravity(const glm::vec3& gravity) {
    m_gravity = gravity;
    if (m_scene) {
        scn(m_scene)->setGravity(toPx(gravity));
    }
}

glm::vec3 PhysicsSystem::GetGravity() const {
    return m_gravity;
}

// ---------------------------------------------------------------------------
// PhysicsSystem::Raycast
// ---------------------------------------------------------------------------
bool PhysicsSystem::Raycast(const glm::vec3& origin,
                             const glm::vec3& direction,
                             float maxDistance,
                             glm::vec3* hitPosition,
                             glm::vec3* hitNormal,
                             Entity** hitEntity) const {
    if (!m_initialized || !m_scene) return false;

    PxRaycastBuffer hit;
    bool hadHit = scn(m_scene)->raycast(toPx(origin), toPx(glm::normalize(direction)), maxDistance, hit);
    if (!hadHit || !hit.hasBlock) return false;

    if (hitPosition) *hitPosition = fromPx(hit.block.position);
    if (hitNormal)   *hitNormal   = fromPx(hit.block.normal);
    if (hitEntity) {
        *hitEntity = nullptr;
        // Try to find the entity associated with this actor
        std::lock_guard<std::mutex> lk(m_bodiesMutex);
        for (const auto& b : m_bodies) {
            if (b.pxActor == hit.block.actor) {
                *hitEntity = b.entity;
                break;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// PhysicsSystem::ClearActors
// ---------------------------------------------------------------------------
void PhysicsSystem::ClearActors() {
    if (!m_initialized) return;

    // Drain pending creations so they don't fire on the next Update after the scene changed.
    {
        std::lock_guard<std::mutex> lk(m_pendingMutex);
        m_pendingCreations.clear();
    }

    // Release all registered actors and their RigidBody wrappers.
    {
        std::lock_guard<std::mutex> lk(m_bodiesMutex);
        for (auto& b : m_bodies) {
            auto* actor = static_cast<PxRigidActor*>(b.pxActor);
            if (actor && m_scene) {
                scn(m_scene)->removeActor(*actor);
                actor->release();
            }
        }
        m_bodies.clear();
        m_rigidBodies.clear();
    }

    // Tear down and recreate the PxScene so any actors added directly (e.g. ground planes
    // created with PxCreatePlane that were never registered via RegisterActor) are also gone.
    if (m_scene) {
        scn(m_scene)->release();
        m_scene = nullptr;
    }

    PxSceneDesc desc(phys(m_physics)->getTolerancesScale());
    desc.gravity       = toPx(m_gravity);
    desc.cpuDispatcher = disp(m_dispatcher);
    desc.filterShader  = PxDefaultSimulationFilterShader;
    m_scene = phys(m_physics)->createScene(desc);
    if (!m_scene) {
        std::cerr << "[PhysX] ClearActors: failed to recreate scene\n";
        m_initialized = false;
    }

    m_accumulator = 0.0f;
    std::cout << "[PhysX] Scene cleared.\n";
}

// ---------------------------------------------------------------------------
// PhysicsSystem::EnqueueRigidBodyCreation
// ---------------------------------------------------------------------------
void PhysicsSystem::EnqueueRigidBodyCreation(Entity* entity,
                                              CollisionShape shape,
                                              float mass,
                                              bool kinematic,
                                              float restitution,
                                              float friction) {
    std::lock_guard<std::mutex> lk(m_pendingMutex);
    m_pendingCreations.push_back({entity, shape, mass, kinematic, restitution, friction});
}

// ---------------------------------------------------------------------------
// PhysicsSystem::RegisterActor
// ---------------------------------------------------------------------------
void PhysicsSystem::RegisterActor(void* pxActor, Entity* entity, bool isDynamic) {
    if (!pxActor) return;
    std::lock_guard<std::mutex> lk(m_bodiesMutex);

    // Create a ConcreteRigidBody that wraps the external actor
    auto* crb = new ConcreteRigidBody();
    crb->pxActor   = pxActor;
    crb->pxDynamic = isDynamic ? pxActor : nullptr;
    crb->kinematic = false;

    BodyEntry entry;
    entry.pxActor   = pxActor;
    entry.isDynamic = isDynamic;
    entry.entity    = entity;
    entry.shape     = CollisionShape::Box;  // shape hint, unused for sync
    entry.rigidBody = crb;
    m_bodies.push_back(entry);
    m_rigidBodies.push_back(std::unique_ptr<ConcreteRigidBody>(crb));
}
