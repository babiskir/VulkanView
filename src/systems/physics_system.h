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
#pragma once

#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

class Entity;

/**
 * @brief Enum for different collision shapes.
 */
enum class CollisionShape {
  Box,
  Sphere,
  Capsule,
  Mesh
};

/**
 * @brief Class representing a rigid body for physics simulation.
 */
class RigidBody {
  public:
    RigidBody() = default;
    virtual ~RigidBody() = default;

    virtual void SetPosition(const glm::vec3& position) = 0;
    virtual void SetRotation(const glm::quat& rotation) = 0;
    virtual void SetScale(const glm::vec3& scale) = 0;
    virtual void SetMass(float mass) = 0;
    virtual void SetRestitution(float restitution) = 0;
    virtual void SetFriction(float friction) = 0;
    virtual void ApplyForce(const glm::vec3& force, const glm::vec3& localPosition = glm::vec3(0.0f)) = 0;
    virtual void ApplyImpulse(const glm::vec3& impulse, const glm::vec3& localPosition = glm::vec3(0.0f)) = 0;
    virtual void SetLinearVelocity(const glm::vec3& velocity) = 0;
    virtual void SetAngularVelocity(const glm::vec3& velocity) = 0;
    [[nodiscard]] virtual glm::vec3 GetPosition() const = 0;
    [[nodiscard]] virtual glm::quat GetRotation() const = 0;
    [[nodiscard]] virtual glm::vec3 GetLinearVelocity() const = 0;
    [[nodiscard]] virtual glm::vec3 GetAngularVelocity() const = 0;
    virtual void SetKinematic(bool kinematic) = 0;
    [[nodiscard]] virtual bool IsKinematic() const = 0;
};

/**
 * @brief Class for managing physics simulation using PhysX 5.
 */
class PhysicsSystem {
  public:
    PhysicsSystem() {
      if (!Initialize()) {
        throw std::runtime_error("PhysicsSystem: initialization failed");
      }
    }

    ~PhysicsSystem();

    /**
     * @brief Update the physics system (legacy — owns internal fixed-step accumulator).
     * @param deltaTime The time elapsed since the last update (milliseconds).
     */
    void Update(std::chrono::milliseconds deltaTime);

    /**
     * @brief Execute a single fixed-timestep simulation step.
     *
     * This is the preferred entry point when UpdatePublisher owns the fixed-step
     * accumulator.  Drains pending body creations, simulates exactly @p fixedDt
     * seconds, then syncs actor poses back to entity transforms.
     *
     * @param fixedDt Fixed timestep in seconds (e.g. 1/60).
     */
    void Step(float fixedDt);

    /**
     * @brief Create a rigid body.
     * @param entity The entity to attach the rigid body to.
     * @param shape The collision shape.
     * @param mass The mass (0 = static/kinematic).
     * @return Pointer to the created rigid body, or nullptr if creation failed.
     */
    RigidBody* CreateRigidBody(Entity* entity, CollisionShape shape, float mass);

    /**
     * @brief Destroy a rigid body.
     * @param rigidBody The rigid body to destroy.
     * @return True if destruction was successful.
     */
    bool DestroyRigidBody(RigidBody* rigidBody);

    /**
     * @brief Set the gravity of the physics world.
     */
    void SetGravity(const glm::vec3& gravity);

    /**
     * @brief Get the gravity of the physics world.
     */
    [[nodiscard]] glm::vec3 GetGravity() const;

    /**
     * @brief Perform a raycast.
     */
    bool Raycast(const glm::vec3& origin,
                 const glm::vec3& direction,
                 float maxDistance,
                 glm::vec3* hitPosition,
                 glm::vec3* hitNormal,
                 Entity** hitEntity) const;

    /**
     * @brief Set current camera position (kept for backward compatibility with engine.cpp).
     */
    void SetCameraPosition(const glm::vec3& cameraPosition) {
      m_cameraPosition = cameraPosition;
    }

    /**
     * @brief Thread-safe enqueue for rigid body creation from any thread.
     */
    void EnqueueRigidBodyCreation(Entity* entity,
                                  CollisionShape shape,
                                  float mass,
                                  bool kinematic,
                                  float restitution,
                                  float friction);

    // Raw PhysX objects — for advanced snippet scenes that need direct PxPhysics/PxScene access.
    // Returns void* to keep PhysX out of this header; cast in the caller's .cpp (which includes PxPhysicsAPI.h).
    void* GetRawPhysics()  const { return m_physics; }
    void* GetRawScene()    const { return m_scene; }
    void* GetRawMaterial() const { return m_material; }

    // Register an externally-created PxRigidActor* (cast to void*) with an entity for per-frame pose sync.
    // Call this after creating actors directly via PxPhysics to have them appear in the renderer.
    void RegisterActor(void* pxActor, Entity* entity, bool isDynamic);

    // Remove all registered actors from the PhysX scene and clear m_bodies / m_rigidBodies.
    // Actors that were added directly to the PxScene (e.g. ground planes not registered via
    // RegisterActor) are also flushed by releasing the current scene and creating a fresh one.
    void ClearActors();

  private:
    bool Initialize();

    // PhysX pimpl — no PhysX headers in .h
    void* m_pimpl      = nullptr;  // PhysicsImpl (allocator + error callback)
    void* m_foundation = nullptr;
    void* m_physics    = nullptr;
    void* m_scene      = nullptr;
    void* m_dispatcher = nullptr;
    void* m_material   = nullptr;

    // Body entry linking a PhysX actor to the VulkanView entity + RigidBody
    struct BodyEntry {
      void*          pxActor;   // PxRigidActor* cast to void*
      bool           isDynamic;
      Entity*        entity;
      CollisionShape shape;
      RigidBody*     rigidBody; // owned in m_rigidBodies
    };
    std::vector<BodyEntry>  m_bodies;
    mutable std::mutex      m_bodiesMutex;

    // RigidBody ownership
    std::vector<std::unique_ptr<RigidBody>> m_rigidBodies;

    // Pending creations queued from background threads
    struct PendingCreation {
      Entity*        entity;
      CollisionShape shape;
      float          mass;
      bool           kinematic;
      float          restitution;
      float          friction;
    };
    std::vector<PendingCreation> m_pendingCreations;
    std::mutex                   m_pendingMutex;

    float m_accumulator = 0.0f;
    static constexpr float FIXED_STEP = 1.0f / 60.0f;

    glm::vec3 m_gravity{0.f, -9.81f, 0.f};
    glm::vec3 m_cameraPosition{0.f};
    bool      m_initialized = false;
};
