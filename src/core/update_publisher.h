#pragma once

#include <functional>
#include <vector>

/**
 * UpdatePublisher — Unity-style three-phase game loop dispatcher.
 *
 * Phases (per tick):
 *   FixedUpdate  – fixed timestep (default 50 Hz / 0.02 s); runs physics-coupled logic.
 *   Update       – once per frame; runs game logic.
 *   LateUpdate   – once per frame after Update; runs camera / dependent transforms.
 *
 * Subscriptions are split into "core" (engine lifetime) and "scene" (cleared on
 * ClearScene) so scenes can register their own physics callbacks without leaking.
 *
 * Usage:
 *   engine->SubscribeSceneFixedUpdate([scene](float dt){ scene->ApplyBuoyancy(dt); });
 *   engine->SubscribeSceneUpdate([scene](float dt){ scene->Tick(dt); });
 */
class UpdatePublisher {
public:
    using Callback = std::function<void(float)>;

    explicit UpdatePublisher(float fixedTimestep = 0.02f);  // 50 Hz

    // ------------------------------------------------------------------ core
    // Core callbacks persist for the engine lifetime.
    void SubscribeCoreFixedUpdate(Callback cb);
    void SubscribeCoreUpdate     (Callback cb);
    void SubscribeCoreLateUpdate (Callback cb);

    // ----------------------------------------------------------------- scene
    // Scene callbacks are registered per-scene and cleared by ClearSceneCallbacks().
    void SubscribeSceneFixedUpdate(Callback cb);
    void SubscribeSceneUpdate     (Callback cb);
    void SubscribeSceneLateUpdate (Callback cb);

    /** Clear all scene-level subscriptions.  Call from Engine::ClearScene(). */
    void ClearSceneCallbacks();

    /**
     * Advance the publisher by an externally-measured dt (seconds).
     * Runs FixedUpdate N times to consume the accumulator, then Update + LateUpdate once.
     */
    void TickWithDt(float dt);

    float GetFixedTimestep() const { return m_fixedTimestep; }
    void  SetFixedTimestep(float ts) { m_fixedTimestep = ts; }

private:
    float m_fixedTimestep;
    float m_accumulator = 0.f;

    std::vector<Callback> m_coreFixed;
    std::vector<Callback> m_coreUpdate;
    std::vector<Callback> m_coreLate;

    std::vector<Callback> m_sceneFixed;
    std::vector<Callback> m_sceneUpdate;
    std::vector<Callback> m_sceneLate;
};
