#include "update_publisher.h"
#include <algorithm>

UpdatePublisher::UpdatePublisher(float fixedTimestep)
    : m_fixedTimestep(fixedTimestep)
    , m_accumulator(0.f)
{}

void UpdatePublisher::SubscribeCoreFixedUpdate(Callback cb) { m_coreFixed .push_back(std::move(cb)); }
void UpdatePublisher::SubscribeCoreUpdate     (Callback cb) { m_coreUpdate.push_back(std::move(cb)); }
void UpdatePublisher::SubscribeCoreLateUpdate (Callback cb) { m_coreLate  .push_back(std::move(cb)); }

void UpdatePublisher::SubscribeSceneFixedUpdate(Callback cb) { m_sceneFixed .push_back(std::move(cb)); }
void UpdatePublisher::SubscribeSceneUpdate     (Callback cb) { m_sceneUpdate.push_back(std::move(cb)); }
void UpdatePublisher::SubscribeSceneLateUpdate (Callback cb) { m_sceneLate  .push_back(std::move(cb)); }

void UpdatePublisher::ClearSceneCallbacks() {
    m_sceneFixed .clear();
    m_sceneUpdate.clear();
    m_sceneLate  .clear();
}

void UpdatePublisher::TickWithDt(float dt) {
    // Clamp to prevent spiral-of-death when the frame takes too long
    dt = std::min(dt, 0.25f);
    m_accumulator += dt;

    // FixedUpdate — may fire zero or multiple times per frame
    while (m_accumulator >= m_fixedTimestep) {
        for (auto& cb : m_coreFixed)  cb(m_fixedTimestep);
        for (auto& cb : m_sceneFixed) cb(m_fixedTimestep);
        m_accumulator -= m_fixedTimestep;
    }

    // Update
    for (auto& cb : m_coreUpdate)  cb(dt);
    for (auto& cb : m_sceneUpdate) cb(dt);

    // LateUpdate
    for (auto& cb : m_coreLate)  cb(dt);
    for (auto& cb : m_sceneLate) cb(dt);
}
