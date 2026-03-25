#pragma once

#include "scene_ui.h"

#include <atomic>
#include <list>
#include <memory>
#include <string>
#include <vector>

class Engine;

// ---------------------------------------------------------------------------
// LoadedModel — tracks one GLTF file loaded into the model manager.
// ---------------------------------------------------------------------------
struct LoadedModel {
    std::string  path;
    std::string  displayName;   ///< Filename without directory
    bool         visible = true;
    std::atomic<bool> loading{false};

    // Non-copyable (atomic<bool> is not copyable), but movable.
    LoadedModel() = default;
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&) = default;
    LoadedModel& operator=(LoadedModel&&) = default;
};

// ---------------------------------------------------------------------------
// ModelManagerScene
//
// A runtime model workspace.  Scenes loaded here are additive; any number of
// GLTF files can be loaded side-by-side.  The UI is driven by SceneBrowserPanel
// + InspectorPanel and a compact "Load Model" top bar.
//
// Lifetime: managed via shared_ptr captured inside the SetSceneUI lambda so
// it lives as long as the engine does.
// ---------------------------------------------------------------------------
class ModelManagerScene {
public:
    explicit ModelManagerScene(Engine* engine);

    /// Called once per ImGui frame (from the SetSceneUI callback).
    void DrawUI(Engine* engine);

private:
    SceneBrowserPanel         browser_;
    InspectorPanel            inspector_;
    SceneSelection            selection_;
    std::list<LoadedModel>    models_;   ///< list = stable addresses (atomic<bool> can't be moved)

    // Path input buffer for the "Load" bar
    char pathBuf_[512] = {};

    // ---- helpers -----------------------------------------------------------
    void DrawLoadBar(Engine* engine);
    void DrawModelList();
    void StartLoad(Engine* engine, const std::string& path);
};

// ---------------------------------------------------------------------------
// Factory function — called by SceneFactory::RegisterAll().
// ---------------------------------------------------------------------------
void LoadModelManagerScene(Engine* engine);
