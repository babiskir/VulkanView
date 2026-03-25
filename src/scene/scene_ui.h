#pragma once

#include <string>

// Forward declarations — keep headers lean
class Engine;
class Renderer;
class CameraComponent;
class Entity;

// ---------------------------------------------------------------------------
// Gizmo operation mode
// ---------------------------------------------------------------------------
enum class GizmoMode { Translate = 0, Rotate = 1, Scale = 2 };

// ---------------------------------------------------------------------------
// SceneSelection — shared selection state between browser and inspector.
// ---------------------------------------------------------------------------
struct SceneSelection {
    Entity*    entity    = nullptr;   ///< Currently selected entity (nullptr = nothing)
    GizmoMode  gizmoMode = GizmoMode::Translate;
};

// ---------------------------------------------------------------------------
// SceneBrowserPanel
//
// A dockable ImGui window showing all engine entities in a searchable list,
// plus a Stats tab. Clicking an entity sets sel.entity.
// ---------------------------------------------------------------------------
class SceneBrowserPanel {
public:
    /// Draw the panel. Call once per ImGui frame.
    void Draw(Engine* engine, SceneSelection& sel);

private:
    char searchBuf_[128] = {};
};

// ---------------------------------------------------------------------------
// InspectorPanel
//
// A dockable ImGui window that shows the selected entity's components:
//   • TransformComponent : editable TRS + ImGuizmo overlay
//   • MeshComponent      : vertex/index counts + texture assignments
//   • CameraComponent    : FOV, clip planes
// ---------------------------------------------------------------------------
class InspectorPanel {
public:
    /// Draw the panel. cam may be nullptr (gizmo is skipped).
    void Draw(Engine*          engine,
              Renderer*        renderer,
              CameraComponent* cam,
              SceneSelection&  sel);
};
