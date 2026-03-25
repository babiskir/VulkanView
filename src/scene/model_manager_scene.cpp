#include "model_manager_scene.h"

#include "camera_component.h"
#include "engine.h"
#include "renderer.h"
#include "scene_loading.h"
#include "transform_component.h"

#include "imgui/imgui.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>

// ---------------------------------------------------------------------------
// ModelManagerScene
// ---------------------------------------------------------------------------

ModelManagerScene::ModelManagerScene(Engine* engine)
{
    // Default camera: wide angle, looking down at the origin
    auto* entity = engine->CreateEntity("Camera");
    if (entity) {
        auto* xform = entity->AddComponent<TransformComponent>();
        xform->SetPosition({0.f, 5.f, 10.f});

        auto* cam = entity->AddComponent<CameraComponent>();
        cam->SetAspectRatio(800.f / 600.f);
        cam->SetFieldOfView(60.f);
        cam->SetClipPlanes(0.01f, 2000.f);
        cam->SetTarget({0.f, 0.f, 0.f});
        engine->SetActiveCamera(cam);
    }
}

// ---------------------------------------------------------------------------

void ModelManagerScene::DrawUI(Engine* engine)
{
    DrawLoadBar(engine);
    DrawModelList();

    auto* renderer = engine->GetRenderer();
    auto* cam      = engine->GetActiveCamera();
    browser_.Draw(engine, selection_);
    inspector_.Draw(engine, renderer, cam, selection_);
}

// ---------------------------------------------------------------------------

void ModelManagerScene::DrawLoadBar(Engine* engine)
{
    ImGuiIO& io = ImGui::GetIO();
    // Fixed top bar across full width
    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 52.f), ImGuiCond_Always);
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    if (ImGui::Begin("##MMTopBar", nullptr, flags)) {
        ImGui::SetNextItemWidth(io.DisplaySize.x - 130.f);
        ImGui::InputText("##path", pathBuf_, sizeof(pathBuf_),
                         ImGuiInputTextFlags_EnterReturnsTrue);
        bool enterPressed = ImGui::IsItemFocused() && ImGui::IsKeyPressed(ImGuiKey_Enter);

        ImGui::SameLine();
        bool loadClicked = ImGui::Button("Load Model", ImVec2(110.f, 0.f));

        if ((loadClicked || enterPressed) && pathBuf_[0] != '\0') {
            StartLoad(engine, pathBuf_);
            pathBuf_[0] = '\0';
        }

        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Paste a path to a .gltf or .glb file and press Load");
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

// ---------------------------------------------------------------------------

void ModelManagerScene::DrawModelList()
{
    if (models_.empty()) return;

    ImGui::SetNextWindowSize(ImVec2(300.f, 0.f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Loaded Models")) {
        ImGui::End();
        return;
    }

    for (auto& m : models_) {
        if (m.loading.load()) {
            ImGui::TextDisabled("[loading] %s", m.displayName.c_str());
        } else {
            ImGui::Checkbox(("##vis_" + m.displayName).c_str(), &m.visible);
            ImGui::SameLine();
            ImGui::TextUnformatted(m.displayName.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", m.path.c_str());
        }
    }

    ImGui::End();
}

// ---------------------------------------------------------------------------

void ModelManagerScene::StartLoad(Engine* engine, const std::string& path)
{
    // Build display name from the file stem
    std::filesystem::path fp(path);
    std::string displayName = fp.stem().string();

    // Append to model list (before thread starts so UI shows it immediately)
    models_.emplace_back();
    LoadedModel& m = models_.back();
    m.path        = path;
    m.displayName = displayName;
    m.loading.store(true);

    // Show loading overlay
    auto* renderer = engine->GetRenderer();
    if (renderer) {
        renderer->SetLoading(true);
        renderer->SetLoadingPhase(Renderer::LoadingPhase::Textures);
    }

    // Background load (LoadGLTFModel is designed for background-thread use)
    LoadedModel* mPtr = &m;  // safe: vector won't reallocate because we only append
    std::thread([engine, path, mPtr]() {
        LoadGLTFModel(engine, path);
        mPtr->loading.store(false);
        std::cout << "[ModelManager] Loaded: " << path << "\n";
    }).detach();
}

// ---------------------------------------------------------------------------
// Factory function (called from SceneFactory::RegisterAll)
// ---------------------------------------------------------------------------

void LoadModelManagerScene(Engine* engine)
{
    // Keep the scene alive in the lambda via shared_ptr
    auto scene = std::make_shared<ModelManagerScene>(engine);

    engine->SetSceneUI([scene, engine]() mutable {
        scene->DrawUI(engine);
    });
}
