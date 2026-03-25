#include "scene_factory.h"

#include "camera_component.h"
#include "engine.h"
#include "mesh_component.h"
#include "buoyancy_scene.h"
#include "model_manager_scene.h"
#include "physics_scenes.h"
#include "spawner_scene.h"
#include "primitive_meshes.h"
#include "renderer.h"
#include "scene_loading.h"
#include "scene_ui.h"
#include "transform_component.h"
#include "water_system.h"

#include "imgui/imgui.h"

#include <glm/vec3.hpp>
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>

// ---------------------------------------------------------------------------
// Shared UI installer — gives every scene a browser + inspector sidebar.
// The panels are heap-allocated and kept alive by the lambda captured in
// Engine::SetSceneUI so they live for the entire session.
// ---------------------------------------------------------------------------
static void InstallSceneUI(Engine* engine)
{
    auto browser   = std::make_shared<SceneBrowserPanel>();
    auto inspector = std::make_shared<InspectorPanel>();
    auto selection = std::make_shared<SceneSelection>();

    engine->SetSceneUI([engine, browser, inspector, selection]() mutable {
        auto* cam = engine->GetActiveCamera();
        auto* renderer = engine->GetRenderer();
        auto* imgui = engine->GetImGuiSystem();
        if (!imgui || imgui->panelState.showScene)
            browser->Draw(engine, *selection);
        if (!imgui || imgui->panelState.showInspector)
            inspector->Draw(engine, renderer, cam, *selection);
    });
}

// ---------------------------------------------------------------------------
// Camera helper
// ---------------------------------------------------------------------------
static CameraComponent* MakeCamera(Engine* engine,
                                    glm::vec3 eye,
                                    glm::vec3 target    = glm::vec3(0.f),
                                    float     fov       = 60.f,
                                    float     nearPlane = 0.1f,
                                    float     farPlane  = 500.f)
{
    auto* entity = engine->CreateEntity("Camera");
    if (!entity) throw std::runtime_error("Failed to create camera entity");

    auto* xform = entity->AddComponent<TransformComponent>();
    xform->SetPosition(eye);

    auto* cam = entity->AddComponent<CameraComponent>();
    // Use a sensible default aspect; the engine will update it on resize.
    cam->SetAspectRatio(800.f / 600.f);
    cam->SetFieldOfView(fov);
    cam->SetClipPlanes(nearPlane, farPlane);
    cam->SetTarget(target);
    engine->SetActiveCamera(cam);
    return cam;
}

// ---------------------------------------------------------------------------
// Mesh helper
// ---------------------------------------------------------------------------
static void FillMesh(MeshComponent* mesh, const PrimMesh& prim)
{
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
    mesh->SetVertices(verts);
    mesh->SetIndices(prim.indices);
}

// ---------------------------------------------------------------------------
// Scene: Bistro Interior
// ---------------------------------------------------------------------------
static void LoadBistroScene(Engine* engine)
{
    MakeCamera(engine, {0.f, 2.f, 5.f}, {0.f, 1.f, 0.f});

    auto* renderer = engine->GetRenderer();
    if (renderer) {
        renderer->SetLoading(true);
        renderer->SetLoadingPhase(Renderer::LoadingPhase::Textures);
    }

    std::thread([engine] {
        LoadGLTFModel(engine, "../Assets/bistro/bistro.gltf");
    }).detach();

    InstallSceneUI(engine);
}

// ---------------------------------------------------------------------------
// Scene: Water Surface (Tessendorf ocean)
// ---------------------------------------------------------------------------
static void LoadWaterScene(Engine* engine)
{
    auto* cam = MakeCamera(engine,
                           {0.f,  8.f, -150.f},
                           {0.f,  0.f,  300.f},
                           80.f, 0.1f, 2000.f);
    (void)cam;

    auto* renderer = engine->GetRenderer();
    if (!renderer) throw std::runtime_error("No renderer for water scene");

    auto* waterSystem = new WaterSystem();
    if (!waterSystem->Initialize(renderer)) {
        delete waterSystem;
        throw std::runtime_error("Failed to initialize WaterSystem");
    }
    renderer->waterSystem = waterSystem;
    std::cout << "[Scene] Water scene ready.\n";

    InstallSceneUI(engine);
}

// ---------------------------------------------------------------------------
// Scene: Physics (generic drop demo — kept for the old LoadPhysicsScene path)
// ---------------------------------------------------------------------------
static void LoadPhysicsDropScene(Engine* engine)
{
    auto* cam = MakeCamera(engine, {0.f, 15.f, 25.f}, {0.f, 0.f, 0.f}, 60.f, 0.1f, 500.f);
    (void)cam;

    auto* renderer      = engine->GetRenderer();
    auto* physicsSystem = engine->GetPhysicsSystem();
    if (!renderer || !physicsSystem) {
        std::cerr << "[PhysicsDropScene] Missing renderer or physics system\n";
        return;
    }

    std::vector<Entity*> allEntities;

    // Ground plane (static)
    {
        auto* groundEntity = engine->CreateEntity("PhysGround");
        auto* t = groundEntity->AddComponent<TransformComponent>();
        t->SetPosition({0.f, -0.5f, 0.f});
        t->SetScale({40.f, 1.f, 40.f});

        auto* m = groundEntity->AddComponent<MeshComponent>();
        FillMesh(m, MakeBox(0.5f));
        m->SetTexturePath("__shared_default_albedo__");

        allEntities.push_back(groundEntity);
        physicsSystem->EnqueueRigidBodyCreation(groundEntity, CollisionShape::Box,
                                                0.f, true, 0.4f, 0.6f);
    }

    // 25 random falling shapes
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> xzDist(-5.f, 5.f);
    std::uniform_real_distribution<float> yDist(3.f, 30.f);
    std::uniform_int_distribution<int>    shapeDist(0, 2);

    for (int i = 0; i < 25; ++i) {
        int  shapeIdx = shapeDist(rng);
        auto shape    = static_cast<CollisionShape>(shapeIdx);
        glm::vec3 pos{xzDist(rng), yDist(rng), xzDist(rng)};

        auto* entity = engine->CreateEntity("PhysObj_" + std::to_string(i));
        auto* t = entity->AddComponent<TransformComponent>();
        t->SetPosition(pos);
        t->SetScale(glm::vec3(1.f));

        auto* m = entity->AddComponent<MeshComponent>();
        PrimMesh prim;
        if      (shape == CollisionShape::Box)    prim = MakeBox(0.5f);
        else if (shape == CollisionShape::Sphere) prim = MakeSphere(0.5f);
        else                                      prim = MakeCapsule(0.25f, 0.5f);
        FillMesh(m, prim);
        m->SetTexturePath("__shared_default_albedo__");

        allEntities.push_back(entity);
        physicsSystem->EnqueueRigidBodyCreation(entity, shape, 1.f, false, 0.4f, 0.5f);
    }

    renderer->EnqueueEntityPreallocationBatch(allEntities);

    std::cout << "[Scene] Physics drop demo loaded (ground + 25 objects).\n";
    InstallSceneUI(engine);
}

// ---------------------------------------------------------------------------
// SceneFactory singleton
// ---------------------------------------------------------------------------
SceneFactory& SceneFactory::Instance()
{
    static SceneFactory inst;
    return inst;
}

void SceneFactory::Register(Entry e)
{
    entries_.push_back(std::move(e));
}

// Wrap a void(Engine*) loader so it calls InstallSceneUI after loading.
static std::function<void(Engine*)> WithUI(void(*loader)(Engine*))
{
    return [loader](Engine* e) {
        loader(e);
        InstallSceneUI(e);
    };
}

void SceneFactory::RegisterAll()
{
    // Root-level scenes (loaders already call InstallSceneUI internally)
    Register({"bistro",        "Bistro Interior  (GLTF + PBR)",          "", LoadBistroScene});
    Register({"water",         "Water Surface    (Tessendorf Ocean)",     "", LoadWaterScene});
    Register({"physics",       "Physics Drop     (25 random shapes)",     "", LoadPhysicsDropScene});
    Register({"model-manager", "Model Manager    (load any GLTF at runtime)", "", LoadModelManagerScene});
    Register({"buoyancy",      "Buoyancy Demo    (JONSWAP ocean + PhysX mesh buoyancy)", "", LoadBuoyancyScene});
    Register({"spawner",       "Object Spawner   (EnoSea-style: 5 shapes + JONSWAP buoyancy)", "", LoadSpawnerScene});

    // Physics sub-scenes (group = "physics") — wrapped to also install UI panels
    Register({"physics-hello",    "Hello World  (Box Stacks + Sphere)", "physics", WithUI(LoadPhysicsScene_HelloWorld)});
    Register({"physics-joints",   "Joints       (Spherical Chain)",     "physics", WithUI(LoadPhysicsScene_Joints)});
    Register({"physics-ccd",      "CCD          (Fast Ball vs Wall)",   "physics", WithUI(LoadPhysicsScene_CCD)});
    Register({"physics-gyro",     "Gyroscopic   (Dzhanibekov Effect)",  "physics", WithUI(LoadPhysicsScene_Gyroscopic)});
    Register({"physics-triggers", "Triggers     (Trigger Volumes)",     "physics", WithUI(LoadPhysicsScene_Triggers)});
}

bool SceneFactory::Load(const std::string& id, Engine* engine) const
{
    for (const auto& e : entries_) {
        if (e.id == id) {
            engine->ClearScene();
            e.load(engine);
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// ImGui picker — built from the registry
// ---------------------------------------------------------------------------
void SceneFactory::InstallPicker(Engine* engine) const
{
    // Capture the singleton reference (lives forever) and the engine pointer.
    const SceneFactory& factory = *this;

    engine->SetScenePicker([engine, &factory]() {
        static bool subMenuOpen = false;

        // Helper: fire a scene load safely via the deferred mechanism.
        auto fireScene = [engine](std::function<void(Engine*)> load) {
            engine->SetScenePicker(nullptr);
            engine->SetSceneLoaded(true);
            engine->SetPendingSceneLoader([engine, load = std::move(load)]() {
                // Clear stale entities / physics actors from any previously loaded scene.
                engine->ClearScene();
                load(engine);
            });
        };

        // Partition entries into root and physics groups.
        std::vector<const SceneFactory::Entry*> rootEntries;
        std::vector<const SceneFactory::Entry*> physicsEntries;
        for (const auto& e : factory.Entries()) {
            if      (e.group.empty())    rootEntries.push_back(&e);
            else if (e.group == "physics") physicsEntries.push_back(&e);
        }

        ImGuiIO& io = ImGui::GetIO();

        // Full-screen dark background
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
        ImGuiWindowFlags bgFlags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.07f, 1.f));
        ImGui::Begin("##PickerBg", nullptr, bgFlags);
        ImGui::PopStyleColor();
        ImGui::End();

        // Centred card — fixed width, scrollable height
        const float cardW    = 480.f;
        const float btnH     = 48.f;
        const float hdrH     = 56.f;  // title + separator + spacing
        const float maxCardH = io.DisplaySize.y * 0.82f;

        // Compute natural height then clamp
        int  nItems  = subMenuOpen ? (int)physicsEntries.size() + 1 /*Back btn*/
                                   : (int)rootEntries.size();
        float naturalH = hdrH + nItems * (btnH + 6.f) + 20.f;
        float cardH    = std::min(naturalH, maxCardH);

        float posX = (io.DisplaySize.x - cardW) * 0.5f;
        float posY = (io.DisplaySize.y - cardH) * 0.5f;
        ImGui::SetNextWindowPos(ImVec2(posX, posY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(cardW, cardH), ImGuiCond_Always);
        ImGuiWindowFlags cardFlags =
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.f, 10.f));
        if (ImGui::Begin("  VulkanView — Select Scene", nullptr, cardFlags)) {
            ImGui::PopStyleVar();
            const float btnW = cardW - 28.f;

            if (!subMenuOpen) {
                // ---- Header ----
                ImGui::Spacing();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.75f, 0.95f, 1.f));
                ImGui::TextUnformatted("Select a scene to load:");
                ImGui::PopStyleColor();
                ImGui::Separator();
                ImGui::Spacing();

                // Scrollable list
                ImGui::BeginChild("##SceneList", ImVec2(0, 0), false);
                for (const auto* ep : rootEntries) {
                    const auto& e = *ep;
                    bool isPhysicsGroup = (e.id == "physics");
                    if (isPhysicsGroup) {
                        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.13f,0.20f,0.31f,1.f));
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f,0.30f,0.46f,1.f));
                        if (ImGui::Button("  [PhysX]  Physics Demos   (5 PhysX 5 Snippet Scenes) >",
                                          ImVec2(btnW, btnH)))
                            subMenuOpen = true;
                        ImGui::PopStyleColor(2);
                    } else {
                        if (ImGui::Button(("  " + e.label).c_str(), ImVec2(btnW, btnH)))
                            fireScene(e.load);
                    }
                    ImGui::Spacing();
                }
                ImGui::EndChild();
            } else {
                // ---- Physics submenu ----
                ImGui::Spacing();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.75f, 0.95f, 1.f));
                ImGui::TextUnformatted("PhysX 5 Demo Scenes:");
                ImGui::PopStyleColor();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::BeginChild("##PhysicsList", ImVec2(0, cardH - hdrH - 50.f), false);
                for (const auto* ep : physicsEntries) {
                    const auto& e = *ep;
                    if (ImGui::Button(("  " + e.label).c_str(), ImVec2(btnW - 8.f, btnH - 6.f))) {
                        subMenuOpen = false;
                        fireScene(e.load);
                    }
                    ImGui::Spacing();
                }
                ImGui::EndChild();

                ImGui::Separator();
                ImGui::Spacing();
                if (ImGui::Button("< Back", ImVec2(90.f, 28.f)))
                    subMenuOpen = false;
            }
        } else {
            ImGui::PopStyleVar();
        }
        ImGui::End();
    });
}
