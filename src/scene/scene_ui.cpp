#include "scene_ui.h"

#include "camera_component.h"
#include "engine.h"
#include "entity.h"
#include "mesh_component.h"
#include "renderer.h"
#include "transform_component.h"

#include "imgui/imgui.h"
#include "imgui/ImGuizmo.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <algorithm>
#include <cctype>
#include <string>

// ============================================================================
// SceneBrowserPanel
// ============================================================================

void SceneBrowserPanel::Draw(Engine* engine, SceneSelection& sel)
{
    if (!ImGui::Begin("Scene")) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("SceneTabs")) {

        // ---- Entities tab --------------------------------------------------
        if (ImGui::BeginTabItem("Entities")) {
            ImGui::SetNextItemWidth(-1.f);
            ImGui::InputText("##search", searchBuf_, sizeof(searchBuf_));
            ImGui::Separator();

            std::string filterLower = searchBuf_;
            std::transform(filterLower.begin(), filterLower.end(),
                           filterLower.begin(), [](unsigned char c){ return std::tolower(c); });

            ImGui::BeginChild("EntityList", ImVec2(0.f, 0.f), false,
                              ImGuiWindowFlags_HorizontalScrollbar);

            const auto& entities = engine->GetEntities();
            for (const auto& uptr : entities) {
                Entity* e = uptr.get();
                const std::string& name = e->GetName();

                // Apply filter
                if (!filterLower.empty()) {
                    std::string nameLower = name;
                    std::transform(nameLower.begin(), nameLower.end(),
                                   nameLower.begin(), [](unsigned char c){ return std::tolower(c); });
                    if (nameLower.find(filterLower) == std::string::npos)
                        continue;
                }

                // Component icon prefix
                bool hasMesh   = e->GetComponent<MeshComponent>()      != nullptr;
                bool hasCam    = e->GetComponent<CameraComponent>()     != nullptr;
                bool hasXform  = e->GetComponent<TransformComponent>()  != nullptr;

                const char* icon = hasMesh ? "[M]" : hasCam ? "[C]" : hasXform ? "[T]" : "[ ]";
                std::string label = std::string(icon) + " " + name + "##" + name;

                ImGuiTreeNodeFlags flags =
                    ImGuiTreeNodeFlags_Leaf        |
                    ImGuiTreeNodeFlags_NoTreePushOnOpen |
                    ImGuiTreeNodeFlags_SpanAvailWidth;
                if (sel.entity == e)
                    flags |= ImGuiTreeNodeFlags_Selected;

                ImGui::TreeNodeEx(label.c_str(), flags);
                if (ImGui::IsItemClicked()) {
                    sel.entity = (sel.entity == e) ? nullptr : e;
                }
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    sel.entity = nullptr; // double-click deselects
                }
            }

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ---- Stats tab -----------------------------------------------------
        if (ImGui::BeginTabItem("Stats")) {
            const auto& entities = engine->GetEntities();
            size_t meshCount  = 0;
            size_t totalVerts = 0;
            size_t totalIdx   = 0;
            for (const auto& uptr : entities) {
                auto* mc = uptr->GetComponent<MeshComponent>();
                if (mc) {
                    ++meshCount;
                    totalVerts += mc->GetVertices().size();
                    totalIdx   += mc->GetIndices().size();
                }
            }
            ImGui::Text("Entities : %zu", entities.size());
            ImGui::Text("Meshes   : %zu", meshCount);
            ImGui::Text("Vertices : %zu", totalVerts);
            ImGui::Text("Triangles: %zu", totalIdx / 3);
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

// ============================================================================
// InspectorPanel
// ============================================================================

void InspectorPanel::Draw(Engine*          engine,
                          Renderer*        renderer,
                          CameraComponent* cam,
                          SceneSelection&  sel)
{
    if (!ImGui::Begin("Inspector")) {
        ImGui::End();
        return;
    }

    if (!sel.entity) {
        ImGui::TextDisabled("No entity selected");
        ImGui::End();
        return;
    }

    Entity* e = sel.entity;
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.f), "%s", e->GetName().c_str());
    ImGui::Separator();

    // ------------------------------------------------------------------ //
    // TransformComponent
    // ------------------------------------------------------------------ //
    auto* tc = e->GetComponent<TransformComponent>();
    if (tc) {
        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {

            // Gizmo mode selector
            bool t = sel.gizmoMode == GizmoMode::Translate;
            bool r = sel.gizmoMode == GizmoMode::Rotate;
            bool s = sel.gizmoMode == GizmoMode::Scale;
            if (ImGui::RadioButton("T", t)) sel.gizmoMode = GizmoMode::Translate;
            ImGui::SameLine();
            if (ImGui::RadioButton("R", r)) sel.gizmoMode = GizmoMode::Rotate;
            ImGui::SameLine();
            if (ImGui::RadioButton("S", s)) sel.gizmoMode = GizmoMode::Scale;

            glm::vec3 pos = tc->GetPosition();
            glm::vec3 rot = glm::degrees(tc->GetRotation()); // display in degrees
            glm::vec3 scl = tc->GetScale();

            ImGui::SetNextItemWidth(-1.f);
            if (ImGui::DragFloat3("##pos", glm::value_ptr(pos), 0.01f))
                tc->SetPosition(pos);

            ImGui::SetNextItemWidth(-1.f);
            if (ImGui::DragFloat3("##rot", glm::value_ptr(rot), 0.5f))
                tc->SetRotation(glm::radians(rot));

            ImGui::SetNextItemWidth(-1.f);
            if (ImGui::DragFloat3("##scl", glm::value_ptr(scl), 0.01f, 0.0001f, 1000.f))
                tc->SetScale(scl);

            ImGui::TextDisabled("Pos / Rot(°) / Scale");

            // 3D gizmo overlay (only when we have a valid camera)
            if (cam) {
                glm::mat4 view = cam->GetViewMatrix();
                glm::mat4 proj = cam->GetProjectionMatrix();
                // Flip Y for ImGuizmo (GLM uses OpenGL convention, Vulkan flips Y)
                proj[1][1] *= -1.f;

                glm::mat4 model = tc->GetModelMatrix();

                ImGuiIO& io = ImGui::GetIO();
                ImGuizmo::SetRect(0.f, 0.f, io.DisplaySize.x, io.DisplaySize.y);
                ImGuizmo::SetOrthographic(false);

                ImGuizmo::OPERATION op = ImGuizmo::TRANSLATE;
                if (sel.gizmoMode == GizmoMode::Rotate) op = ImGuizmo::ROTATE;
                else if (sel.gizmoMode == GizmoMode::Scale) op = ImGuizmo::SCALE;

                float viewArr[16], projArr[16], modelArr[16];
                memcpy(viewArr,  glm::value_ptr(view),  sizeof(viewArr));
                memcpy(projArr,  glm::value_ptr(proj),  sizeof(projArr));
                memcpy(modelArr, glm::value_ptr(model), sizeof(modelArr));

                if (ImGuizmo::Manipulate(viewArr, projArr, op, ImGuizmo::WORLD, modelArr)) {
                    // Decompose result back to TRS
                    glm::mat4 newModel;
                    memcpy(glm::value_ptr(newModel), modelArr, sizeof(modelArr));

                    glm::vec3 newPos, newScale, skew;
                    glm::quat newRot;
                    glm::vec4 perspective;
                    if (glm::decompose(newModel, newScale, newRot, newPos, skew, perspective)) {
                        tc->SetPosition(newPos);
                        tc->SetRotation(glm::eulerAngles(newRot));
                        tc->SetScale(newScale);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ //
    // MeshComponent
    // ------------------------------------------------------------------ //
    auto* mc = e->GetComponent<MeshComponent>();
    if (mc) {
        if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Vertices  : %zu", mc->GetVertices().size());
            ImGui::Text("Triangles : %zu", mc->GetIndices().size() / 3);

            // Material properties from renderer cache (read-only)
            if (renderer) {
                MaterialProperties mp{};
                if (renderer->GetEntityMaterialProps(e, mp)) {
                    ImGui::Separator();
                    ImGui::TextDisabled("Material (read-only)");
                    glm::vec3 col = glm::vec3(mp.baseColorFactor);
                    ImGui::ColorButton("##bc", ImVec4(col.r, col.g, col.b, 1.f),
                                       ImGuiColorEditFlags_NoTooltip,
                                       ImVec2(14.f, 14.f));
                    ImGui::SameLine();
                    ImGui::Text("Base color  (%.2f %.2f %.2f)",
                                col.r, col.g, col.b);
                    ImGui::Text("Metallic    %.2f", mp.metallicFactor);
                    ImGui::Text("Roughness   %.2f", mp.roughnessFactor);
                    if (mp.emissiveStrength > 0.001f)
                        ImGui::Text("Emissive    %.2f", mp.emissiveStrength);
                    if (mp.transmissionFactor > 0.001f)
                        ImGui::Text("Transmission %.2f", mp.transmissionFactor);
                }
            }

            // Texture paths (truncated for readability)
            auto showTex = [](const char* label, const std::string& path) {
                if (path.empty()) return;
                // Show only the last two path components
                auto sep = path.rfind('/');
                if (sep == std::string::npos) sep = path.rfind('\\');
                std::string short_path = (sep != std::string::npos) ? path.substr(sep + 1) : path;
                ImGui::TextDisabled("%s: %s", label, short_path.c_str());
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", path.c_str());
            };

            ImGui::Separator();
            showTex("Albedo",    mc->GetBaseColorTexturePath());
            showTex("Normal",    mc->GetNormalTexturePath());
            showTex("MetalRgh",  mc->GetMetallicRoughnessTexturePath());
            showTex("Occlusion", mc->GetOcclusionTexturePath());
            showTex("Emissive",  mc->GetEmissiveTexturePath());
        }
    }

    // ------------------------------------------------------------------ //
    // CameraComponent
    // ------------------------------------------------------------------ //
    auto* cc = e->GetComponent<CameraComponent>();
    if (cc) {
        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::Text("FOV   : %.1f°", cc->GetFieldOfView());
            ImGui::Text("Near  : %.4f", cc->GetNearPlane());
            ImGui::Text("Far   : %.1f", cc->GetFarPlane());
            ImGui::Text("Aspect: %.3f", cc->GetAspectRatio());
        }
    }

    ImGui::End();
}
