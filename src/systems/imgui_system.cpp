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
#include "imgui_system.h"
#include "audio_system.h"
#include "camera_component.h"
#include "renderer.h"
#include "renderdoc_debug_system.h"
#include "transform_component.h"

// Include ImGui headers
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"   // DockBuilder API
#include "imgui/ImGuizmo.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>

// ---------------------------------------------------------------------------
// Static debug log storage
// ---------------------------------------------------------------------------
std::deque<std::string> ImGuiSystem::s_debugMessages;
std::mutex              ImGuiSystem::s_debugMutex;

void ImGuiSystem::DebugLog(const std::string& msg) {
    // Also echo to stderr for terminal viewing
    std::cerr << "[DBG] " << msg << "\n";
    std::lock_guard<std::mutex> lk(s_debugMutex);
    // Timestamp
    auto now  = std::chrono::system_clock::now();
    auto t    = std::chrono::system_clock::to_time_t(now);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    s_debugMessages.push_back(std::string(buf) + "  " + msg);
    while ((int)s_debugMessages.size() > kMaxDebugLines)
        s_debugMessages.pop_front();
}

void ImGuiSystem::DebugLogClear() {
    std::lock_guard<std::mutex> lk(s_debugMutex);
    s_debugMessages.clear();
}

// ---------------------------------------------------------------------------
// Engine visual style — close match to nvpro-samples/vk_gltf_renderer
// Palette: near-black bg (#0E0F11), dark-slate panels (#161820),
//          blue accent (#4D9FD6 / 0.302,0.624,0.839), warm text (#E8E8EA)
// ---------------------------------------------------------------------------
static void ApplyEngineStyle()
{
    ImGuiStyle& s = ImGui::GetStyle();

    // ---- Shape ----
    s.WindowRounding     = 2.f;
    s.ChildRounding      = 2.f;
    s.FrameRounding      = 2.f;
    s.PopupRounding      = 2.f;
    s.ScrollbarRounding  = 2.f;
    s.GrabRounding       = 2.f;
    s.TabRounding        = 2.f;

    // ---- Sizing ----
    s.WindowPadding      = ImVec2(8.f,  6.f);
    s.FramePadding       = ImVec2(5.f,  3.f);
    s.CellPadding        = ImVec2(4.f,  2.f);
    s.ItemSpacing        = ImVec2(6.f,  3.f);
    s.ItemInnerSpacing   = ImVec2(4.f,  4.f);
    s.IndentSpacing      = 14.f;
    s.ScrollbarSize      = 11.f;
    s.GrabMinSize        = 8.f;
    s.TabBarBorderSize   = 1.f;

    // ---- Borders ----
    s.WindowBorderSize   = 1.f;
    s.ChildBorderSize    = 0.f;
    s.PopupBorderSize    = 1.f;
    s.FrameBorderSize    = 0.f;

    ImVec4* c = s.Colors;

    // ---- Base surfaces (dark slate) ----
    c[ImGuiCol_WindowBg]           = ImVec4(0.086f, 0.090f, 0.102f, 1.00f);  // #16171A
    c[ImGuiCol_ChildBg]            = ImVec4(0.071f, 0.075f, 0.086f, 1.00f);  // #121316
    c[ImGuiCol_PopupBg]            = ImVec4(0.086f, 0.090f, 0.102f, 0.98f);
    c[ImGuiCol_Border]             = ImVec4(0.200f, 0.208f, 0.235f, 0.80f);
    c[ImGuiCol_BorderShadow]       = ImVec4(0.000f, 0.000f, 0.000f, 0.00f);

    // ---- Text ----
    c[ImGuiCol_Text]               = ImVec4(0.910f, 0.910f, 0.918f, 1.00f);  // #E8E8EA
    c[ImGuiCol_TextDisabled]       = ImVec4(0.420f, 0.430f, 0.460f, 1.00f);

    // ---- Frames (inputs, sliders, checkboxes) ----
    c[ImGuiCol_FrameBg]            = ImVec4(0.141f, 0.149f, 0.169f, 1.00f);  // #242630
    c[ImGuiCol_FrameBgHovered]     = ImVec4(0.192f, 0.200f, 0.224f, 1.00f);
    c[ImGuiCol_FrameBgActive]      = ImVec4(0.220f, 0.360f, 0.520f, 0.60f);

    // ---- Title bars ----
    c[ImGuiCol_TitleBg]            = ImVec4(0.059f, 0.063f, 0.075f, 1.00f);  // #0F1013
    c[ImGuiCol_TitleBgActive]      = ImVec4(0.071f, 0.078f, 0.094f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed]   = ImVec4(0.059f, 0.063f, 0.075f, 0.80f);

    // ---- Menu bar ----
    c[ImGuiCol_MenuBarBg]          = ImVec4(0.055f, 0.059f, 0.071f, 1.00f);  // #0E0F12

    // ---- Scrollbar ----
    c[ImGuiCol_ScrollbarBg]        = ImVec4(0.039f, 0.039f, 0.047f, 0.00f);
    c[ImGuiCol_ScrollbarGrab]      = ImVec4(0.220f, 0.231f, 0.259f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered]= ImVec4(0.302f, 0.318f, 0.357f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.400f, 0.420f, 0.471f, 1.00f);

    // ---- Accent: blue (#4D9FD6 ≈ 0.302,0.624,0.839) ----
    const ImVec4 accent      = ImVec4(0.302f, 0.624f, 0.839f, 1.00f);
    const ImVec4 accentDim   = ImVec4(0.200f, 0.490f, 0.700f, 1.00f);
    const ImVec4 accentFaint = ImVec4(0.200f, 0.420f, 0.620f, 0.40f);

    // ---- Checkboxes, sliders ----
    c[ImGuiCol_CheckMark]          = accent;
    c[ImGuiCol_SliderGrab]         = accentDim;
    c[ImGuiCol_SliderGrabActive]   = accent;

    // ---- Buttons ----
    c[ImGuiCol_Button]             = ImVec4(0.157f, 0.165f, 0.188f, 1.00f);
    c[ImGuiCol_ButtonHovered]      = ImVec4(0.220f, 0.376f, 0.569f, 1.00f);
    c[ImGuiCol_ButtonActive]       = ImVec4(0.157f, 0.329f, 0.541f, 1.00f);

    // ---- Headers (CollapsingHeader, Selectable) ----
    c[ImGuiCol_Header]             = ImVec4(0.180f, 0.329f, 0.510f, 0.40f);
    c[ImGuiCol_HeaderHovered]      = ImVec4(0.220f, 0.400f, 0.612f, 0.80f);
    c[ImGuiCol_HeaderActive]       = ImVec4(0.220f, 0.400f, 0.612f, 1.00f);

    // ---- Separators ----
    c[ImGuiCol_Separator]          = ImVec4(0.200f, 0.208f, 0.235f, 0.80f);
    c[ImGuiCol_SeparatorHovered]   = accentDim;
    c[ImGuiCol_SeparatorActive]    = accent;

    // ---- Resize grips ----
    c[ImGuiCol_ResizeGrip]         = accentFaint;
    c[ImGuiCol_ResizeGripHovered]  = ImVec4(0.302f, 0.624f, 0.839f, 0.67f);
    c[ImGuiCol_ResizeGripActive]   = ImVec4(0.302f, 0.624f, 0.839f, 0.95f);

    // ---- Tabs ----
    c[ImGuiCol_Tab]                = ImVec4(0.110f, 0.118f, 0.137f, 1.00f);
    c[ImGuiCol_TabHovered]         = ImVec4(0.220f, 0.376f, 0.569f, 0.80f);
    c[ImGuiCol_TabActive]          = ImVec4(0.160f, 0.310f, 0.490f, 1.00f);
    c[ImGuiCol_TabUnfocused]       = ImVec4(0.090f, 0.094f, 0.110f, 1.00f);
    c[ImGuiCol_TabUnfocusedActive] = ImVec4(0.130f, 0.220f, 0.345f, 1.00f);

    // ---- Docking ----
    c[ImGuiCol_DockingPreview]     = ImVec4(0.302f, 0.624f, 0.839f, 0.70f);
    c[ImGuiCol_DockingEmptyBg]     = ImVec4(0.059f, 0.063f, 0.075f, 1.00f);

    // ---- Plot ----
    c[ImGuiCol_PlotLines]          = ImVec4(0.549f, 0.620f, 0.710f, 1.00f);
    c[ImGuiCol_PlotLinesHovered]   = ImVec4(0.960f, 0.510f, 0.180f, 1.00f);
    c[ImGuiCol_PlotHistogram]      = accentDim;
    c[ImGuiCol_PlotHistogramHovered]= accent;

    // ---- Tables ----
    c[ImGuiCol_TableHeaderBg]      = ImVec4(0.110f, 0.118f, 0.137f, 1.00f);
    c[ImGuiCol_TableBorderStrong]  = ImVec4(0.200f, 0.208f, 0.235f, 1.00f);
    c[ImGuiCol_TableBorderLight]   = ImVec4(0.149f, 0.157f, 0.176f, 1.00f);
    c[ImGuiCol_TableRowBg]         = ImVec4(0.000f, 0.000f, 0.000f, 0.00f);
    c[ImGuiCol_TableRowBgAlt]      = ImVec4(1.000f, 1.000f, 1.000f, 0.03f);

    // ---- Misc ----
    c[ImGuiCol_TextSelectedBg]     = ImVec4(0.302f, 0.624f, 0.839f, 0.35f);
    c[ImGuiCol_DragDropTarget]     = ImVec4(1.000f, 0.800f, 0.000f, 0.90f);
    c[ImGuiCol_NavHighlight]       = accent;
    c[ImGuiCol_NavWindowingHighlight]= ImVec4(1.f, 1.f, 1.f, 0.70f);
    c[ImGuiCol_NavWindowingDimBg]  = ImVec4(0.8f, 0.8f, 0.8f, 0.20f);
    c[ImGuiCol_ModalWindowDimBg]   = ImVec4(0.06f, 0.06f, 0.08f, 0.60f);
}

// ---------------------------------------------------------------------------
// Initial dock layout — called once when no saved layout exists.
// Arranges the built-in panels around the 3D viewport.
// ---------------------------------------------------------------------------
static void BuildInitialDockLayout(ImGuiID dockID)
{
    ImVec2 size = ImGui::GetMainViewport()->WorkSize;

    ImGui::DockBuilderRemoveNode(dockID);
    ImGui::DockBuilderAddNode(dockID, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockID, size);

    // ┌────────────────────────────────┬──────────────────┐
    // │                                │  Scene Browser   │
    // │        3D Viewport             ├──────────────────┤
    // │        (pass-through)          │  Inspector       │
    // ├───────────────────────┬────────┴──────────────────┤
    // │   Debug Output        │  Performance   │  Audio   │
    // └────────────────────────────────────────────────────┘

    ImGuiID rightID, centerID;
    ImGui::DockBuilderSplitNode(dockID, ImGuiDir_Right, 0.26f, &rightID, &centerID);

    ImGuiID rightTopID, rightBotID;
    ImGui::DockBuilderSplitNode(rightID, ImGuiDir_Down, 0.50f, &rightBotID, &rightTopID);

    ImGuiID bottomID, viewportID;
    ImGui::DockBuilderSplitNode(centerID, ImGuiDir_Down, 0.22f, &bottomID, &viewportID);

    ImGui::DockBuilderDockWindow("Scene",               rightTopID);
    ImGui::DockBuilderDockWindow("Loaded Models",       rightTopID);
    ImGui::DockBuilderDockWindow("Inspector",           rightBotID);
    ImGui::DockBuilderDockWindow("Performance",         bottomID);
    ImGui::DockBuilderDockWindow("Debug Output",        bottomID);
    ImGui::DockBuilderDockWindow("HRTF Audio Controls", bottomID);
    ImGui::DockBuilderDockWindow("Object Spawner",      rightTopID);
    ImGui::DockBuilderDockWindow("Ocean Environment",   rightBotID);

    ImGui::DockBuilderFinish(dockID);
}

// ---------------------------------------------------------------------------
// DrawToolbar — vk_gltf_renderer–style icon/toggle bar below the menu bar
// ---------------------------------------------------------------------------
void ImGuiSystem::DrawToolbar()
{
    ImGuiIO& io = ImGui::GetIO();

    // Position just below the menu bar
    const float menuBarH = ImGui::GetFrameHeight();
    ImGui::SetNextWindowPos(ImVec2(0, menuBarH), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 32.f), ImGuiCond_Always);

    ImGuiWindowFlags tbFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize  |
        ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoScrollbar|
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.f, 4.f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.f, 0.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.059f, 0.063f, 0.075f, 1.f));

    if (ImGui::Begin("##Toolbar", nullptr, tbFlags)) {
        // Helper: toggled button with highlight when active
        auto ToggleBtn = [&](const char* label, bool& state, const char* tip = nullptr) {
            if (state) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.16f,0.31f,0.49f,1.f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f,0.40f,0.61f,1.f));
            }
            if (ImGui::Button(label, ImVec2(0, 22.f))) state = !state;
            if (state) ImGui::PopStyleColor(2);
            if (tip && ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tip);
        };

        // ---- PBR shading ----
        ToggleBtn(pbrEnabled ? "PBR" : "Flat", pbrEnabled, "Toggle PBR / flat shading");
        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        // ---- Viewport toggles ----
        ToggleBtn("Grid",      toolbarState.showGrid,      "Toggle ground grid");
        ImGui::SameLine();
        ToggleBtn("Wire",      toolbarState.showWireframe, "Toggle wireframe overlay");
        ImGui::SameLine();
        ToggleBtn("PhysDbg",   toolbarState.showPhysDebug, "Toggle physics debug shapes");
        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        // ---- Debug output panel ----
        ToggleBtn("Log", toolbarState.showDebugPanel, "Toggle debug log panel");
        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        // ---- RenderDoc ----
        auto& rdoc = RenderDocDebugSystem::GetInstance();
        if (rdoc.IsAvailable()) {
            if (ImGui::Button("RDC", ImVec2(0, 22.f)))
                rdoc.TriggerCapture();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Trigger RenderDoc frame capture");
            ImGui::SameLine();
        }

        // ---- Right-aligned frame stats ----
        if (renderer) {
            char statsText[64];
            const uint32_t upTot  = renderer->GetUploadJobsTotal();
            const uint32_t upDone = renderer->GetUploadJobsCompleted();
            if (upTot > 0 && upDone < upTot)
                std::snprintf(statsText, sizeof(statsText), "Streaming %u/%u tex", upDone, upTot);
            else
                std::snprintf(statsText, sizeof(statsText), "");

            if (statsText[0]) {
                float tw = ImGui::CalcTextSize(statsText).x + 10.f;
                ImGui::SetCursorPosX(io.DisplaySize.x - tw);
                ImGui::TextDisabled("%s", statsText);
            }
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(4);
}

// ---------------------------------------------------------------------------
// DrawDebugPanel — scrollable log window (dockable at bottom)
// ---------------------------------------------------------------------------
void ImGuiSystem::DrawDebugPanel()
{
    if (!toolbarState.showDebugPanel) return;

    ImGui::SetNextWindowSize(ImVec2(600, 180), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Debug Output", &toolbarState.showDebugPanel)) {
        // Buttons row
        if (ImGui::SmallButton("Clear")) DebugLogClear();
        ImGui::SameLine();
        ImGui::TextDisabled("(%d lines)", (int)s_debugMessages.size());
        ImGui::SameLine();
        static bool autoScroll = true;
        ImGui::Checkbox("Auto-scroll", &autoScroll);

        ImGui::Separator();

        // Message list
        ImGui::BeginChild("##DebugLog", ImVec2(0, 0), false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        {
            std::lock_guard<std::mutex> lk(s_debugMutex);
            for (const auto& line : s_debugMessages) {
                // Colour-code errors and warnings
                bool isErr  = (line.find("ERROR")   != std::string::npos ||
                               line.find("CRASH")   != std::string::npos ||
                               line.find("EXCEPTION")!= std::string::npos);
                bool isWarn = (line.find("WARNING") != std::string::npos ||
                               line.find("WARN")    != std::string::npos ||
                               line.find("failed")  != std::string::npos ||
                               line.find("Failed")  != std::string::npos);
                if (isErr)
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.35f, 0.35f, 1.f));
                else if (isWarn)
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.80f, 0.20f, 1.f));

                ImGui::TextUnformatted(line.c_str());

                if (isErr || isWarn)
                    ImGui::PopStyleColor();
            }
            if (autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.f)
                ImGui::SetScrollHereY(1.f);
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

// This implementation corresponds to the GUI chapter in the tutorial:
// @see en/Building_a_Simple_Engine/GUI/02_imgui_setup.adoc

ImGuiSystem::ImGuiSystem() {
  // Constructor implementation
}

ImGuiSystem::~ImGuiSystem() {
  // Destructor implementation
  Cleanup();
}

bool ImGuiSystem::Initialize(Renderer* renderer, uint32_t width, uint32_t height) {
  if (initialized) {
    return true;
  }

  this->renderer = renderer;
  this->width = width;
  this->height = height;

  // Create ImGui context
  context = ImGui::CreateContext();
  if (!context) {
    std::cerr << "Failed to create ImGui context" << std::endl;
    return false;
  }

  // Configure ImGui
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  // Set display size
  io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
  io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

  // Set up ImGui style
  ApplyEngineStyle();

  // Create Vulkan resources
  if (!createResources()) {
    std::cerr << "Failed to create ImGui Vulkan resources" << std::endl;
    Cleanup();
    return false;
  }

  // Initialize per-frame buffers containers
  if (renderer) {
    uint32_t frames = renderer->GetMaxFramesInFlight();
    vertexBuffers.clear();
    vertexBuffers.reserve(frames);
    vertexBufferMemories.clear();
    vertexBufferMemories.reserve(frames);
    indexBuffers.clear();
    indexBuffers.reserve(frames);
    indexBufferMemories.clear();
    indexBufferMemories.reserve(frames);
    for (uint32_t i = 0; i < frames; ++i) {
      vertexBuffers.emplace_back(nullptr);
      vertexBufferMemories.emplace_back(nullptr);
      indexBuffers.emplace_back(nullptr);
      indexBufferMemories.emplace_back(nullptr);
    }
    vertexCounts.assign(frames, 0);
    indexCounts.assign(frames, 0);
  }

  initialized = true;
  return true;
}

void ImGuiSystem::Cleanup() {
  if (!initialized) {
    return;
  }

  // Wait for the device to be idle before cleaning up
  if (renderer) {
    renderer->WaitIdle();
  }
  // Destroy ImGui context
  if (context) {
    ImGui::DestroyContext(context);
    context = nullptr;
  }

  initialized = false;
}

void ImGuiSystem::SetAudioSystem(AudioSystem* audioSystem) {
  this->audioSystem = audioSystem;

  // Load the grass-step-right.wav file and create audio source
  if (audioSystem) {
    if (audioSystem->LoadAudio("../Assets/grass-step-right.wav", "grass_step")) {
      audioSource = audioSystem->CreateAudioSource("grass_step");
      if (audioSource) {
        audioSource->SetPosition(audioSourceX, audioSourceY, audioSourceZ);
        audioSource->SetVolume(0.8f);
        audioSource->SetLoop(true);
        std::cout << "Audio source created and configured for HRTF demo" << std::endl;
      }
    }

    // Also create a debug ping source for testing
    debugPingSource = audioSystem->CreateDebugPingSource("debug_ping");
    if (debugPingSource) {
      debugPingSource->SetPosition(audioSourceX, audioSourceY, audioSourceZ);
      debugPingSource->SetVolume(0.8f);
      debugPingSource->SetLoop(true);
      std::cout << "Debug ping source created for audio debugging" << std::endl;
    }
  }
}

void ImGuiSystem::NewFrame() {
  if (!initialized) {
    return;
  }

  // Reset the flag at the start of each frame
  frameAlreadyRendered = false;

  // Track frame time for the profiling window
  {
    auto now = std::chrono::steady_clock::now();
    if (lastFrameTimestamp.time_since_epoch().count() != 0) {
      float ms = std::chrono::duration<float, std::milli>(now - lastFrameTimestamp).count();
      frameTimes[frameTimeIdx] = ms;
      frameTimeIdx = (frameTimeIdx + 1) % kFrameHistLen;
    }
    lastFrameTimestamp = now;
  }

  ImGui::NewFrame();
  ImGuizmo::BeginFrame();

  // Loading overlay: show a fullscreen progress bar while the initial scene is loading.
  // The bar resets between phases (Textures -> Physics -> AS -> Finalizing) so users
  // don't stare at a 100% bar while the engine is still doing work.
  if (renderer) {
    const bool modelLoading = renderer->IsLoading();
    if (modelLoading) {
      ImGuiIO& io = ImGui::GetIO();
      // Suppress right-click while loading
      if (io.MouseDown[1])
        io.MouseDown[1] = false;

      const ImVec2 dispSize = io.DisplaySize;

      ImGui::SetNextWindowPos(ImVec2(0, 0));
      ImGui::SetNextWindowSize(dispSize);
      ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
          ImGuiWindowFlags_NoResize |
          ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_NoScrollbar |
          ImGuiWindowFlags_NoCollapse |
          ImGuiWindowFlags_NoSavedSettings |
          ImGuiWindowFlags_NoBringToFrontOnFocus |
          ImGuiWindowFlags_NoNav;

      if (ImGui::Begin("##LoadingOverlay", nullptr, flags)) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        // Center the progress elements
        const float barWidth = dispSize.x * 0.8f;
        const float barX = (dispSize.x - barWidth) * 0.5f;
        const float barY = dispSize.y * 0.45f;
        ImGui::SetCursorPos(ImVec2(barX, barY));
        ImGui::BeginGroup();

        // Phase-aware progress (resets between phases).
        float frac = 0.0f;
        auto phase = renderer->GetLoadingPhase();
        if (phase == Renderer::LoadingPhase::Textures) {
          const uint32_t scheduled = renderer->GetTextureTasksScheduled();
          const uint32_t completed = renderer->GetTextureTasksCompleted();
          frac = (scheduled > 0) ? (static_cast<float>(completed) / static_cast<float>(scheduled)) : 0.0f;
        } else if (phase == Renderer::LoadingPhase::AccelerationStructures) {
          frac = renderer->GetASBuildProgress();
        } else {
          frac = renderer->GetLoadingPhaseProgress();
        }
        ImGui::ProgressBar(frac, ImVec2(barWidth, 0.0f));
        ImGui::Dummy(ImVec2(0.0f, 10.0f));
        ImGui::SetCursorPosX(barX);
        ImGui::Text("Loading: %s", renderer->GetLoadingPhaseName());
        if (phase == Renderer::LoadingPhase::Textures) {
          const uint32_t scheduled = renderer->GetTextureTasksScheduled();
          const uint32_t completed = renderer->GetTextureTasksCompleted();
          ImGui::Text("Textures: %u/%u", completed, scheduled);
        } else if (phase == Renderer::LoadingPhase::AccelerationStructures) {
          const uint32_t done = renderer->GetASBuildItemsDone();
          const uint32_t total = renderer->GetASBuildItemsTotal();
          ImGui::Text("%s (%u/%u, %.1fs)", renderer->GetASBuildStage(), done, total, renderer->GetASBuildElapsedSeconds());
        }
        ImGui::EndGroup();
        ImGui::PopStyleVar();
      }
      ImGui::End();
      return;
    }
  }

  // --- Scene picker overlay (shown until a scene is chosen) ---
  if (scenePickerCallback && (!renderer || !renderer->IsLoading())) {
    scenePickerCallback();
    // Don't render the normal editor UI while picker is up; still allow render to flush.
    return;
  }

  // --- Main menu bar (File / View / Render / Debug) ---
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Exit", "Alt+F4")) { /* handled by OS */ }
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
      ImGui::MenuItem("Scene Browser",  nullptr, &panelState.showScene);
      ImGui::MenuItem("Inspector",      nullptr, &panelState.showInspector);
      ImGui::MenuItem("Performance",    nullptr, &panelState.showPerformance);
      ImGui::MenuItem("Audio Controls", nullptr, &panelState.showAudio);
      ImGui::Separator();
      ImGui::MenuItem("Debug Log",      nullptr, &toolbarState.showDebugPanel);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Render")) {
      ImGui::MenuItem("PBR Shading",   nullptr, &pbrEnabled);
      ImGui::Separator();
      ImGui::MenuItem("Wireframe",     nullptr, &toolbarState.showWireframe);
      ImGui::MenuItem("Ground Grid",   nullptr, &toolbarState.showGrid);
      ImGui::MenuItem("Physics Debug", nullptr, &toolbarState.showPhysDebug);
      ImGui::EndMenu();
    }
    // Right-aligned: engine name + mode
    {
      const char* modeLabel = pbrEnabled ? "  PBR  " : "  Flat  ";
      float x = ImGui::GetContentRegionMax().x
              - ImGui::CalcTextSize("VulkanView Engine").x
              - ImGui::CalcTextSize(modeLabel).x - 20.f;
      ImGui::SetCursorPosX(x);
      ImGui::TextDisabled("VulkanView Engine");
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.302f, 0.624f, 0.839f, 1.f));
      ImGui::Text("%s", modeLabel);
      ImGui::PopStyleColor();
    }
    ImGui::EndMainMenuBar();
  }

  // --- Toolbar (row of icon buttons just below the menu bar) ---
  DrawToolbar();

  // --- Full-screen DockSpace host window ---
  // Covers the area below menu bar + toolbar.
  {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float toolbarH = 32.f;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + toolbarH));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, viewport->WorkSize.y - toolbarH));
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGuiWindowFlags hostFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize   | ImGuiWindowFlags_NoMove      |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("##DockSpaceHost", nullptr, hostFlags);
    ImGui::PopStyleVar(3);
    ImGuiID dockID = ImGui::GetID("MainDockSpace");
    // Rebuild the default panel arrangement once per session.  Using a static bool (not
    // DockBuilderGetNode == nullptr) ensures the layout is applied even when an old imgui.ini
    // exists that might place panels outside the viewport.
    static bool layoutInitialized = false;
    if (!layoutInitialized) {
      BuildInitialDockLayout(dockID);
      layoutInitialized = true;
    }
    ImGui::DockSpace(dockID, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();
  }

  // --- Streaming status: small progress indicator in the upper-right ---
  // Once the scene is visible, textures may continue streaming to the GPU.
  // Show a compact progress bar in the top-right while there are still
  // outstanding texture tasks, and hide it once everything is fully loaded.
  if (renderer) {
    const uint32_t uploadTotal = renderer->GetUploadJobsTotal();
    const uint32_t uploadDone = renderer->GetUploadJobsCompleted();
    const bool modelLoading = renderer->IsLoading();
    const bool showASBuild = renderer->ShouldShowASBuildProgressInUI();

    // Acceleration structure build can happen after initial load completes.
    // If it takes a long time, show a compact progress window.
    if (!modelLoading && showASBuild) {
      ImGuiIO& io = ImGui::GetIO();
      const ImVec2 dispSize = io.DisplaySize;

      const float windowWidth = std::min(320.0f, dispSize.x * 0.42f);
      const float windowHeight = 90.0f;
      const ImVec2 winPos(dispSize.x - windowWidth - 10.0f, 10.0f);

      ImGui::SetNextWindowPos(winPos, ImGuiCond_Always);
      ImGui::SetNextWindowSize(ImVec2(windowWidth, windowHeight));
      ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize |
          ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_NoCollapse |
          ImGuiWindowFlags_NoSavedSettings;

      if (ImGui::Begin("##ASBuildStatus", nullptr, flags)) {
        ImGui::Text("Building acceleration structures...");
        const float asFrac = renderer->GetASBuildProgress();
        ImGui::ProgressBar(asFrac, ImVec2(-1.0f, 0.0f));
        const uint32_t done = renderer->GetASBuildItemsDone();
        const uint32_t total = renderer->GetASBuildItemsTotal();
        ImGui::Text("%s (%u/%u, %.1fs)",
                    renderer->GetASBuildStage(),
                    done,
                    total,
                    renderer->GetASBuildElapsedSeconds());
      }
      ImGui::End();
    }

    if (!modelLoading && uploadTotal > 0 && uploadDone < uploadTotal) {
      ImGuiIO& io = ImGui::GetIO();
      const ImVec2 dispSize = io.DisplaySize;

      const float windowWidth = std::min(260.0f, dispSize.x * 0.35f);
      const float windowHeight = 120.0f;
      // If the AS build status window is visible, offset streaming window below it.
      const float yBase = 10.0f + (showASBuild ? (90.0f + 10.0f) : 0.0f);
      const ImVec2 winPos(dispSize.x - windowWidth - 10.0f, yBase);

      ImGui::SetNextWindowPos(winPos, ImGuiCond_Always);
      ImGui::SetNextWindowSize(ImVec2(windowWidth, windowHeight));
      ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
          ImGuiWindowFlags_NoResize |
          ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_NoScrollbar |
          ImGuiWindowFlags_NoSavedSettings |
          ImGuiWindowFlags_NoCollapse;

      if (ImGui::Begin("##StreamingTextures", nullptr, flags)) {
        ImGui::TextUnformatted("Streaming textures to GPU");
        float frac = (uploadTotal > 0) ? (float) uploadDone / (float) uploadTotal : 0.0f;
        ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.0f));

        // Perf counters
        const double mbps = renderer->GetUploadThroughputMBps();
        const double avgMs = renderer->GetAverageUploadMs();
        const double totalMB = (double) renderer->GetBytesUploadedTotal() / (1024.0 * 1024.0);
        ImGui::Text("Throughput: %.1f MB/s", mbps);
        ImGui::SameLine();
        ImGui::Text("Avg upload: %.2f ms/tex", avgMs);
        ImGui::Text("Total uploaded: %.1f MB", totalMB);
      }
      ImGui::End();
    }
  }

  // --- Profiling window ---
  if (panelState.showPerformance) {
    // Compute stats from the ring buffer
    float sumMs = 0.f, minMs = FLT_MAX, maxMs = 0.f;
    int   validCount = 0;
    for (int i = 0; i < kFrameHistLen; ++i) {
      if (frameTimes[i] > 0.f) {
        sumMs += frameTimes[i];
        minMs  = std::min(minMs, frameTimes[i]);
        maxMs  = std::max(maxMs, frameTimes[i]);
        ++validCount;
      }
    }
    float avgMs  = (validCount > 0) ? sumMs / static_cast<float>(validCount) : 0.f;
    float curFps = (avgMs > 0.f) ? 1000.f / avgMs : 0.f;

    ImGui::SetNextWindowSize(ImVec2(300, 160), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Performance", &panelState.showPerformance)) {
      ImGui::Text("FPS: %.1f  |  Frame: %.2f ms", curFps, avgMs);
      ImGui::Text("Min: %.2f ms  Max: %.2f ms", minMs, maxMs);

      // Build a display buffer ordered oldest → newest
      float display[kFrameHistLen];
      for (int i = 0; i < kFrameHistLen; ++i)
        display[i] = frameTimes[(frameTimeIdx + i) % kFrameHistLen];

      char overlay[32];
      std::snprintf(overlay, sizeof(overlay), "%.1f fps", curFps);
      ImGui::PlotLines("##ft", display, kFrameHistLen, 0, overlay,
                       0.f, maxMs > 0.f ? maxMs * 1.5f : 50.f,
                       ImVec2(-1, 60));

      ImGui::Separator();
      auto& rdoc = RenderDocDebugSystem::GetInstance();
      if (rdoc.IsAvailable()) {
        if (ImGui::Button("Capture Frame (RenderDoc)")) {
          rdoc.TriggerCapture();
        }
      } else {
        ImGui::TextDisabled("RenderDoc not attached");
      }
    }
    ImGui::End();
  } // if panelState.showPerformance

  // --- ImGuizmo ViewManipulate (axis-alignment gizmo in lower-right corner) ---
  if (activeCamera) {
    ImGuiIO& io = ImGui::GetIO();
    const float gizmoSize = 128.f;
    const float gizmoX    = io.DisplaySize.x - gizmoSize - 10.f;
    const float gizmoY    = io.DisplaySize.y - gizmoSize - 10.f;

    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

    glm::mat4 viewMat = activeCamera->GetViewMatrix();
    float viewArr[16];
    memcpy(viewArr, glm::value_ptr(viewMat), sizeof(viewArr));

    glm::vec3 camPos = activeCamera->GetPosition();
    glm::vec3 camTgt = activeCamera->GetTarget();
    float camDist    = glm::length(camTgt - camPos);
    if (camDist < 0.001f) camDist = 1.f;

    ImGuizmo::ViewManipulate(viewArr, camDist,
                             ImVec2(gizmoX, gizmoY),
                             ImVec2(gizmoSize, gizmoSize),
                             0x10101080u);

    // If ViewManipulate changed the matrix, update the camera
    glm::mat4 newView;
    memcpy(glm::value_ptr(newView), viewArr, sizeof(viewArr));
    if (newView != viewMat) {
      glm::mat4 invView = glm::inverse(newView);
      glm::vec3 newPos  = glm::vec3(invView[3]);
      // Camera forward is -Z column of the inverse view matrix
      glm::vec3 forward = glm::normalize(-glm::vec3(invView[2]));
      glm::vec3 newTgt  = newPos + forward * camDist;

      auto* transform = activeCamera->GetOwner()->GetComponent<TransformComponent>();
      if (transform) transform->SetPosition(newPos);
      activeCamera->LookAt(newTgt);
    }
  }

  // Create HRTF Audio Control UI
  if (panelState.showAudio) {
  ImGui::Begin("HRTF Audio Controls", &panelState.showAudio);
  ImGui::Text("3D Audio Position Control");

  // Audio source selection
  ImGui::Separator();
  ImGui::Text("Audio Source Selection:");

  static bool useDebugPing = false;
  if (ImGui::Checkbox("Use Debug Ping (800Hz sine wave)", &useDebugPing)) {
    // Stop current audio
    if (audioSource&& audioSource
    
    ->
    IsPlaying()
    ) {
      audioSource->Stop();
    }
    if (debugPingSource&& debugPingSource
    
    ->
    IsPlaying()
    ) {
      debugPingSource->Stop();
    }
    std::cout << "Switched to " << (useDebugPing ? "debug ping" : "file audio") << " source" << std::endl;
  }

  // Display current audio source position
  ImGui::Text("Audio Source Position: (%.2f, %.2f, %.2f)", audioSourceX, audioSourceY, audioSourceZ);
  ImGui::Text("Current Source: %s", useDebugPing ? "Debug Ping (800Hz)" : "grass-step-right.wav");

  // Directional control buttons
  ImGui::Separator();
  ImGui::Text("Directional Controls:");

  // Get current active source
  AudioSource* currentSource = useDebugPing ? debugPingSource : audioSource;

  // Up button
  if (ImGui::Button("Up")) {
    audioSourceY += 0.5f;
    if (currentSource) {
      currentSource->SetPosition(audioSourceX, audioSourceY, audioSourceZ);
    }
    std::cout << (useDebugPing ? "Debug ping" : "Audio") << " moved up to (" << audioSourceX << ", " << audioSourceY << ", " << audioSourceZ << ")" << std::endl;
  }

  // Left and Right buttons on same line
  if (ImGui::Button("Left")) {
    audioSourceX -= 0.5f;
    if (currentSource) {
      currentSource->SetPosition(audioSourceX, audioSourceY, audioSourceZ);
    }
    std::cout << (useDebugPing ? "Debug ping" : "Audio") << " moved left to (" << audioSourceX << ", " << audioSourceY << ", " << audioSourceZ << ")" << std::endl;
  }
  ImGui::SameLine();
  if (ImGui::Button("Right")) {
    audioSourceX += 0.5f;
    if (currentSource) {
      currentSource->SetPosition(audioSourceX, audioSourceY, audioSourceZ);
    }
    std::cout << (useDebugPing ? "Debug ping" : "Audio") << " moved right to (" << audioSourceX << ", " << audioSourceY << ", " << audioSourceZ << ")" << std::endl;
  }

  // Down button
  if (ImGui::Button("Down")) {
    audioSourceY -= 0.5f;
    if (currentSource) {
      currentSource->SetPosition(audioSourceX, audioSourceY, audioSourceZ);
    }
    std::cout << (useDebugPing ? "Debug ping" : "Audio") << " moved down to (" << audioSourceX << ", " << audioSourceY << ", " << audioSourceZ << ")" << std::endl;
  }

  // Audio playback controls
  ImGui::Separator();
  ImGui::Text("Playback Controls:");

  // Play button
  if (ImGui::Button("Play")) {
    if (currentSource) {
      currentSource->Play();
      if (audioSystem) {
        audioSystem->FlushOutput();
      }
      if (useDebugPing) {
        std::cout << "Started playing debug ping (800Hz sine wave) with HRTF processing" << std::endl;
      } else {
        std::cout << "Started playing grass-step-right.wav with HRTF processing" << std::endl;
      }
    } else {
      std::cout << "No audio source available - audio system not initialized" << std::endl;
    }
  }
  ImGui::SameLine();

  // Stop button
  if (ImGui::Button("Stop")) {
    if (currentSource) {
      currentSource->Stop();
      if (useDebugPing) {
        std::cout << "Stopped debug ping playback" << std::endl;
      } else {
        std::cout << "Stopped audio playback" << std::endl;
      }
    }
  }

  // Additional info
  ImGui::Separator();
  if (audioSystem&& audioSystem
  
  ->
  IsHRTFEnabled()
  ) {
    ImGui::Text("HRTF Processing: ENABLED");
    ImGui::Text("Use directional buttons to move the audio source in 3D space");
    ImGui::Text("You should hear the audio move around you!");

    // HRTF Processing Mode: GPU only (checkbox removed)
    ImGui::Separator();
    ImGui::Text("HRTF Processing Mode:");
    ImGui::Text("Current Mode: Vulkan shader processing (GPU)");
  }
  else {
    ImGui::Text("HRTF Processing: DISABLED");
  }

  // Ball Debugging Controls
  ImGui::Separator();
  ImGui::Text("Ball Debugging Controls:");

  if (ImGui::Checkbox("Ball-Only Rendering", &ballOnlyRenderingEnabled)) {
    std::cout << "Ball-only rendering " << (ballOnlyRenderingEnabled ? "enabled" : "disabled") << std::endl;
  }
  ImGui::SameLine();
  if (ImGui::Button("?##BallOnlyHelp")) {
    // Help tooltip will be shown on hover
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("When enabled, only balls will be rendered.\nAll other geometry (bistro scene) will be hidden.");
  }

  if (ImGui::Checkbox("Camera Track Ball", &cameraTrackingEnabled)) {
    std::cout << "Camera tracking " << (cameraTrackingEnabled ? "enabled" : "disabled") << std::endl;
  }
  ImGui::SameLine();
  if (ImGui::Button("?##CameraTrackHelp")) {
    // Help tooltip will be shown on hover
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("When enabled, camera will automatically\nfollow and look at the ball.");
  }

  // Status display
  if (ballOnlyRenderingEnabled) {
    ImGui::Text("Status: Only balls are being rendered");
  } else {
    ImGui::Text("Status: All geometry is being rendered");
  }

  if (cameraTrackingEnabled) {
    ImGui::Text("Camera: Tracking ball automatically");
  } else {
    ImGui::Text("Camera: Manual control (WASD + mouse)");
  }

  // Texture loading progress
  if (renderer) {
    const uint32_t scheduled = renderer->GetTextureTasksScheduled();
    const uint32_t completed = renderer->GetTextureTasksCompleted();
    if (scheduled > 0 && completed < scheduled) {
      ImGui::Separator();
      float frac = scheduled ? (float) completed / (float) scheduled : 1.0f;
      ImGui::Text("Loading textures: %u / %u", completed, scheduled);
      ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0.0f));
      ImGui::Text("You can continue interacting while textures stream in...");
    }
  }

  ImGui::End();
  } // if panelState.showAudio

  // --- Debug output panel ---
  DrawDebugPanel();

  // --- Engine-level settings panels (simulation, UpdatePublisher controls, etc.) ---
  if (engineSettingsCallback)
    engineSettingsCallback();

  // --- Scene-specific UI panels (browser, inspector, etc.) ---
  if (sceneUICallback)
    sceneUICallback();
}

void ImGuiSystem::Render(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex) {
  if (!initialized) {
    return;
  }

  // End the frame and prepare for rendering
  ImGui::Render();

  // Update vertex and index buffers for this frame
  updateBuffers(frameIndex);

  // Record rendering commands
  ImDrawData* drawData = ImGui::GetDrawData();
  if (!drawData || drawData->CmdListsCount == 0) {
    return;
  }

  try {
    // Bind the pipeline
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

    // Set viewport
    vk::Viewport viewport;
    viewport.width = ImGui::GetIO().DisplaySize.x;
    viewport.height = ImGui::GetIO().DisplaySize.y;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    commandBuffer.setViewport(0, {viewport});

    // Set push constants
    struct PushConstBlock {
      float scale[2];
      float translate[2];
    } pushConstBlock{};

    pushConstBlock.scale[0] = 2.0f / ImGui::GetIO().DisplaySize.x;
    pushConstBlock.scale[1] = 2.0f / ImGui::GetIO().DisplaySize.y;
    pushConstBlock.translate[0] = -1.0f;
    pushConstBlock.translate[1] = -1.0f;

    commandBuffer.pushConstants<PushConstBlock>(*pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pushConstBlock);

    // Bind vertex and index buffers for this frame
    commandBuffer.bindVertexBuffers(0, *vertexBuffers[frameIndex], vk::DeviceSize{0});
    commandBuffer.bindIndexBuffer(*indexBuffers[frameIndex], 0, vk::IndexType::eUint16);

    // Render command lists
    int vertexOffset = 0;
    int indexOffset = 0;

    for (int i = 0; i < drawData->CmdListsCount; i++) {
      const ImDrawList* cmdList = drawData->CmdLists[i];

      for (int j = 0; j < cmdList->CmdBuffer.Size; j++) {
        const ImDrawCmd* pcmd = &cmdList->CmdBuffer[j];

        // Set scissor rectangle
        vk::Rect2D scissor;
        scissor.offset.x = std::max(static_cast<int32_t>(pcmd->ClipRect.x), 0);
        scissor.offset.y = std::max(static_cast<int32_t>(pcmd->ClipRect.y), 0);
        scissor.extent.width = static_cast<uint32_t>(pcmd->ClipRect.z - pcmd->ClipRect.x);
        scissor.extent.height = static_cast<uint32_t>(pcmd->ClipRect.w - pcmd->ClipRect.y);
        commandBuffer.setScissor(0, {scissor});

        // Bind descriptor set (font texture)
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, {*descriptorSet}, {});

        // Draw
        commandBuffer.drawIndexed(pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
        indexOffset += pcmd->ElemCount;
      }

      vertexOffset += cmdList->VtxBuffer.Size;
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to render ImGui: " << e.what() << std::endl;
  }
}

void ImGuiSystem::HandleMouse(float x, float y, uint32_t buttons) {
  if (!initialized) {
    return;
  }

  ImGuiIO& io = ImGui::GetIO();

  // Update mouse position
  io.MousePos = ImVec2(x, y);

  // Update mouse buttons
  io.MouseDown[0] = (buttons & 0x01) != 0; // Left button
  io.MouseDown[1] = (buttons & 0x02) != 0; // Right button
  io.MouseDown[2] = (buttons & 0x04) != 0; // Middle button
}

void ImGuiSystem::HandleKeyboard(uint32_t key, bool pressed) {
  if (!initialized) {
    return;
  }

  ImGuiIO& io = ImGui::GetIO();

  // ImGui 1.87+ replaced KeysDown[] with AddKeyEvent().
  // Map a subset of GLFW key codes to ImGuiKey values.
  auto glfwToImGuiKey = [](uint32_t k) -> ImGuiKey {
    // Printable ASCII range (space..tilde) maps directly via ImGuiKey_A etc.
    if (k >= 32 && k <= 90)  return static_cast<ImGuiKey>(ImGuiKey_A + (k - 65));
    switch (k) {
      case 256: return ImGuiKey_Escape;
      case 257: return ImGuiKey_Enter;
      case 258: return ImGuiKey_Tab;
      case 259: return ImGuiKey_Backspace;
      case 260: return ImGuiKey_Insert;
      case 261: return ImGuiKey_Delete;
      case 262: return ImGuiKey_RightArrow;
      case 263: return ImGuiKey_LeftArrow;
      case 264: return ImGuiKey_DownArrow;
      case 265: return ImGuiKey_UpArrow;
      case 266: return ImGuiKey_PageUp;
      case 267: return ImGuiKey_PageDown;
      case 268: return ImGuiKey_Home;
      case 269: return ImGuiKey_End;
      case 290: return ImGuiKey_F1;  case 291: return ImGuiKey_F2;
      case 292: return ImGuiKey_F3;  case 293: return ImGuiKey_F4;
      case 294: return ImGuiKey_F5;  case 295: return ImGuiKey_F6;
      case 296: return ImGuiKey_F7;  case 297: return ImGuiKey_F8;
      case 298: return ImGuiKey_F9;  case 299: return ImGuiKey_F10;
      case 300: return ImGuiKey_F11; case 301: return ImGuiKey_F12;
      case 340: return ImGuiKey_LeftShift;
      case 341: return ImGuiKey_LeftCtrl;
      case 342: return ImGuiKey_LeftAlt;
      case 343: return ImGuiKey_LeftSuper;
      case 344: return ImGuiKey_RightShift;
      case 345: return ImGuiKey_RightCtrl;
      case 346: return ImGuiKey_RightAlt;
      case 347: return ImGuiKey_RightSuper;
      default:  return ImGuiKey_None;
    }
  };

  ImGuiKey imkey = glfwToImGuiKey(key);
  if (imkey != ImGuiKey_None)
    io.AddKeyEvent(imkey, pressed);

  // Keep modifier state in sync
  io.AddKeyEvent(ImGuiMod_Ctrl,  io.KeyCtrl);
  io.AddKeyEvent(ImGuiMod_Shift, io.KeyShift);
  io.AddKeyEvent(ImGuiMod_Alt,   io.KeyAlt);
  io.AddKeyEvent(ImGuiMod_Super, io.KeySuper);
}

void ImGuiSystem::HandleChar(uint32_t c) {
  if (!initialized) {
    return;
  }

  ImGuiIO& io = ImGui::GetIO();
  io.AddInputCharacter(c);
}

void ImGuiSystem::HandleResize(uint32_t width, uint32_t height) {
  if (!initialized) {
    return;
  }

  this->width = width;
  this->height = height;

  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
}

bool ImGuiSystem::WantCaptureKeyboard() const {
  if (!initialized) {
    return false;
  }

  return ImGui::GetIO().WantCaptureKeyboard;
}

bool ImGuiSystem::WantCaptureMouse() const {
  if (!initialized) {
    return false;
  }

  return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiSystem::createResources() {
  // Create all Vulkan resources needed for ImGui rendering
  if (!createFontTexture()) {
    return false;
  }

  if (!createDescriptorSetLayout()) {
    return false;
  }

  if (!createDescriptorPool()) {
    return false;
  }

  if (!createDescriptorSet()) {
    return false;
  }

  if (!createPipelineLayout()) {
    return false;
  }

  if (!createPipeline()) {
    return false;
  }

  return true;
}

bool ImGuiSystem::createFontTexture() {
  // Get font texture from ImGui
  ImGuiIO& io = ImGui::GetIO();
  unsigned char* fontData;
  int texWidth, texHeight;
  io.Fonts->GetTexDataAsRGBA32(&fontData, &texWidth, &texHeight);
  vk::DeviceSize uploadSize = texWidth * texHeight * 4 * sizeof(char);

  try {
    // Create the font image
    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = vk::Format::eR8G8B8A8Unorm;
    imageInfo.extent.width = static_cast<uint32_t>(texWidth);
    imageInfo.extent.height = static_cast<uint32_t>(texHeight);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;

    const vk::raii::Device& device = renderer->GetRaiiDevice();
    fontImage = vk::raii::Image(device, imageInfo);

    // Allocate memory for the image
    vk::MemoryRequirements memRequirements = fontImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = renderer->FindMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

    fontMemory = vk::raii::DeviceMemory(device, allocInfo);
    fontImage.bindMemory(*fontMemory, 0);

    // Create a staging buffer for uploading the font data
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.size = uploadSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;

    vk::raii::Buffer stagingBuffer(device, bufferInfo);

    vk::MemoryRequirements stagingMemRequirements = stagingBuffer.getMemoryRequirements();

    vk::MemoryAllocateInfo stagingAllocInfo;
    stagingAllocInfo.allocationSize = stagingMemRequirements.size;
    stagingAllocInfo.memoryTypeIndex = renderer->FindMemoryType(stagingMemRequirements.memoryTypeBits,
                                                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    vk::raii::DeviceMemory stagingBufferMemory(device, stagingAllocInfo);
    stagingBuffer.bindMemory(*stagingBufferMemory, 0);

    // Copy font data to staging buffer
    void* data = stagingBufferMemory.mapMemory(0, uploadSize);
    memcpy(data, fontData, uploadSize);
    stagingBufferMemory.unmapMemory();

    // Transition image layout and copy data
    renderer->TransitionImageLayout(*fontImage,
                                    vk::Format::eR8G8B8A8Unorm,
                                    vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::eTransferDstOptimal);
    renderer->CopyBufferToImage(*stagingBuffer,
                                *fontImage,
                                static_cast<uint32_t>(texWidth),
                                static_cast<uint32_t>(texHeight));
    renderer->TransitionImageLayout(*fontImage,
                                    vk::Format::eR8G8B8A8Unorm,
                                    vk::ImageLayout::eTransferDstOptimal,
                                    vk::ImageLayout::eShaderReadOnlyOptimal);

    // Staging buffer and memory will be automatically cleaned up by RAII

    // Create image view
    vk::ImageViewCreateInfo viewInfo;
    viewInfo.image = *fontImage;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = vk::Format::eR8G8B8A8Unorm;
    viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    fontView = vk::raii::ImageView(device, viewInfo);

    // Create sampler
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = vk::CompareOp::eAlways;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    fontSampler = vk::raii::Sampler(device, samplerInfo);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create font texture: " << e.what() << std::endl;
    return false;
  }
}

bool ImGuiSystem::createDescriptorSetLayout() {
  try {
    vk::DescriptorSetLayoutBinding binding;
    binding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    binding.descriptorCount = 1;
    binding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    binding.binding = 0;

    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    const vk::raii::Device& device = renderer->GetRaiiDevice();
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create descriptor set layout: " << e.what() << std::endl;
    return false;
  }
}

bool ImGuiSystem::createDescriptorPool() {
  try {
    vk::DescriptorPoolSize poolSize;
    poolSize.type = vk::DescriptorType::eCombinedImageSampler;
    poolSize.descriptorCount = 1;

    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    const vk::raii::Device& device = renderer->GetRaiiDevice();
    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create descriptor pool: " << e.what() << std::endl;
    return false;
  }
}

bool ImGuiSystem::createDescriptorSet() {
  try {
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &(*descriptorSetLayout);

    const vk::raii::Device& device = renderer->GetRaiiDevice();
    vk::raii::DescriptorSets descriptorSets(device, allocInfo);
    descriptorSet = std::move(descriptorSets[0]); // Store the first (and only) descriptor set
    std::cout << "ImGui created descriptor set with handle: " << *descriptorSet << std::endl;

    // Update descriptor set
    vk::DescriptorImageInfo imageInfo;
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = *fontView;
    imageInfo.sampler = *fontSampler;

    vk::WriteDescriptorSet writeSet;
    writeSet.dstSet = *descriptorSet;
    writeSet.descriptorCount = 1;
    writeSet.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writeSet.pImageInfo = &imageInfo;
    writeSet.dstBinding = 0;

    device.updateDescriptorSets({writeSet}, {});

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create descriptor set: " << e.what() << std::endl;
    return false;
  }
}

bool ImGuiSystem::createPipelineLayout() {
  try {
    // Push constant range for the transformation matrix
    vk::PushConstantRange pushConstantRange;
    pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eVertex;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(float) * 4; // 2 floats for scale, 2 floats for translate

    // Create pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &(*descriptorSetLayout);
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    const vk::raii::Device& device = renderer->GetRaiiDevice();
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create pipeline layout: " << e.what() << std::endl;
    return false;
  }
}

bool ImGuiSystem::createPipeline() {
  try {
    // Load shaders
    vk::raii::ShaderModule shaderModule = renderer->CreateShaderModule("shaders/imgui.spv");

    // Shader stage creation
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = *shaderModule;
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = *shaderModule;
    fragShaderStageInfo.pName = "main";

    std::array shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

    // Vertex input
    vk::VertexInputBindingDescription bindingDescription;
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(ImDrawVert);
    bindingDescription.inputRate = vk::VertexInputRate::eVertex;

    std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions;
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[0].offset = offsetof(ImDrawVert, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[1].offset = offsetof(ImDrawVert, uv);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = vk::Format::eR8G8B8A8Unorm;
    attributeDescriptions[2].offset = offsetof(ImDrawVert, col);

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    // Input assembly
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Viewport and scissor
    vk::PipelineViewportStateCreateInfo viewportState;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    viewportState.pViewports = nullptr; // Dynamic state
    viewportState.pScissors = nullptr; // Dynamic state

    // Rasterization
    vk::PipelineRasterizationStateCreateInfo rasterizer;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;

    // Multisampling
    vk::PipelineMultisampleStateCreateInfo multisampling;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

    // Depth and stencil testing
    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = vk::CompareOp::eLessOrEqual;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // Color blending
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;

    vk::PipelineColorBlendStateCreateInfo colorBlending;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // Dynamic state
    std::vector<vk::DynamicState> dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    vk::Format depthFormat = renderer->findDepthFormat();
    // Create the graphics pipeline with dynamic rendering
    vk::PipelineRenderingCreateInfo renderingInfo;
    renderingInfo.colorAttachmentCount = 1;
    vk::Format colorFormat = renderer->GetSwapChainImageFormat(); // Get the actual swapchain format
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    renderingInfo.depthAttachmentFormat = depthFormat;

    vk::GraphicsPipelineCreateInfo pipelineInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = *pipelineLayout;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.basePipelineHandle = nullptr;

    const vk::raii::Device& device = renderer->GetRaiiDevice();
    pipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create graphics pipeline: " << e.what() << std::endl;
    return false;
  }
}

void ImGuiSystem::updateBuffers(uint32_t frameIndex) {
  ImDrawData* drawData = ImGui::GetDrawData();
  if (!drawData || drawData->CmdListsCount == 0) {
    return;
  }

  try {
    const vk::raii::Device& device = renderer->GetRaiiDevice();

    // Calculate required buffer sizes
    vk::DeviceSize vertexBufferSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
    vk::DeviceSize indexBufferSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

    // Resize buffers if needed for this frame
    if (frameIndex >= vertexCounts.size())
      return; // Safety

    if (static_cast<uint32_t>(drawData->TotalVtxCount) > vertexCounts[frameIndex]) {
      // Clean up old buffer
      vertexBuffers[frameIndex] = vk::raii::Buffer(nullptr);
      vertexBufferMemories[frameIndex] = vk::raii::DeviceMemory(nullptr);

      // Create new vertex buffer
      vk::BufferCreateInfo bufferInfo;
      bufferInfo.size = vertexBufferSize;
      bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
      bufferInfo.sharingMode = vk::SharingMode::eExclusive;

      vertexBuffers[frameIndex] = vk::raii::Buffer(device, bufferInfo);

      vk::MemoryRequirements memRequirements = vertexBuffers[frameIndex].getMemoryRequirements();

      vk::MemoryAllocateInfo allocInfo;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = renderer->FindMemoryType(memRequirements.memoryTypeBits,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

      vertexBufferMemories[frameIndex] = vk::raii::DeviceMemory(device, allocInfo);
      vertexBuffers[frameIndex].bindMemory(*vertexBufferMemories[frameIndex], 0);
      vertexCounts[frameIndex] = drawData->TotalVtxCount;
    }

    if (static_cast<uint32_t>(drawData->TotalIdxCount) > indexCounts[frameIndex]) {
      // Clean up old buffer
      indexBuffers[frameIndex] = vk::raii::Buffer(nullptr);
      indexBufferMemories[frameIndex] = vk::raii::DeviceMemory(nullptr);

      // Create new index buffer
      vk::BufferCreateInfo bufferInfo;
      bufferInfo.size = indexBufferSize;
      bufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer;
      bufferInfo.sharingMode = vk::SharingMode::eExclusive;

      indexBuffers[frameIndex] = vk::raii::Buffer(device, bufferInfo);

      vk::MemoryRequirements memRequirements = indexBuffers[frameIndex].getMemoryRequirements();

      vk::MemoryAllocateInfo allocInfo;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = renderer->FindMemoryType(memRequirements.memoryTypeBits,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

      indexBufferMemories[frameIndex] = vk::raii::DeviceMemory(device, allocInfo);
      indexBuffers[frameIndex].bindMemory(*indexBufferMemories[frameIndex], 0);
      indexCounts[frameIndex] = drawData->TotalIdxCount;
    }

    // Upload data to buffers for this frame (only if we have data to upload)
    if (drawData->TotalVtxCount > 0 && drawData->TotalIdxCount > 0) {
      void* vtxMappedMemory = vertexBufferMemories[frameIndex].mapMemory(0, vertexBufferSize);
      void* idxMappedMemory = indexBufferMemories[frameIndex].mapMemory(0, indexBufferSize);

      ImDrawVert* vtxDst = static_cast<ImDrawVert *>(vtxMappedMemory);
      ImDrawIdx* idxDst = static_cast<ImDrawIdx *>(idxMappedMemory);

      for (int n = 0; n < drawData->CmdListsCount; n++) {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        memcpy(vtxDst, cmdList->VtxBuffer.Data, cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idxDst, cmdList->IdxBuffer.Data, cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtxDst += cmdList->VtxBuffer.Size;
        idxDst += cmdList->IdxBuffer.Size;
      }

      vertexBufferMemories[frameIndex].unmapMemory();
      indexBufferMemories[frameIndex].unmapMemory();
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to update buffers: " << e.what() << std::endl;
  }
}