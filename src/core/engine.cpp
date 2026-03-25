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
#include "engine.h"
#include "mesh_component.h"
#include "scene_loading.h"
#include "imgui_system.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <execinfo.h>
#include <iostream>
#include <random>
#include <ranges>
#include <stdexcept>
#include <sstream>

// ---------------------------------------------------------------------------
// Crash signal handler — prints a stack trace so we can see what crashed
// ---------------------------------------------------------------------------
static void crashSignalHandler(int sig) {
    const char* sigName = (sig == SIGSEGV) ? "SIGSEGV" : (sig == SIGABRT) ? "SIGABRT" : "SIGNAL";
    std::cerr << "\n[CRASH] Signal " << sigName << " (" << sig << ") received!\n";
    void* frames[64];
    int n = backtrace(frames, 64);
    char** syms = backtrace_symbols(frames, n);
    std::cerr << "Stack trace (" << n << " frames):\n";
    for (int i = 0; i < n; ++i)
        std::cerr << "  [" << i << "] " << (syms ? syms[i] : "???") << "\n";
    free(syms);
    std::cerr << std::flush;
    // Re-raise with default handler so we get a core dump
    signal(sig, SIG_DFL);
    raise(sig);
}

static void installCrashHandlers() {
    signal(SIGSEGV, crashSignalHandler);
    signal(SIGABRT, crashSignalHandler);
    signal(SIGFPE,  crashSignalHandler);
    signal(SIGILL,  crashSignalHandler);
}

// This implementation corresponds to the Engine_Architecture chapter in the tutorial:
// @see en/Building_a_Simple_Engine/Engine_Architecture/02_architectural_patterns.adoc

Engine::Engine() : resourceManager(std::make_unique<ResourceManager>()) {
}

bool Engine::IsMainThread() const {
  return std::this_thread::get_id() == mainThreadId;
}

void Engine::ProcessPendingEntityRemovals() {
  std::vector<std::string> names; {
    std::lock_guard<std::mutex> lk(pendingEntityRemovalsMutex);
    if (pendingEntityRemovalNames.empty())
      return;
    names.swap(pendingEntityRemovalNames);
  }

  // Process on the main thread only (safety)
  if (!IsMainThread()) {
    // Put them back; we'll retry next main-thread tick
    std::lock_guard<std::mutex> lk(pendingEntityRemovalsMutex);
    pendingEntityRemovalNames.insert(pendingEntityRemovalNames.end(), names.begin(), names.end());
    return;
  }

  // Apply removals using the normal API (which takes the appropriate locks).
  for (const auto& name : names) {
    (void) RemoveEntity(name);
  }
}

Engine::~Engine() {
  Cleanup();
}

bool Engine::Initialize(const std::string& appName, int width, int height, bool enableValidationLayers) {
  // Create platform
#if defined(PLATFORM_ANDROID)
  // For Android, the platform is created with the android_app
  // This will be handled in the android_main function
  return false;
#else
  // Record main thread identity for deferring destructive operations from background threads
  mainThreadId = std::this_thread::get_id();

  platform = CreatePlatform();
  if (!platform->Initialize(appName, width, height)) {
    return false;
  }

  // Set resize callback
  platform->SetResizeCallback([this](int width, int height) {
    HandleResize(width, height);
  });

  // Set mouse callback
  platform->SetMouseCallback([this](float x, float y, uint32_t buttons) {
    handleMouseInput(x, y, buttons);
  });

  // Set keyboard callback
  platform->SetKeyboardCallback([this](uint32_t key, bool pressed) {
    handleKeyInput(key, pressed);
  });

  // Set char callback
  platform->SetCharCallback([this](uint32_t c) {
    if (imguiSystem) {
      imguiSystem->HandleChar(c);
    }
  });

  // Create renderer
  renderer = std::make_unique<Renderer>(platform.get());
  if (!renderer->Initialize(appName, enableValidationLayers)) {
    return false;
  }

  try {
    // Model loader via constructor; also wire into renderer
    modelLoader = std::make_unique<ModelLoader>(renderer.get());
    renderer->SetModelLoader(modelLoader.get());

    // Audio system via constructor
    audioSystem = std::make_unique<AudioSystem>(this, renderer.get());

    physicsSystem = std::make_unique<PhysicsSystem>();

    // UpdatePublisher — default 100 Hz fixed step (0.01 s)
    updatePublisher = std::make_unique<UpdatePublisher>(0.01f);

    // ImGui via constructor, then connect audio system
    imguiSystem = std::make_unique<ImGuiSystem>(renderer.get(), width, height);
    imguiSystem->SetAudioSystem(audioSystem.get());
  } catch (const std::exception& e) {
    std::cerr << "Subsystem initialization failed: " << e.what() << std::endl;
    return false;
  }

  // Register the "Simulation" settings panel — lets the user adjust the
  // UpdatePublisher fixed timestep (and see derived Hz) at runtime.
  imguiSystem->SetEngineSettingsCallback([this]() {
    if (!updatePublisher) return;
    ImGui::SetNextWindowSize(ImVec2(300, 130), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Simulation")) {
      float ts = updatePublisher->GetFixedTimestep();
      ImGui::TextUnformatted("UpdatePublisher Fixed Timestep");
      ImGui::Separator();
      ImGui::Spacing();
      if (ImGui::SliderFloat("Step (s)", &ts, 0.002f, 0.05f, "%.4f")) {
        updatePublisher->SetFixedTimestep(ts);
      }
      ImGui::Text("  \xe2\x86\x92 %.0f Hz", ts > 0.f ? 1.f / ts : 0.f);
      ImGui::Spacing();
      ImGui::TextDisabled("Default: 0.01 s  (100 Hz)");
    }
    ImGui::End();
  });

  // Generate ball material properties once at load time
  GenerateBallMaterial();

  // Initialize physics scaling system
  InitializePhysicsScaling();

  // ------------------------------------------------------------------
  // Wire ALL per-frame logic into UpdatePublisher as core subscribers.
  // These run every frame for the engine's lifetime (not cleared on
  // ClearScene).  Order within each phase is insertion order.
  // ------------------------------------------------------------------

  // FixedUpdate — physics at 60 Hz (matches PhysX FIXED_STEP)
  updatePublisher->SubscribeCoreFixedUpdate([this](float fixedDt) {
    if (!physicsSystem) return;
    if (activeCamera)
      physicsSystem->SetCameraPosition(activeCamera->GetPosition());
    physicsSystem->Step(fixedDt);
  });

  // Update — audio system + all active entity updates
  updatePublisher->SubscribeCoreUpdate([this](float dt) {
    auto dtMs = std::chrono::milliseconds(static_cast<long long>(dt * 1000.f));
    if (audioSystem) audioSystem->Update(dtMs);

    std::vector<Entity*> snapshot;
    {
      std::shared_lock<std::shared_mutex> lk(entitiesMutex);
      snapshot.reserve(entities.size());
      for (auto& uptr : entities) snapshot.push_back(uptr.get());
    }
    for (Entity* e : snapshot) {
      if (e && e->IsActive()) e->Update(dtMs);
    }
  });

  // LateUpdate — camera controls (after physics has moved entities)
  updatePublisher->SubscribeCoreLateUpdate([this](float dt) {
    if (activeCamera)
      UpdateCameraControls(std::chrono::milliseconds(static_cast<long long>(dt * 1000.f)));
  });

  initialized = true;
  return true;
#endif
}

void Engine::Run() {
  if (!initialized) {
    throw std::runtime_error("Engine not initialized");
  }

  installCrashHandlers();
  running = true;

  // Main loop
  while (running) {
    // Process platform events
    if (!platform->ProcessEvents()) {
      running = false;
      break;
    }

    // Calculate delta time
    deltaTimeMs = CalculateDeltaTimeMs();

    // Update frame counter and FPS
    frameCount++;
    fpsUpdateTimer += deltaTimeMs.count() * 0.001f;

    // Update window title with FPS and frame time every second
    if (fpsUpdateTimer >= 1.0f) {
      uint64_t framesSinceLastUpdate = frameCount - lastFPSUpdateFrame;
      double avgMs = 0.0;
      if (framesSinceLastUpdate > 0 && fpsUpdateTimer > 0.0f) {
        currentFPS = static_cast<float>(static_cast<double>(framesSinceLastUpdate) / static_cast<double>(fpsUpdateTimer));
        avgMs = (fpsUpdateTimer / static_cast<double>(framesSinceLastUpdate)) * 1000.0;
      } else {
        // Avoid divide-by-zero; keep previous FPS and estimate avgMs from last delta
        currentFPS = std::max(currentFPS, 1.0f);
        avgMs = static_cast<double>(deltaTimeMs.count());
      }

      // Update window title with frame count, FPS, and frame time
      std::string title = "Simple Engine - Frame: " + std::to_string(frameCount) +
          " | FPS: " + std::to_string(static_cast<int>(currentFPS)) +
          " | ms: " + std::to_string(static_cast<int>(avgMs));
      platform->SetWindowTitle(title);

      // Reset timer and frame counter for next update
      fpsUpdateTimer = 0.0f;
      lastFPSUpdateFrame = frameCount;
    }

    // Drain any scene loader queued by the picker last frame.
    // WaitIdle ensures no GPU work is in flight before scene resources are created.
    if (pendingSceneLoader) {
      if (renderer) renderer->WaitIdle();
      auto loader = std::move(pendingSceneLoader);
      pendingSceneLoader = nullptr;
      ImGuiSystem::DebugLog("[Engine] Loading scene...");
      try {
        loader();
        ImGuiSystem::DebugLog("[Engine] Scene loaded OK.");
      } catch (const std::exception& ex) {
        std::string msg = std::string("[Engine] SCENE LOAD EXCEPTION: ") + ex.what();
        std::cerr << msg << "\n";
        ImGuiSystem::DebugLog(msg);
      } catch (...) {
        std::cerr << "[Engine] SCENE LOAD UNKNOWN EXCEPTION\n";
        ImGuiSystem::DebugLog("[Engine] SCENE LOAD UNKNOWN EXCEPTION");
      }
    }

    // Update
    Update(deltaTimeMs);

    // Render
    Render();
  }
}

void Engine::Cleanup() {
  if (initialized) {
    // Wait for the device to be idle before cleaning up
    if (renderer) {
      renderer->WaitIdle();
    }

    // Clear entities
    {
      std::unique_lock<std::shared_mutex> lk(entitiesMutex);
      entities.clear();
      entityMap.clear();
    }

    // Stop the update loop first so no callbacks fire during teardown
    updatePublisher.reset();

    // Clean up subsystems in reverse order of creation
    imguiSystem.reset();
    physicsSystem.reset();
    audioSystem.reset();
    modelLoader.reset();
    renderer.reset();
    platform.reset();

    initialized = false;
  }
}

void Engine::ClearScene() {
  ImGuiSystem::DebugLog("[Engine] ClearScene() start");

  // Clear scene-level callbacks FIRST so old closures (which capture entity/WaterSystem
  // pointers) stop firing before we free anything.
  if (imguiSystem) {
    imguiSystem->SetSceneUI(nullptr);
  }
  if (updatePublisher) {
    updatePublisher->ClearSceneCallbacks();
  }

  // Detach waterSystem from renderer so it won't be rendered/updated while we reset.
  // (The WaterSystem object itself is owned by the scene closure; clearing the UI
  //  callback releases the closure which destroys the WaterSystem.)
  if (renderer) {
    renderer->waterSystem = nullptr;
  }

  // Remove all entities (releases component resources on the main thread).
  {
    std::unique_lock<std::shared_mutex> lk(entitiesMutex);
    entities.clear();
    entityMap.clear();
  }
  activeCamera = nullptr;

  // Reset the physics scene: removes all actors, recreates a fresh PxScene.
  if (physicsSystem) {
    physicsSystem->ClearActors();
  }

  ImGuiSystem::DebugLog("[Engine] ClearScene() done");
}

Entity* Engine::CreateEntity(const std::string& name) {
  std::unique_lock<std::shared_mutex> lk(entitiesMutex);
  // Always allow duplicate names; map stores a representative entity
  // Create the entity
  auto entity = std::make_unique<Entity>(name);
  // Add to the vector and map
  entities.push_back(std::move(entity));
  Entity* rawPtr = entities.back().get();
  // Update the map to point to the most recently created entity with this name
  entityMap[name] = rawPtr;

  return rawPtr;
}

Entity* Engine::GetEntity(const std::string& name) {
  std::shared_lock<std::shared_mutex> lk(entitiesMutex);
  auto it = entityMap.find(name);
  if (it != entityMap.end()) {
    return it->second;
  }
  return nullptr;
}

bool Engine::RemoveEntity(Entity* entity) {
  if (!entity) {
    return false;
  }

  // If called from a background thread, defer removal to avoid deleting entities
  // while the render thread may be iterating a snapshot.
  if (!IsMainThread()) {
    std::lock_guard<std::mutex> lk(pendingEntityRemovalsMutex);
    pendingEntityRemovalNames.push_back(entity->GetName());
    return true;
  }

  std::unique_lock<std::shared_mutex> lk(entitiesMutex);

  // Remember the name before erasing ownership
  std::string name = entity->GetName();

  // Find the entity in the vector
  auto it = std::ranges::find_if(entities,
                                 [entity](const std::unique_ptr<Entity>& e) {
                                   return e.get() == entity;
                                 });

  if (it != entities.end()) {
    // Remove from the vector (ownership)
    entities.erase(it);

    // Update the map: point to another entity with the same name if one exists
    auto remainingIt = std::ranges::find_if(entities,
                                            [&name](const std::unique_ptr<Entity>& e) {
                                              return e->GetName() == name;
                                            });

    if (remainingIt != entities.end()) {
      entityMap[name] = remainingIt->get();
    } else {
      entityMap.erase(name);
    }

    return true;
  }

  return false;
}

bool Engine::RemoveEntity(const std::string& name) {
  // If called from a background thread, defer removal to avoid deleting entities
  // while the render thread may be iterating a snapshot.
  if (!IsMainThread()) {
    std::lock_guard<std::mutex> lk(pendingEntityRemovalsMutex);
    pendingEntityRemovalNames.push_back(name);
    return true;
  }

  std::unique_lock<std::shared_mutex> lk(entitiesMutex);
  auto it = entityMap.find(name);
  if (it == entityMap.end())
    return false;
  Entity* entity = it->second;
  if (!entity)
    return false;

  // Find the entity in the vector
  auto vecIt = std::ranges::find_if(entities,
                                    [entity](const std::unique_ptr<Entity>& e) {
                                      return e.get() == entity;
                                    });
  if (vecIt == entities.end()) {
    entityMap.erase(name);
    return false;
  }

  entities.erase(vecIt);

  // Update the map: point to another entity with the same name if one exists
  auto remainingIt = std::ranges::find_if(entities,
                                          [&name](const std::unique_ptr<Entity>& e) {
                                            return e && e->GetName() == name;
                                          });
  if (remainingIt != entities.end()) {
    entityMap[name] = remainingIt->get();
  } else {
    entityMap.erase(name);
  }
  return true;
}

void Engine::SetActiveCamera(CameraComponent* cameraComponent) {
  activeCamera = cameraComponent;
  if (imguiSystem)
    imguiSystem->SetActiveCamera(cameraComponent);
}

const CameraComponent* Engine::GetActiveCamera() const {
  return activeCamera;
}

CameraComponent* Engine::GetActiveCamera() {
  return activeCamera;
}

const ResourceManager* Engine::GetResourceManager() const {
  return resourceManager.get();
}

const Platform* Engine::GetPlatform() const {
  return platform.get();
}

Renderer* Engine::GetRenderer() {
  return renderer.get();
}

ModelLoader* Engine::GetModelLoader() {
  return modelLoader.get();
}

const AudioSystem* Engine::GetAudioSystem() const {
  return audioSystem.get();
}

PhysicsSystem* Engine::GetPhysicsSystem() {
  return physicsSystem.get();
}

const ImGuiSystem* Engine::GetImGuiSystem() const {
  return imguiSystem.get();
}

ImGuiSystem* Engine::GetImGuiSystem() {
  return imguiSystem.get();
}

// Forward SetScenePicker to ImGuiSystem so the callback is rendered each frame.
// Override so that calling engine.SetScenePicker() is enough — no need to touch ImGuiSystem directly.
// (The base storage in engine.h is kept for GetScenePicker() / IsSceneLoaded() checks.)
void Engine::SetScenePicker(std::function<void()> callback) {
  scenePickerUI = callback;
  if (imguiSystem) {
    imguiSystem->SetScenePicker(callback);
  }
}

void Engine::SetSceneUI(std::function<void()> callback) {
  if (imguiSystem)
    imguiSystem->SetSceneUI(std::move(callback));
}

void Engine::SubscribeSceneFixedUpdate(std::function<void(float)> cb) {
  if (updatePublisher) updatePublisher->SubscribeSceneFixedUpdate(std::move(cb));
}
void Engine::SubscribeSceneUpdate(std::function<void(float)> cb) {
  if (updatePublisher) updatePublisher->SubscribeSceneUpdate(std::move(cb));
}
void Engine::SubscribeSceneLateUpdate(std::function<void(float)> cb) {
  if (updatePublisher) updatePublisher->SubscribeSceneLateUpdate(std::move(cb));
}

void Engine::handleMouseInput(float x, float y, uint32_t buttons) {
  // Check if ImGui wants to capture mouse input first
  bool imguiWantsMouse = imguiSystem && imguiSystem->WantCaptureMouse();

  // Suppress right-click while loading
  if (renderer&& renderer
  
  ->
  IsLoading()
  ) {
    buttons &= ~2u; // clear right button bit
  }

  if (!imguiWantsMouse) {
    // Handle mouse click for ball throwing (right mouse button)
    if (buttons & 2) {
      // Right mouse button (bit 1)
      if (!cameraControl.mouseRightPressed) {
        cameraControl.mouseRightPressed = true;
        // Throw a ball on mouse click
        ThrowBall(x, y);
      }
    } else {
      cameraControl.mouseRightPressed = false;
    }

    // Handle camera rotation when left mouse button is pressed
    if (buttons & 1) {
      // Left mouse button (bit 0)
      if (!cameraControl.mouseLeftPressed) {
        cameraControl.mouseLeftPressed = true;
        cameraControl.firstMouse = true;
      }

      if (cameraControl.firstMouse) {
        cameraControl.lastMouseX = x;
        cameraControl.lastMouseY = y;
        cameraControl.firstMouse = false;
      }

      float xOffset = x - cameraControl.lastMouseX;
      float yOffset = y - cameraControl.lastMouseY;
      cameraControl.lastMouseX = x;
      cameraControl.lastMouseY = y;

      xOffset *= cameraControl.mouseSensitivity;
      yOffset *= cameraControl.mouseSensitivity;

      // Mouse look: positive X moves view to the right; positive Y moves view up.
      // Platform mouse coordinates increase downward, so invert Y.
      cameraControl.yaw -= xOffset;
      cameraControl.pitch -= yOffset;

      // Constrain pitch to avoid gimbal lock
      if (cameraControl.pitch > 89.0f)
        cameraControl.pitch = 89.0f;
      if (cameraControl.pitch < -89.0f)
        cameraControl.pitch = -89.0f;
    } else {
      cameraControl.mouseLeftPressed = false;
    }
  }

  if (imguiSystem) {
    imguiSystem->HandleMouse(x, y, buttons);
  }

  // Always perform hover detection (even when ImGui is active)
  HandleMouseHover(x, y);
}
void Engine::handleKeyInput(uint32_t key, bool pressed) {
#if !defined(PLATFORM_ANDROID)
  switch (key) {
    case GLFW_KEY_W:
    case GLFW_KEY_UP:
      cameraControl.moveForward = pressed;
      break;
    case GLFW_KEY_S:
    case GLFW_KEY_DOWN:
      cameraControl.moveBackward = pressed;
      break;
    case GLFW_KEY_A:
    case GLFW_KEY_LEFT:
      cameraControl.moveLeft = pressed;
      break;
    case GLFW_KEY_D:
    case GLFW_KEY_RIGHT:
      cameraControl.moveRight = pressed;
      break;
    case GLFW_KEY_E:
    case GLFW_KEY_PAGE_UP:
      cameraControl.moveUp = pressed;
      break;
    case GLFW_KEY_Q:
    case GLFW_KEY_PAGE_DOWN:
      cameraControl.moveDown = pressed;
      break;
    default:
      break;
  }

  if (imguiSystem) {
    imguiSystem->HandleKeyboard(key, pressed);
  }
#else
  // Android uses different input handling via touch events
  (void) key;
  (void) pressed;
#endif
}

void Engine::Update(TimeDelta deltaTime) {
  // Drain entity removals queued from background threads.
  ProcessPendingEntityRemovals();

  // While a scene is being loaded on a background thread, only drive the
  // loading overlay — skip all game-logic updates.
  if (renderer && renderer->IsLoading()) {
    if (imguiSystem) imguiSystem->NewFrame();
    return;
  }

  // Create any balls requested this frame before the physics step fires.
  ProcessPendingBalls();

  // UpdatePublisher is the single source of truth for all per-frame logic:
  //   CoreFixedUpdate  → physicsSystem->Step()  (60 Hz)
  //   CoreUpdate       → audioSystem, entity updates
  //   CoreLateUpdate   → camera controls
  //   SceneFixedUpdate → buoyancy forces, scene-specific fixed logic
  //   SceneUpdate      → scene per-frame logic
  //   SceneLateUpdate  → scene post-update logic
  if (updatePublisher)
    updatePublisher->TickWithDt(deltaTime.count() * 0.001f);

  // ImGui new-frame must happen after all logic updates and before Render().
  if (imguiSystem) imguiSystem->NewFrame();
}

void Engine::Render() {
  // Ensure renderer is ready
  if (!renderer || !renderer->IsInitialized()) {
    return;
  }

  // Apply any entity removals requested by background threads before taking a snapshot.
  ProcessPendingEntityRemovals();

  // Snapshot entity pointers under a short shared lock, then release the lock
  // before rendering. This prevents starving the background loader/physics threads
  // that need the unique lock to create entities/components.
  std::vector<Entity *> snapshot; {
    std::shared_lock<std::shared_mutex> lk(entitiesMutex);
    snapshot.reserve(entities.size());
    for (auto& uptr : entities) {
      snapshot.push_back(uptr.get());
    }
  }

  // Render the scene (ImGui will be rendered within the render pass)
  renderer->Render(snapshot, activeCamera, imguiSystem.get());
}

std::chrono::milliseconds Engine::CalculateDeltaTimeMs() {
  // Get current time using a steady clock to avoid system time jumps
  uint64_t currentTime = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch())
    .count());

  // Initialize lastFrameTimeMs on first call
  if (lastFrameTimeMs == 0) {
    lastFrameTimeMs = currentTime;
    return std::chrono::milliseconds(16); // ~16ms as a sane initial guess
  }

  // Calculate delta time in milliseconds
  uint64_t delta = currentTime - lastFrameTimeMs;

  // Update last frame time
  lastFrameTimeMs = currentTime;

  return std::chrono::milliseconds(static_cast<long long>(delta));
}

void Engine::HandleResize(int width, int height) const {
  if (height <= 0 || width <= 0) {
    return;
  }
  // Update the active camera's aspect ratio
  if (activeCamera) {
    activeCamera->SetAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  }

  // Notify the renderer that the framebuffer has been resized
  if (renderer) {
    renderer->SetFramebufferResized();
  }

  // Notify ImGui system about the resize
  if (imguiSystem) {
    imguiSystem->HandleResize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
  }
}

void Engine::UpdateCameraControls(TimeDelta deltaTime) {
  if (!activeCamera)
    return;

  // Get a camera transform component
  auto* cameraTransform = activeCamera->GetOwner()->GetComponent<TransformComponent>();
  if (!cameraTransform)
    return;

  // Check if camera tracking is enabled
  if (imguiSystem&& imguiSystem
  
  ->
  IsCameraTrackingEnabled()
  ) {
    // Find the first active ball entity
    Entity* ballEntity = nullptr; {
      std::shared_lock<std::shared_mutex> lk(entitiesMutex);
      auto ballEntityIt = std::ranges::find_if(entities,
                                               [](auto const& entity) {
                                                 return entity && entity->IsActive() && (entity->GetName().find("Ball_") != std::string::npos);
                                               });
      ballEntity = (ballEntityIt != entities.end()) ? ballEntityIt->get() : nullptr;
    }

    if (ballEntity) {
      // Get ball's transform component
      auto* ballTransform = ballEntity->GetComponent<TransformComponent>();
      if (ballTransform) {
        glm::vec3 ballPosition = ballTransform->GetPosition();

        // Position camera at a fixed offset from the ball for good viewing
        glm::vec3 cameraOffset = glm::vec3(2.0f, 1.5f, 2.0f); // Behind and above the ball
        glm::vec3 cameraPosition = ballPosition + cameraOffset;

        // Update camera position and target
        cameraTransform->SetPosition(cameraPosition);
        activeCamera->SetTarget(ballPosition);

        return; // Skip manual controls when tracking
      }
    }
  }

  // Manual camera controls (only when tracking is disabled)
  // Calculate movement speed
  float velocity = cameraControl.cameraSpeed * deltaTime.count() * .001f;

  // Capture base orientation from GLTF camera once and then apply mouse deltas relative to it
  if (!cameraControl.baseOrientationCaptured) {
    // TransformComponent stores Euler in radians; convert to quaternion
    glm::vec3 baseEuler = cameraTransform->GetRotation();
    const glm::quat qx = glm::angleAxis(baseEuler.x, glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::quat qy = glm::angleAxis(baseEuler.y, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat qz = glm::angleAxis(baseEuler.z, glm::vec3(0.0f, 0.0f, 1.0f));
    // Match CameraComponent::UpdateViewMatrix composition (q = qz * qy * qx)
    cameraControl.baseOrientation = qz * qy * qx;
    cameraControl.baseOrientationCaptured = true;
  }

  // Build delta orientation from yaw/pitch mouse deltas (degrees -> radians)
  const float yawRad = glm::radians(cameraControl.yaw);
  const float pitchRad = glm::radians(cameraControl.pitch);
  const glm::quat qDeltaY = glm::angleAxis(yawRad, glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::quat qDeltaX = glm::angleAxis(pitchRad, glm::vec3(1.0f, 0.0f, 0.0f));
  // Apply yaw then pitch in the same convention as CameraComponent (ZYX overall), so delta = Ry * Rx
  glm::quat qDelta = qDeltaY * qDeltaX;
  glm::quat qFinal = cameraControl.baseOrientation * qDelta;

  // Derive camera basis directly from rotated axes to avoid ambiguity
  glm::vec3 right = glm::normalize(qFinal * glm::vec3(1.0f, 0.0f, 0.0f));
  glm::vec3 up = glm::normalize(qFinal * glm::vec3(0.0f, 1.0f, 0.0f));
  // Camera forward in world space.
  // Our view/projection conventions assume the camera looks down -Z in its local space.
  glm::vec3 front = glm::normalize(qFinal * glm::vec3(0.0f, 0.0f, -1.0f));

  // Get the current camera position
  glm::vec3 position = cameraTransform->GetPosition();

  // Apply movement based on input
  if (cameraControl.moveForward) {
    position += front * velocity;
  }
  if (cameraControl.moveBackward) {
    position -= front * velocity;
  }
  if (cameraControl.moveLeft) {
    position -= right * velocity;
  }
  if (cameraControl.moveRight) {
    position += right * velocity;
  }
  if (cameraControl.moveUp) {
    position += up * velocity;
  }
  if (cameraControl.moveDown) {
    position -= up * velocity;
  }

  // Update camera position
  cameraTransform->SetPosition(position);
  // Apply rotation to the camera transform based on GLTF base orientation plus mouse deltas
  // TransformComponent expects radians Euler (ZYX order in our CameraComponent).
  cameraTransform->SetRotation(glm::eulerAngles(qFinal));

  // Update camera target based on a direction
  glm::vec3 target = position + front;
  activeCamera->SetTarget(target);

  // Ensure the camera view matrix reflects the new transform immediately this frame
  activeCamera->ForceViewMatrixUpdate();
}

void Engine::GenerateBallMaterial() {
  // Generate 8 random material properties for PBR
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Generate bright, vibrant albedo colors for better visibility
  std::uniform_real_distribution<float> brightDis(0.6f, 1.0f); // Ensure bright colors
  ballMaterial.albedo = glm::vec3(brightDis(gen), brightDis(gen), brightDis(gen));

  // Random metallic value (0.0 to 1.0)
  ballMaterial.metallic = dis(gen);

  // Random roughness value (0.0 to 1.0)
  ballMaterial.roughness = dis(gen);

  // Random ambient occlusion (typically 0.8 to 1.0 for good lighting)
  ballMaterial.ao = 0.8f + dis(gen) * 0.2f;

  // Random emissive color (usually subtle)
  ballMaterial.emissive = glm::vec3(dis(gen) * 0.3f, dis(gen) * 0.3f, dis(gen) * 0.3f);

  // Decent bounciness (0.6 to 0.9) so bounces are clearly visible
  ballMaterial.bounciness = 0.6f + dis(gen) * 0.3f;
}

void Engine::InitializePhysicsScaling() {
  // Based on issue analysis: balls reaching 120+ m/s and extreme positions like (-244, -360, -244)
  // The previous 200.0f force scale was causing supersonic speeds and balls flying out of scene
  // Need much more conservative scaling for realistic visual gameplay

  // Use smaller game unit scale for more controlled physics
  physicsScaling.gameUnitsToMeters = 0.1f; // 1 game unit = 0.1 meter (10cm) - smaller scale

  // Much reduced force scaling to prevent extreme speeds
  // With base forces 0.01f-0.05f, this gives final forces of 0.001f-0.005f
  physicsScaling.forceScale = 1.0f; // Minimal force scaling for realistic movement
  physicsScaling.physicsTimeScale = 1.0f; // Keep time scale normal
  physicsScaling.gravityScale = 1.0f; // Keep gravity proportional to scale

  // Apply scaled gravity to physics system
  glm::vec3 realWorldGravity(0.0f, -9.81f, 0.0f);
  glm::vec3 scaledGravity = ScaleGravityForPhysics(realWorldGravity);
  physicsSystem->SetGravity(scaledGravity);
}

float Engine::ScaleForceForPhysics(float gameForce) const {
  // Scale force based on the relationship between game units and real world
  // and the force scaling factor to make physics feel right
  return gameForce * physicsScaling.forceScale * physicsScaling.gameUnitsToMeters;
}

glm::vec3 Engine::ScaleGravityForPhysics(const glm::vec3& realWorldGravity) const {
  // Scale gravity based on game unit scale and gravity scaling factor
  // If 1 game unit = 1 meter, then gravity should remain -9.81
  // If 1 game unit = 0.1 meter, then gravity should be -0.981
  return realWorldGravity * physicsScaling.gravityScale * physicsScaling.gameUnitsToMeters;
}

float Engine::ScaleTimeForPhysics(float deltaTime) const {
  // Scale time for physics simulation if needed
  // This can be used to slow down or speed up physics relative to rendering
  return deltaTime * physicsScaling.physicsTimeScale;
}

void Engine::ThrowBall(float mouseX, float mouseY) {
  if (!activeCamera || !physicsSystem) {
    return;
  }

  // Get window dimensions
  int windowWidth, windowHeight;
  platform->GetWindowSize(&windowWidth, &windowHeight);

  // Convert mouse coordinates to normalized device coordinates (-1 to 1)
  float ndcX = (2.0f * mouseX) / static_cast<float>(windowWidth) - 1.0f;
  float ndcY = 1.0f - (2.0f * mouseY) / static_cast<float>(windowHeight);

  // Get camera matrices
  glm::mat4 viewMatrix = activeCamera->GetViewMatrix();
  glm::mat4 projMatrix = activeCamera->GetProjectionMatrix();

  // Calculate inverse matrices
  glm::mat4 invView = glm::inverse(viewMatrix);
  glm::mat4 invProj = glm::inverse(projMatrix);

  // Convert NDC to world space for direction
  glm::vec4 rayClip = glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
  glm::vec4 rayEye = invProj * rayClip;
  rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
  glm::vec4 rayWorld = invView * rayEye;

  // Calculate screen center in world coordinates
  // Screen center is at NDC (0, 0) which corresponds to the center of the view
  glm::vec4 screenCenterClip = glm::vec4(0.0f, 0.0f, -1.0f, 1.0f);
  glm::vec4 screenCenterEye = invProj * screenCenterClip;
  screenCenterEye = glm::vec4(screenCenterEye.x, screenCenterEye.y, -1.0f, 0.0f);
  glm::vec4 screenCenterWorld = invView * screenCenterEye;
  glm::vec3 screenCenterDirection = glm::normalize(glm::vec3(screenCenterWorld));

  // Calculate world position for screen center at a reasonable distance from camera
  glm::vec3 cameraPosition = activeCamera->GetPosition();
  glm::vec3 screenCenterWorldPos = cameraPosition + screenCenterDirection * 2.0f; // 2 units in front of camera

  // Calculate throw direction from screen center toward mouse position
  glm::vec3 throwDirection = glm::normalize(glm::vec3(rayWorld));

  // Add upward component for realistic arc trajectory
  throwDirection.y += 0.3f; // Add upward bias for throwing arc
  throwDirection = glm::normalize(throwDirection); // Re-normalize after modification

  // Generate ball properties now
  static int ballCounter = 0;
  std::string ballName = "Ball_" + std::to_string(ballCounter++);

  std::random_device rd;
  std::mt19937 gen(rd());

  // Launch balls from screen center toward mouse cursor
  glm::vec3 spawnPosition = screenCenterWorldPos;

  // Add small random variation to avoid identical paths
  std::uniform_real_distribution<float> posDis(-0.1f, 0.1f);
  spawnPosition.x += posDis(gen);
  spawnPosition.y += posDis(gen);
  spawnPosition.z += posDis(gen);

  std::uniform_real_distribution<float> spinDis(-10.0f, 10.0f);
  std::uniform_real_distribution<float> forceDis(15.0f, 35.0f); // Stronger force range for proper throwing feel

  // Store ball creation data for processing outside rendering loop
  PendingBall pendingBall;
  pendingBall.spawnPosition = spawnPosition;
  pendingBall.throwDirection = throwDirection; // This is now the corrected direction toward geometry
  pendingBall.throwForce = ScaleForceForPhysics(forceDis(gen)); // Apply physics scaling to force
  pendingBall.randomSpin = glm::vec3(spinDis(gen), spinDis(gen), spinDis(gen));
  pendingBall.ballName = ballName;

  pendingBalls.push_back(pendingBall);
}

void Engine::ProcessPendingBalls() {
  // Process all pending balls
  for (const auto& pendingBall : pendingBalls) {
    // Create ball entity
    Entity* ballEntity = CreateEntity(pendingBall.ballName);
    if (!ballEntity) {
      std::cerr << "Failed to create ball entity: " << pendingBall.ballName << std::endl;
      continue;
    }

    // Add transform component
    auto* transform = ballEntity->AddComponent<TransformComponent>();
    if (!transform) {
      std::cerr << "Failed to add TransformComponent to ball: " << pendingBall.ballName << std::endl;
      continue;
    }
    transform->SetPosition(pendingBall.spawnPosition);
    transform->SetScale(glm::vec3(1.0f)); // Tennis ball size scale

    // Add mesh component with sphere geometry
    auto* mesh = ballEntity->AddComponent<MeshComponent>();
    if (!mesh) {
      std::cerr << "Failed to add MeshComponent to ball: " << pendingBall.ballName << std::endl;
      continue;
    }
    // Create tennis ball-sized, bright red sphere
    glm::vec3 brightRed(1.0f, 0.0f, 0.0f);
    mesh->CreateSphere(0.0335f, brightRed, 32); // Tennis ball radius, bright color, high detail
    mesh->SetTexturePath(renderer->SHARED_BRIGHT_RED_ID); // Use bright red texture for visibility

    // Verify mesh geometry was created
    const auto& vertices = mesh->GetVertices();
    const auto& indices = mesh->GetIndices();
    if (vertices.empty() || indices.empty()) {
      std::cerr << "ERROR: CreateSphere failed to generate geometry!" << std::endl;
      continue;
    }

    // Pre-allocate Vulkan resources for this entity (now outside rendering loop)
    if (!renderer->preAllocateEntityResources(ballEntity)) {
      std::cerr << "Failed to pre-allocate resources for ball: " << pendingBall.ballName << std::endl;
      continue;
    }

    // Create rigid body with sphere collision shape
    RigidBody* rigidBody = physicsSystem->CreateRigidBody(ballEntity, CollisionShape::Sphere, 1.0f);
    if (rigidBody) {
      // Set bounciness from material
      rigidBody->SetRestitution(ballMaterial.bounciness);

      // Request an acceleration structure build so the new ball is included in Ray Query mode.
      // We do this after creating the rigid body and initializing the entity.
      renderer->RequestAccelerationStructureBuild("Ball spawned");

      // Apply throw force and spin
      glm::vec3 throwImpulse = pendingBall.throwDirection * pendingBall.throwForce;
      rigidBody->ApplyImpulse(throwImpulse, glm::vec3(0.0f));
      rigidBody->SetAngularVelocity(pendingBall.randomSpin);
    }
  }

  // Clear processed balls
  pendingBalls.clear();
}

void Engine::HandleMouseHover(float mouseX, float mouseY) {
  // Update current mouse position for any systems that might need it
  currentMouseX = mouseX;
  currentMouseY = mouseY;
}

#if defined(PLATFORM_ANDROID)
// Android-specific implementation
bool Engine::InitializeAndroid(android_app* app, const std::string& appName, bool enableValidationLayers) {
  // Create platform
  platform = CreatePlatform(app);
  if (!platform->Initialize(appName, 0, 0)) {
    return false;
  }

  // Set resize callback
  platform->SetResizeCallback([this](int width, int height) {
    HandleResize(width, height);
  });

  // Set mouse callback
  platform->SetMouseCallback([this](float x, float y, uint32_t buttons) {
    // Check if ImGui wants to capture mouse input first
    bool imguiWantsMouse = imguiSystem && imguiSystem->WantCaptureMouse();

    if (!imguiWantsMouse) {
      // Handle mouse click for ball throwing (right mouse button)
      if (buttons & 2) {
        // Right mouse button (bit 1)
        if (!cameraControl.mouseRightPressed) {
          cameraControl.mouseRightPressed = true;
          // Throw a ball on mouse click
          ThrowBall(x, y);
        }
      } else {
        cameraControl.mouseRightPressed = false;
      }
    }

    if (imguiSystem) {
      imguiSystem->HandleMouse(x, y, buttons);
    }
  });

  // Set keyboard callback
  platform->SetKeyboardCallback([this](uint32_t key, bool pressed) {
    if (imguiSystem) {
      imguiSystem->HandleKeyboard(key, pressed);
    }
  });

  // Set char callback
  platform->SetCharCallback([this](uint32_t c) {
    if (imguiSystem) {
      imguiSystem->HandleChar(c);
    }
  });

  // Create renderer
  renderer = std::make_unique<Renderer>(platform.get());
  if (!renderer->Initialize(appName, enableValidationLayers)) {
    return false;
  }

  // Get window dimensions from platform for ImGui initialization
  int width, height;
  platform->GetWindowSize(&width, &height);

  try {
    // Model loader via constructor; also wire into renderer
    modelLoader = std::make_unique<ModelLoader>(renderer.get());
    renderer->SetModelLoader(modelLoader.get());

    // Audio system via constructor
    audioSystem = std::make_unique<AudioSystem>(this, renderer.get());

    physicsSystem = std::make_unique<PhysicsSystem>();

    // ImGui via constructor, then connect audio system
    imguiSystem = std::make_unique<ImGuiSystem>(renderer.get(), width, height);
    imguiSystem->SetAudioSystem(audioSystem.get());
  } catch (const std::exception& e) {
    std::cerr << "Subsystem initialization failed: " << e.what() << std::endl;
    return false;
  }

  // Generate ball material properties once at load time
  GenerateBallMaterial();

  // Initialize physics scaling system
  InitializePhysicsScaling();

  initialized = true;
  return true;
}

void Engine::RunAndroid() {
  if (!initialized) {
    throw std::runtime_error("Engine not initialized");
  }

  running = true;

  // Main loop is handled by the platform
  // We just need to update and render when the platform is ready

  // Calculate delta time
  deltaTimeMs = CalculateDeltaTimeMs();

  // Update
  Update(deltaTimeMs);

  // Render
  Render();
}
#endif