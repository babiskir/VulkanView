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
#include "renderer.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <ranges>
#include <set>
#include <thread>
#include <type_traits>

// Dispatch loader storage is defined in vulkan_dispatch.cpp

#include <vulkan/vk_platform.h>
#include <vulkan/vulkan.h>          // For PFN_vkGetInstanceProcAddr and C types
#include <vulkan/vulkan_raii.hpp>

// Debug callback for vk::raii - uses raw Vulkan C types for cross-platform compatibility
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallbackVkRaii(
  VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  [[maybe_unused]] void* pUserData) {
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    // Print a message to the console
    std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
  } else {
    // Print a message to the console
    std::cout << "Validation layer: " << pCallbackData->pMessage << std::endl;
  }

  return VK_FALSE;
}

// Vulkan-Hpp style callback signature for newer headers expecting vk:: types
static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallbackVkHpp(
  vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  [[maybe_unused]] vk::DebugUtilsMessageTypeFlagsEXT messageType,
  const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
  [[maybe_unused]] void* pUserData) {
  if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
  } else {
    std::cout << "Validation layer: " << pCallbackData->pMessage << std::endl;
  }
  return vk::False;
}

// Watchdog thread function - monitors frame updates and aborts if application hangs
static void WatchdogThreadFunc(std::atomic<std::chrono::steady_clock::time_point>* lastFrameTime,
                               std::atomic<bool>* running,
                               std::atomic<bool>* suppressed,
                               std::atomic<const char *>* progressLabel,
                               std::atomic<uint32_t>* progressIndex) {
  while (running->load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::seconds(5));

    if (!running->load(std::memory_order_relaxed)) {
      break; // Shutdown requested
    }

    // Check if frame timestamp was updated recently.
    // Some operations (e.g., BLAS/TLAS builds in Debug on large scenes) can legitimately take
    // much longer than 5 or 10 seconds. When suppressed, allow a longer grace period.
    auto now = std::chrono::steady_clock::now();
    auto lastUpdate = lastFrameTime->load(std::memory_order_relaxed);
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate).count();
    const int64_t allowedSeconds = (suppressed && suppressed->load(std::memory_order_relaxed)) ? 60 : 10;

    if (elapsed >= allowedSeconds) {
      // APPLICATION HAS HUNG - no frame updates for 10+ seconds
      const char* label = nullptr;
      if (progressLabel) {
        label = progressLabel->load(std::memory_order_relaxed);
      }
      uint32_t idx = 0;
      if (progressIndex) {
        idx = progressIndex->load(std::memory_order_relaxed);
      }

      std::cerr << "\n\n";
      std::cerr << "========================================\n";
      std::cerr << "WATCHDOG: APPLICATION HAS HUNG!\n";
      std::cerr << "========================================\n";
      std::cerr << "Last frame update was " << elapsed << " seconds ago.\n";
      if (label && label[0] != '\0') {
        std::cerr << "Last progress marker: " << label << "\n";
      }
      if (progressIndex) {
        std::cerr << "Progress index: " << idx << "\n";
      }
      std::cerr << "The render loop is not progressing.\n";
      std::cerr << "Aborting to generate stack trace...\n";
      std::cerr << "========================================\n\n";
      std::abort(); // Force crash with stack trace
    }
  }

  std::cout << "[Watchdog] Stopped\n";
}

// Renderer core implementation for the "Rendering Pipeline" chapter of the tutorial.
Renderer::Renderer(Platform* platform) : platform(platform) {
  // Initialize deviceExtensions with required extensions only
  // Optional extensions will be added later after checking device support
  deviceExtensions = requiredDeviceExtensions;
}

// Destructor
Renderer::~Renderer() {
  Cleanup();
}

// Initialize the renderer
bool Renderer::Initialize(const std::string& appName, bool enableValidationLayers) {
  // Initialize the Vulkan-Hpp default dispatcher using the global symbol directly.
  // This avoids differences across Vulkan-Hpp versions for DynamicLoader placement.
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
  // Create a Vulkan instance
  if (!createInstance(appName, enableValidationLayers)) {
    std::cerr << "Failed to create Vulkan instance" << std::endl;
    return false;
  }

  // Setup debug messenger (uses actual runtime state, not the requested flag)
  if (!setupDebugMessenger(validationLayersEnabled)) {
    std::cerr << "Failed to setup debug messenger" << std::endl;
    return false;
  }

  // Create surface
  if (!createSurface()) {
    std::cerr << "Failed to create surface" << std::endl;
    return false;
  }

  // Pick the physical device
  if (!pickPhysicalDevice()) {
    std::cerr << "Failed to pick physical device" << std::endl;
    return false;
  }

  // Create logical device
  if (!createLogicalDevice(validationLayersEnabled)) {
    std::cerr << "Failed to create logical device" << std::endl;
    return false;
  }

  // Initialize VMA allocator
  {
    VmaVulkanFunctions vmaVulkanFuncs{};
    vmaVulkanFuncs.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vmaVulkanFuncs.vkGetDeviceProcAddr   = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    allocatorInfo.physicalDevice   = *physicalDevice;
    allocatorInfo.device           = *device;
    allocatorInfo.instance         = *instance;
    allocatorInfo.pVulkanFunctions = &vmaVulkanFuncs;
    allocatorInfo.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&allocatorInfo, &vmaAllocator) != VK_SUCCESS) {
      std::cerr << "Failed to create VMA allocator" << std::endl;
      return false;
    }
    std::cout << "VMA allocator created." << std::endl;
  }

  // Create swap chain
  if (!createSwapChain()) {
    std::cerr << "Failed to create swap chain" << std::endl;
    return false;
  }

  // Create image views
  if (!createImageViews()) {
    std::cerr << "Failed to create image views" << std::endl;
    return false;
  }

  // Setup dynamic rendering
  if (!setupDynamicRendering()) {
    std::cerr << "Failed to setup dynamic rendering" << std::endl;
    return false;
  }

  // Create the descriptor set layout
  if (!createDescriptorSetLayout()) {
    std::cerr << "Failed to create descriptor set layout" << std::endl;
    return false;
  }

  // Create the graphics pipeline
  if (!createGraphicsPipeline()) {
    std::cerr << "Failed to create graphics pipeline" << std::endl;
    return false;
  }

  // Create PBR pipeline
  if (!createPBRPipeline()) {
    std::cerr << "Failed to create PBR pipeline" << std::endl;
    return false;
  }

  // Create the lighting pipeline
  if (!createLightingPipeline()) {
    std::cerr << "Failed to create lighting pipeline" << std::endl;
    return false;
  }

  // Create composite pipeline (fullscreen pass for off-screen → swapchain)
  if (!createCompositePipeline()) {
    std::cerr << "Failed to create composite pipeline" << std::endl;
    return false;
  }

  // Create compute pipeline
  if (!createComputePipeline()) {
    std::cerr << "Failed to create compute pipeline" << std::endl;
    return false;
  }

  // Ensure light storage buffers exist before creating Forward+ resources
  // so that compute descriptor binding 0 (lights SSBO) can be populated safely.
  if (!createOrResizeLightStorageBuffers(1)) {
    std::cerr << "Failed to create initial light storage buffers" << std::endl;
    return false;
  }

  // Create Forward+ compute and depth pre-pass pipelines/resources
  if (useForwardPlus) {
    if (!createForwardPlusPipelinesAndResources()) {
      std::cerr << "Failed to create Forward+ resources" << std::endl;
      return false;
    }
  }

  // Create ray query descriptor set layout and pipeline (but not resources yet - need descriptor pool first)
  if (!createRayQueryDescriptorSetLayout()) {
    std::cerr << "Failed to create ray query descriptor set layout" << std::endl;
    return false;
  }
  if (!createRayQueryPipeline()) {
    std::cerr << "Failed to create ray query pipeline" << std::endl;
    return false;
  }

  // Create the command pool
  if (!createCommandPool()) {
    std::cerr << "Failed to create command pool" << std::endl;
    return false;
  }

  // Create depth resources
  if (!createDepthResources()) {
    std::cerr << "Failed to create depth resources" << std::endl;
    return false;
  }

  if (useForwardPlus) {
    if (!createDepthPrepassPipeline()) {
      std::cerr << "Failed to create depth prepass pipeline" << std::endl;
      return false;
    }
  }

  // Create CSM depth pipeline (creates csmDepthDescriptorSetLayout used by createCSMResources)
  if (!createCSMDepthPipeline()) {
    std::cerr << "Warning: Failed to create CSM depth pipeline — shadows disabled" << std::endl;
    // Non-fatal: continue without CSM
  }

  // Create the descriptor pool
  if (!createDescriptorPool()) {
    std::cerr << "Failed to create descriptor pool" << std::endl;
    return false;
  }

  // Create ray query resources AFTER descriptor pool (needs pool for descriptor set allocation)
  if (!createRayQueryResources()) {
    std::cerr << "Failed to create ray query resources" << std::endl;
    return false;
  }

  // Note: Acceleration structure build is requested by scene_loading.cpp after entities load
  // No need to request it here during init

  // Light storage buffers were already created earlier to satisfy Forward+ binding requirements

  if (!createOpaqueSceneColorResources()) {
    std::cerr << "Failed to create opaque scene color resources" << std::endl;
    return false;
  }

  createTransparentDescriptorSets();

  // Create default texture resources
  if (!createDefaultTextureResources()) {
    std::cerr << "Failed to create default texture resources" << std::endl;
    return false;
  }

  // Create fallback transparent descriptor sets (must occur after default textures exist)
  createTransparentFallbackDescriptorSets();

  // Create shared default PBR textures (to avoid creating hundreds of identical textures)
  if (!createSharedDefaultPBRTextures()) {
    std::cerr << "Failed to create shared default PBR textures" << std::endl;
    return false;
  }

  // Create CSM resources (shadow map images, samplers, UBO buffers)
  // Requires csmDepthDescriptorSetLayout created by createCSMDepthPipeline()
  if (*csmDepthDescriptorSetLayout) {
    if (!createCSMResources()) {
      std::cerr << "Warning: Failed to create CSM resources — shadows disabled" << std::endl;
    }
  }

  // Create sky pipeline
  if (!createSkyPipeline()) {
    std::cerr << "Warning: Failed to create sky pipeline — sky disabled" << std::endl;
  }

  // Create water pipeline (always — WaterSystem may be activated at runtime)
  if (!createWaterPipeline()) {
    std::cerr << "Warning: Failed to create water pipeline\n";
  }

  // Create bloom pipelines + resources (dual-Kawase HDR bloom)
  if (!createBloomPipelines()) {
    std::cerr << "Warning: Failed to create bloom pipelines — bloom disabled\n";
    enableBloom = false;
  } else if (!createBloomResources()) {
    std::cerr << "Warning: Failed to create bloom resources — bloom disabled\n";
    enableBloom = false;
  }

  // Create auto-exposure pipelines + resources (histogram-based, [Ch18])
  if (!createExposurePipelines()) {
    std::cerr << "Warning: Failed to create exposure pipelines — auto-exposure disabled\n";
    enableAutoExposure = false;
  } else if (!createExposureResources()) {
    std::cerr << "Warning: Failed to create exposure resources — auto-exposure disabled\n";
    enableAutoExposure = false;
  }

  // Load HDR environment map if present (equirectangular, used for sky + water reflections)
  {
    const char* envPath = "Assets/hdri/farm_field_puresky_4k.exr";
    if (std::filesystem::exists(envPath)) {
      if (!LoadEnvTexture(envPath))
        std::cerr << "Warning: Failed to load env texture from " << envPath << "\n";
      else
        useEnvMapForSky = true;  // default to HDR map when one is available
    } else {
      std::cout << "[EnvTex] No env texture found at " << envPath << " — using procedural sky.\n";
    }
  }

  // Create command buffers
  if (!createCommandBuffers()) {
    std::cerr << "Failed to create command buffers" << std::endl;
    return false;
  }

  // Create sync objects
  if (!createSyncObjects()) {
    std::cerr << "Failed to create sync objects" << std::endl;
    return false;
  }

  // Initialize background thread pool for async tasks (textures, etc.) AFTER all Vulkan resources are ready
  try {
    // Size the thread pool based on hardware concurrency, clamped to a sensible range
    unsigned int hw = std::max(2u, std::min(8u, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4u));
    threadPool = std::make_unique<ThreadPool>(hw);
  } catch (const std::exception& e) {
    std::cerr << "Failed to create thread pool: " << e.what() << std::endl;
    return false;
  }

  // Start background uploads worker now that queues/semaphores exist
  StartUploadsWorker();

  // Start watchdog thread to detect application hangs
  lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
  watchdogRunning.store(true, std::memory_order_relaxed);
  watchdogThread = std::thread(WatchdogThreadFunc, &lastFrameUpdateTime, &watchdogRunning, &watchdogSuppressed, &watchdogProgressLabel, &watchdogProgressIndex);

  std::cout << "[Watchdog] Started - will abort if no frame updates for 10+ seconds\n";

  initialized = true;
  return true;
}

void Renderer::ensureThreadLocalVulkanInit() const {
  // Initialize Vulkan-Hpp dispatcher per-thread; required for multi-threaded RAII usage
  static thread_local bool s_tlsInitialized = false;
  if (s_tlsInitialized)
    return;
  try {
    // Initialize the dispatcher for this thread using the global symbol.
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    if (*instance) {
      VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
    }
    if (*device) {
      VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
    }
    s_tlsInitialized = true;
  } catch (...) {
    // best-effort
  }
}

// Clean up renderer resources
void Renderer::Cleanup() {
  // Stop watchdog thread first to prevent false hang detection during shutdown
  if (watchdogRunning.load(std::memory_order_relaxed)) {
    watchdogRunning.store(false, std::memory_order_relaxed);
    if (watchdogThread.joinable()) {
      watchdogThread.join();
    }
  }

  // Ensure background workers are stopped before tearing down Vulkan resources
  StopUploadsWorker();

  // Disallow any further descriptor writes during shutdown.
  // This prevents late updates/frees racing against pool destruction.
  descriptorSetsValid.store(false, std::memory_order_relaxed); {
    std::lock_guard<std::mutex> lk(pendingDescMutex);
    pendingDescOps.clear();
    descriptorRefreshPending.store(false, std::memory_order_relaxed);
  } {
    std::unique_lock<std::shared_mutex> lock(threadPoolMutex);
    if (threadPool) {
      threadPool.reset();
    }
  }

  if (!initialized) {
    return;
  }

  std::cout << "Starting renderer cleanup..." << std::endl;

  // Wait for the device to be idle before cleaning up
  try {
    WaitIdle();
  } catch (...) {
  }

  // 1) Clean up any swapchain-scoped resources first
  cleanupSwapChain();

  // 2) Clear per-entity resources (descriptor sets and buffers) while descriptor pools still exist
  for (auto& kv : entityResources) {
    auto& resources = kv.second;
    resources.basicDescriptorSets.clear();
    resources.pbrDescriptorSets.clear();
    resources.uniformBuffers.clear();
    for (auto& ua : resources.uniformBufferAllocations) {
      if (ua) { vmaFreeMemory(vmaAllocator, ua); ua = VK_NULL_HANDLE; }
    }
    resources.uniformBufferAllocations.clear();
    resources.uniformBuffersMapped.clear();
    resources.instanceBuffer = nullptr;
    if (resources.instanceBufferAllocation) { vmaFreeMemory(vmaAllocator, resources.instanceBufferAllocation); resources.instanceBufferAllocation = VK_NULL_HANDLE; }
    resources.instanceBufferMapped = nullptr;
  }
  entityResources.clear();

  // 3) Clear any global descriptor sets that are allocated from pools to avoid dangling refs
  transparentDescriptorSets.clear();
  transparentFallbackDescriptorSets.clear();
  compositeDescriptorSets.clear();
  computeDescriptorSets.clear();
  rqCompositeDescriptorSets.clear();

  // 3.5) Clear ray query descriptor sets BEFORE destroying descriptor pool
  // Without this, rayQueryDescriptorSets' RAII destructor tries to free them after
  // the pool is destroyed, causing "Invalid VkDescriptorPool Object" validation errors
  rayQueryDescriptorSets.clear();

  // Ray Query composite sampler/sets are allocated from the shared descriptor pool.
  // Ensure they are released before destroying the pool.
  rqCompositeSampler = nullptr;

  // 4) Destroy/Reset pipelines and pipeline layouts (graphics/compute/forward+)
  graphicsPipeline = nullptr;
  pbrGraphicsPipeline = nullptr;
  pbrBlendGraphicsPipeline = nullptr;
  pbrPremulBlendGraphicsPipeline = nullptr;
  pbrPrepassGraphicsPipeline = nullptr;
  pbrReflectionGraphicsPipeline = nullptr;
  glassGraphicsPipeline = nullptr;
  lightingPipeline = nullptr;
  compositePipeline = nullptr;
  forwardPlusPipeline = nullptr;
  depthPrepassPipeline = nullptr;
  computePipeline = nullptr;

  // 4.1) Sky pipeline + env texture descriptor set
  skyDescriptorSet       = nullptr;
  skyDescriptorPool      = nullptr;
  skyDescriptorSetLayout = nullptr;
  skyPipeline            = nullptr;
  skyPipelineLayout      = nullptr;

  // 4.1.1) Environment HDR texture
  envImageView  = nullptr;
  envSampler    = nullptr;
  envImage      = nullptr;
  envMemory     = nullptr;
  hasEnvTexture = false;

  // 4.2) Water pipeline
  waterPipeline = nullptr;
  waterPipelineLayout = nullptr;

  // 4.2.5) CSM depth pipeline
  csmDepthPipeline = nullptr;
  csmDepthPipelineLayout = nullptr;

  pipelineLayout = nullptr;
  pbrPipelineLayout = nullptr;
  lightingPipelineLayout = nullptr;
  compositePipelineLayout = nullptr;
  pbrTransparentPipelineLayout = nullptr;
  forwardPlusPipelineLayout = nullptr;
  computePipelineLayout = nullptr;

  // 4.3) Ray query pipelines and layouts
  rayQueryPipeline = nullptr;
  rayQueryPipelineLayout = nullptr;

  // 4.5) Forward+ per-frame resources (including descriptor sets) must be released
  // BEFORE destroying descriptor pools to avoid vkFreeDescriptorSets with invalid pool
  for (auto& fp : forwardPlusPerFrame) {
    fp.tileHeaders = nullptr;
    if (fp.tileHeadersAlloc) { vmaFreeMemory(vmaAllocator, fp.tileHeadersAlloc); fp.tileHeadersAlloc = VK_NULL_HANDLE; }
    fp.tileLightIndices = nullptr;
    if (fp.tileLightIndicesAlloc) { vmaFreeMemory(vmaAllocator, fp.tileLightIndicesAlloc); fp.tileLightIndicesAlloc = VK_NULL_HANDLE; }
    fp.params = nullptr;
    if (fp.paramsAlloc) { vmaFreeMemory(vmaAllocator, fp.paramsAlloc); fp.paramsAlloc = VK_NULL_HANDLE; }
    fp.paramsMapped = nullptr;
    fp.debugOut = nullptr;
    if (fp.debugOutAlloc) { vmaFreeMemory(vmaAllocator, fp.debugOutAlloc); fp.debugOutAlloc = VK_NULL_HANDLE; }
    fp.probeOffscreen = nullptr;
    if (fp.probeOffscreenAlloc) { vmaFreeMemory(vmaAllocator, fp.probeOffscreenAlloc); fp.probeOffscreenAlloc = VK_NULL_HANDLE; }
    fp.probeSwapchain = nullptr;
    if (fp.probeSwapchainAlloc) { vmaFreeMemory(vmaAllocator, fp.probeSwapchainAlloc); fp.probeSwapchainAlloc = VK_NULL_HANDLE; }
    fp.computeSet = nullptr; // descriptor set allocated from compute/graphics pools
  }
  forwardPlusPerFrame.clear();

  // 5) Destroy descriptor set layouts and pools (compute + graphics)
  descriptorSetLayout = nullptr;
  pbrDescriptorSetLayout = nullptr;
  transparentDescriptorSetLayout = nullptr;
  compositeDescriptorSetLayout = nullptr;
  forwardPlusDescriptorSetLayout = nullptr;
  computeDescriptorSetLayout = nullptr;
  rayQueryDescriptorSetLayout = nullptr;
  csmDepthDescriptorSetLayout = nullptr;
  waterDescriptorSetLayout = nullptr;

  // CSM per-frame resources (descriptor sets allocated from csmFrames' own pools)
  csmFrames.clear();

  // Pools last, after sets are cleared
  computeDescriptorPool = nullptr;
  descriptorPool = nullptr;

  // 6) Clear textures and aliases, including default resources
  {
    std::unique_lock<std::shared_mutex> lk(textureResourcesMutex);
    // Free VMA allocations before RAII image/view handles are destroyed
    for (auto& kv : textureResources) {
      auto& tr = kv.second;
      if (tr.textureImageAllocation) { vmaFreeMemory(vmaAllocator, tr.textureImageAllocation); tr.textureImageAllocation = VK_NULL_HANDLE; }
    }
    textureResources.clear();
    textureAliases.clear();
  }
  // Reset default texture resources
  defaultTextureResources.textureSampler = nullptr;
  defaultTextureResources.textureImageView = nullptr;
  defaultTextureResources.textureImage = nullptr;
  if (defaultTextureResources.textureImageAllocation) { vmaFreeMemory(vmaAllocator, defaultTextureResources.textureImageAllocation); defaultTextureResources.textureImageAllocation = VK_NULL_HANDLE; }

  // 7) Opaque scene color and related descriptors
  opaqueSceneColorSampler = nullptr;
  opaqueSceneColorImages.clear();
  for (auto& a : opaqueSceneColorImageAllocations) { if (a) { vmaFreeMemory(vmaAllocator, a); a = VK_NULL_HANDLE; } }
  opaqueSceneColorImageAllocations.clear();
  opaqueSceneColorImageViews.clear();
  opaqueSceneColorImageLayouts.clear();

  // 7.5) Ray query output image and acceleration structures
  rayQueryOutputImageView = nullptr;
  rayQueryOutputImage = nullptr;
  if (rayQueryOutputImageAllocation) { vmaFreeMemory(vmaAllocator, rayQueryOutputImageAllocation); rayQueryOutputImageAllocation = VK_NULL_HANDLE; }

  // Free ray query uniform buffer allocations
  for (auto& a : rayQueryUniformAllocations) { if (a) { vmaFreeMemory(vmaAllocator, a); a = VK_NULL_HANDLE; } }
  rayQueryUniformAllocations.clear();
  rayQueryUniformBuffers.clear();
  rayQueryUniformBuffersMapped.clear();

  // Free TLAS persistent buffers
  tlasInstancesBuffer = nullptr;
  if (tlasInstancesAllocation) { vmaFreeMemory(vmaAllocator, tlasInstancesAllocation); tlasInstancesAllocation = VK_NULL_HANDLE; }
  tlasUpdateScratchBuffer = nullptr;
  if (tlasUpdateScratchAllocation) { vmaFreeMemory(vmaAllocator, tlasUpdateScratchAllocation); tlasUpdateScratchAllocation = VK_NULL_HANDLE; }

  // Clear acceleration structures (BLAS and TLAS buffers)
  // (VmaAllocation in AccelerationStructure.allocation freed via BLAS/TLAS loop below)
  for (auto& blas : blasStructures) {
    blas.handle = nullptr;
    blas.buffer = nullptr;
    if (blas.allocation) { vmaFreeMemory(vmaAllocator, blas.allocation); blas.allocation = VK_NULL_HANDLE; }
  }
  blasStructures.clear();
  tlasStructure.handle = nullptr;
  tlasStructure.buffer = nullptr;
  if (tlasStructure.allocation) { vmaFreeMemory(vmaAllocator, tlasStructure.allocation); tlasStructure.allocation = VK_NULL_HANDLE; }

  // Free geometry/material info buffers
  geometryInfoBuffer = nullptr;
  if (geometryInfoAllocation) { vmaFreeMemory(vmaAllocator, geometryInfoAllocation); geometryInfoAllocation = VK_NULL_HANDLE; }
  materialBuffer = nullptr;
  if (materialAllocation) { vmaFreeMemory(vmaAllocator, materialAllocation); materialAllocation = VK_NULL_HANDLE; }

  // 8) Light storage buffers
  for (auto& lsb : lightStorageBuffers) {
    lsb.buffer = nullptr;
    if (lsb.allocation) { vmaFreeMemory(vmaAllocator, lsb.allocation); lsb.allocation = VK_NULL_HANDLE; }
    lsb.mapped = nullptr;
  }
  lightStorageBuffers.clear();

  // 8.1) Mesh resources
  for (auto& kv : meshResources) {
    auto& mr = kv.second;
    mr.vertexBuffer = nullptr;
    if (mr.vertexBufferAllocation) { vmaFreeMemory(vmaAllocator, mr.vertexBufferAllocation); mr.vertexBufferAllocation = VK_NULL_HANDLE; }
    mr.indexBuffer = nullptr;
    if (mr.indexBufferAllocation) { vmaFreeMemory(vmaAllocator, mr.indexBufferAllocation); mr.indexBufferAllocation = VK_NULL_HANDLE; }
  }
  meshResources.clear();

  // 9) Command buffers/pools
  commandBuffers.clear();
  commandPool = nullptr;
  computeCommandPool = nullptr;

  // 10) Sync objects
  imageAvailableSemaphores.clear();
  renderFinishedSemaphores.clear();
  inFlightFences.clear();
  uploadsTimeline = nullptr;

  // 11) Queues and surface (RAII handles will release upon reset; keep device alive until the end)
  graphicsQueue = nullptr;
  presentQueue = nullptr;
  computeQueue = nullptr;
  transferQueue = nullptr;
  surface = nullptr;

  // 12) VMA allocator last (after all buffers/images have been destroyed)
  if (vmaAllocator) {
    vmaDestroyAllocator(vmaAllocator);
    vmaAllocator = VK_NULL_HANDLE;
  }

  // Finally mark uninitialized
  initialized = false;
  std::cout << "Renderer cleanup completed." << std::endl;
}

// Create instance
bool Renderer::createInstance(const std::string& appName, bool enableValidationLayers) {
  try {
    // Create application info
    vk::ApplicationInfo appInfo{
      .pApplicationName = appName.c_str(),
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "Simple Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_3
    };

    // Get required extensions
    std::vector<const char *> extensions;

    // Add required extensions for GLFW
#if defined(PLATFORM_DESKTOP)
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    extensions.insert(extensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);
#endif

    // Gracefully disable validation layers when they are not installed
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      std::cerr << "Validation layers requested, but not available -- continuing without them" << std::endl;
      enableValidationLayers = false;
    }

    // Add debug extension if validation layers are enabled
    if (enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Create instance info
    vk::InstanceCreateInfo createInfo{
      .pApplicationInfo = &appInfo,
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data()
    };

    // Enable validation layers if requested
    vk::ValidationFeaturesEXT validationFeatures{};
    std::vector<vk::ValidationFeatureEnableEXT> enabledValidationFeatures;

    if (enableValidationLayers) {
      createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();

      // Keep validation output quiet by default (no DebugPrintf feature).
      // Ray Query debugPrintf/printf diagnostics are intentionally removed.

      validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(enabledValidationFeatures.size());
      validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures.data();

      createInfo.pNext = &validationFeatures;
    }

    // Create instance
    instance = vk::raii::Instance(context, createInfo);
    validationLayersEnabled = enableValidationLayers;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create instance: " << e.what() << std::endl;
    return false;
  }
}

// Setup debug messenger
bool Renderer::setupDebugMessenger(bool enableValidationLayers) {
  if (!enableValidationLayers) {
    return true;
  }

  try {
    // Create debug messenger info
    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

    // Select callback via simple platform macro: Android typically expects C PFN types in headers
    // while desktop (newer Vulkan-Hpp) expects vk:: types.
#if defined(__ANDROID__)
    createInfo.pfnUserCallback = &debugCallbackVkRaii;
#else
    createInfo.pfnUserCallback = &debugCallbackVkHpp;
#endif

    // Create debug messenger
    debugMessenger = vk::raii::DebugUtilsMessengerEXT(instance, createInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to set up debug messenger: " << e.what() << std::endl;
    return false;
  }
}

// Create surface
bool Renderer::createSurface() {
  try {
    // Create surface
    VkSurfaceKHR _surface;
    if (!platform->CreateVulkanSurface(*instance, &_surface)) {
      std::cerr << "Failed to create window surface" << std::endl;
      return false;
    }

    surface = vk::raii::SurfaceKHR(instance, _surface);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create surface: " << e.what() << std::endl;
    return false;
  }
}

// Pick a physical device
bool Renderer::pickPhysicalDevice() {
  try {
    // Get available physical devices
    std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

    if (devices.empty()) {
      std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
      return false;
    }

    // Prioritize discrete GPUs (like NVIDIA RTX 2080) over integrated GPUs (like Intel UHD Graphics)
    // First, collect all suitable devices with their suitability scores
    std::multimap<int, vk::raii::PhysicalDevice> suitableDevices;

    for (auto& _device : devices) {
      // Print device properties for debugging
      vk::PhysicalDeviceProperties deviceProperties = _device.getProperties();
      std::cout << "Checking device: " << deviceProperties.deviceName
          << " (Type: " << vk::to_string(deviceProperties.deviceType) << ")" << std::endl;

      // Check if the device supports Vulkan 1.3
      bool supportsVulkan1_3 = deviceProperties.apiVersion >= VK_API_VERSION_1_3;
      if (!supportsVulkan1_3) {
        std::cout << "  - Does not support Vulkan 1.3" << std::endl;
        continue;
      }

      // Check queue families
      QueueFamilyIndices indices = findQueueFamilies(_device);
      bool supportsGraphics = indices.isComplete();
      if (!supportsGraphics) {
        std::cout << "  - Missing required queue families" << std::endl;
        continue;
      }

      // Check device extensions
      bool supportsAllRequiredExtensions = checkDeviceExtensionSupport(_device);
      if (!supportsAllRequiredExtensions) {
        std::cout << "  - Missing required extensions" << std::endl;
        continue;
      }

      // Check swap chain support
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_device);
      bool swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
      if (!swapChainAdequate) {
        std::cout << "  - Inadequate swap chain support" << std::endl;
        continue;
      }

      // Check for required features
      auto features = _device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
      bool supportsRequiredFeatures = features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering;
      if (!supportsRequiredFeatures) {
        std::cout << "  - Does not support required features (dynamicRendering)" << std::endl;
        continue;
      }

      // Calculate suitability score - prioritize discrete GPUs
      int score = 0;

      // Discrete GPUs get the highest priority (NVIDIA RTX 2080, AMD, etc.)
      if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        score += 1000;
        std::cout << "  - Discrete GPU: +1000 points" << std::endl;
      }
      // Integrated GPUs get lower priority (Intel UHD Graphics, etc.)
      else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
        score += 100;
        std::cout << "  - Integrated GPU: +100 points" << std::endl;
      }

      // Add points for memory size (more VRAM is better)
      vk::PhysicalDeviceMemoryProperties memProperties = _device.getMemoryProperties();
      for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
        if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
          // Add 1 point per GB of VRAM
          score += static_cast<int>(memProperties.memoryHeaps[i].size / (1024 * 1024 * 1024));
          break;
        }
      }

      std::cout << "  - Device is suitable with score: " << score << std::endl;
      suitableDevices.emplace(score, _device);
    }

    if (!suitableDevices.empty()) {
      // Select the device with the highest score (discrete GPU with most VRAM)
      physicalDevice = suitableDevices.rbegin()->second;
      vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
      std::cout << "Selected device: " << deviceProperties.deviceName
          << " (Type: " << vk::to_string(deviceProperties.deviceType)
          << ", Score: " << suitableDevices.rbegin()->first << ")" << std::endl;

      // Store queue family indices for the selected device
      queueFamilyIndices = findQueueFamilies(physicalDevice);

      // Log capability matrix against the SimpleEngine baseline profile
      checkAndLogVulkanProfile();

      // Add supported optional extensions
      addSupportedOptionalExtensions();

      return true;
    }
    std::cerr << "Failed to find a suitable GPU. Make sure your GPU supports Vulkan and has the required extensions." << std::endl;
    return false;
  } catch (const std::exception& e) {
    std::cerr << "Failed to pick physical device: " << e.what() << std::endl;
    return false;
  }
}

// Add supported optional extensions
void Renderer::addSupportedOptionalExtensions() {
  try {
    // Get available extensions
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    // Build a set of available extension names for quick lookup
    std::set<std::string> avail;
    for (const auto& e : availableExtensions) {
      avail.insert(e.extensionName);
    }

    for (const auto& optionalExt : optionalDeviceExtensions) {
      if (avail.contains(optionalExt)) {
        deviceExtensions.push_back(optionalExt);
        std::cout << "Adding optional extension: " << optionalExt << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to add optional extensions: " << e.what() << std::endl;
  }
}

// Create logical device
bool Renderer::createLogicalDevice(bool enableValidationLayers) {
  try {
    // Create queue create info for each unique queue family
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set uniqueQueueFamilies = {
      queueFamilyIndices.graphicsFamily.value(),
      queueFamilyIndices.presentFamily.value(),
      queueFamilyIndices.computeFamily.value(),
      queueFamilyIndices.transferFamily.value()
    };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueCreateInfo{
        .queueFamilyIndex = queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
      };
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // Query supported features before enabling them
    auto supportedFeatures = physicalDevice.getFeatures2<
      vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceTimelineSemaphoreFeatures,
      vk::PhysicalDeviceVulkanMemoryModelFeatures,
      vk::PhysicalDeviceBufferDeviceAddressFeatures,
      vk::PhysicalDevice8BitStorageFeatures,
      vk::PhysicalDeviceVulkan11Features,
      vk::PhysicalDeviceVulkan13Features>();

    // Verify critical features are supported
    const auto& coreSupported = supportedFeatures.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& timelineSupported = supportedFeatures.get<vk::PhysicalDeviceTimelineSemaphoreFeatures>();
    const auto& memoryModelSupported = supportedFeatures.get<vk::PhysicalDeviceVulkanMemoryModelFeatures>();
    const auto& bufferAddressSupported = supportedFeatures.get<vk::PhysicalDeviceBufferDeviceAddressFeatures>();
    const auto& storage8BitSupported = supportedFeatures.get<vk::PhysicalDevice8BitStorageFeatures>();
    const auto& vulkan11Supported = supportedFeatures.get<vk::PhysicalDeviceVulkan11Features>();
    const auto& vulkan13Supported = supportedFeatures.get<vk::PhysicalDeviceVulkan13Features>();

    // Check for required features
    if (!coreSupported.samplerAnisotropy ||
      !timelineSupported.timelineSemaphore ||
      !memoryModelSupported.vulkanMemoryModel ||
      !bufferAddressSupported.bufferDeviceAddress ||
      !vulkan11Supported.shaderDrawParameters ||
      !vulkan13Supported.dynamicRendering ||
      !vulkan13Supported.synchronization2) {
      throw std::runtime_error("Required Vulkan features not supported by physical device");
    }

    // Enable required features (now verified to be supported)
    auto features = physicalDevice.getFeatures2();
    features.features.samplerAnisotropy = vk::True;
    features.features.depthBiasClamp = coreSupported.depthBiasClamp ? vk::True : vk::False;

    // Explicitly configure device features to prevent validation layer warnings
    // These features are required by extensions or other features, so we enable them explicitly

    // Timeline semaphore features (required for synchronization2)
    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures;
    timelineSemaphoreFeatures.timelineSemaphore = vk::True;

    // Vulkan memory model features (required for some shader operations)
    vk::PhysicalDeviceVulkanMemoryModelFeatures memoryModelFeatures;
    memoryModelFeatures.vulkanMemoryModel = vk::True;
    memoryModelFeatures.vulkanMemoryModelDeviceScope = memoryModelSupported.vulkanMemoryModelDeviceScope ? vk::True : vk::False;

    // Buffer device address features (required for some buffer operations)
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures;
    bufferDeviceAddressFeatures.bufferDeviceAddress = vk::True;

    // 8-bit storage features (required for some shader storage operations)
    vk::PhysicalDevice8BitStorageFeatures storage8BitFeatures;
    storage8BitFeatures.storageBuffer8BitAccess = storage8BitSupported.storageBuffer8BitAccess ? vk::True : vk::False;

    // Enable Vulkan 1.3 features
    vk::PhysicalDeviceVulkan13Features vulkan13Features;
    vulkan13Features.dynamicRendering = vk::True;
    vulkan13Features.synchronization2 = vk::True;

    // Vulkan 1.1 features: shaderDrawParameters to satisfy SPIR-V DrawParameters capability
    vk::PhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.shaderDrawParameters = vk::True;
    // Query extended feature support
#if !defined(PLATFORM_ANDROID)
    auto featureChain = physicalDevice.getFeatures2<
      vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceDescriptorIndexingFeatures,
      vk::PhysicalDeviceRobustness2FeaturesEXT,
      vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR,
      vk::PhysicalDeviceShaderTileImageFeaturesEXT,
      vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
      vk::PhysicalDeviceRayQueryFeaturesKHR>();
    const auto& localReadSupported = featureChain.get<vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>();
    const auto& tileImageSupported = featureChain.get<vk::PhysicalDeviceShaderTileImageFeaturesEXT>();
#else
    auto featureChain = physicalDevice.getFeatures2<
      vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceDescriptorIndexingFeatures,
      vk::PhysicalDeviceRobustness2FeaturesEXT,
      vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
      vk::PhysicalDeviceRayQueryFeaturesKHR>();
#endif
    const auto& coreFeaturesSupported = featureChain.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& indexingFeaturesSupported = featureChain.get<vk::PhysicalDeviceDescriptorIndexingFeatures>();
    const auto& robust2Supported = featureChain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>();
    const auto& accelerationStructureSupported = featureChain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
    const auto& rayQuerySupported = featureChain.get<vk::PhysicalDeviceRayQueryFeaturesKHR>();

    // Ray Query shader uses indexing into a (large) sampled-image array.
    // Some drivers require this core feature to be explicitly enabled.
    if (coreFeaturesSupported.shaderSampledImageArrayDynamicIndexing) {
      features.features.shaderSampledImageArrayDynamicIndexing = vk::True;
    }

    // Prepare descriptor indexing features to enable if supported
    vk::PhysicalDeviceDescriptorIndexingFeatures indexingFeaturesEnable{};
    descriptorIndexingEnabled = false;
    // Enable non-uniform indexing of sampled image arrays when supported — required for
    // `NonUniformResourceIndex()` in the ray-query shader to actually take effect.
    if (indexingFeaturesSupported.shaderSampledImageArrayNonUniformIndexing) {
      indexingFeaturesEnable.shaderSampledImageArrayNonUniformIndexing = vk::True;
      descriptorIndexingEnabled = true;
    }

    // These are not strictly required when writing a fully-populated descriptor array,
    // but enabling them when available avoids edge-case driver behavior for large arrays.
    if (descriptorIndexingEnabled) {
      if (indexingFeaturesSupported.descriptorBindingPartiallyBound) {
        indexingFeaturesEnable.descriptorBindingPartiallyBound = vk::True;
      }
      if (indexingFeaturesSupported.descriptorBindingUpdateUnusedWhilePending) {
        indexingFeaturesEnable.descriptorBindingUpdateUnusedWhilePending = vk::True;
      }
    }
    // Optionally enable UpdateAfterBind flags when supported (not strictly required for RQ textures)
    if (indexingFeaturesSupported.descriptorBindingSampledImageUpdateAfterBind)
      indexingFeaturesEnable.descriptorBindingSampledImageUpdateAfterBind = vk::True;
    if (indexingFeaturesSupported.descriptorBindingUniformBufferUpdateAfterBind)
      indexingFeaturesEnable.descriptorBindingUniformBufferUpdateAfterBind = vk::True;
    if (indexingFeaturesSupported.descriptorBindingUpdateUnusedWhilePending)
      indexingFeaturesEnable.descriptorBindingUpdateUnusedWhilePending = vk::True;

    // Helper to check if an extension is enabled (using string comparison)
    auto hasExtension = [&](const char* name) {
      return std::find_if(deviceExtensions.begin(),
                          deviceExtensions.end(),
                          [&](const char* ext) {
                            return std::strcmp(ext, name) == 0;
                          }) != deviceExtensions.end();
    };

    // Prepare Robustness2 features if the extension is enabled and device supports
    auto hasRobust2 = hasExtension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    vk::PhysicalDeviceRobustness2FeaturesEXT robust2Enable{};
    if (hasRobust2) {
      if (robust2Supported.robustBufferAccess2)
        robust2Enable.robustBufferAccess2 = vk::True;
      if (robust2Supported.robustImageAccess2)
        robust2Enable.robustImageAccess2 = vk::True;
      if (robust2Supported.nullDescriptor)
        robust2Enable.nullDescriptor = vk::True;
    }

#if !defined(PLATFORM_ANDROID)
    // Prepare Dynamic Rendering Local Read features if extension is enabled and supported
    auto hasLocalRead = hasExtension(VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME);
    vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR localReadEnable{};
    if (hasLocalRead && localReadSupported.dynamicRenderingLocalRead) {
      localReadEnable.dynamicRenderingLocalRead = vk::True;
    }

    // Prepare Shader Tile Image features if extension is enabled and supported
    auto hasTileImage = hasExtension(VK_EXT_SHADER_TILE_IMAGE_EXTENSION_NAME);
    vk::PhysicalDeviceShaderTileImageFeaturesEXT tileImageEnable{};
    if (hasTileImage) {
      if (tileImageSupported.shaderTileImageColorReadAccess)
        tileImageEnable.shaderTileImageColorReadAccess = vk::True;
      if (tileImageSupported.shaderTileImageDepthReadAccess)
        tileImageEnable.shaderTileImageDepthReadAccess = vk::True;
      if (tileImageSupported.shaderTileImageStencilReadAccess)
        tileImageEnable.shaderTileImageStencilReadAccess = vk::True;
    }
#endif

    // Prepare Acceleration Structure features if extension is enabled and supported
    auto hasAccelerationStructure = hasExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureEnable{};
    if (hasAccelerationStructure && accelerationStructureSupported.accelerationStructure) {
      accelerationStructureEnable.accelerationStructure = vk::True;
    }

    // Prepare Ray Query features if extension is enabled and supported
    auto hasRayQuery = hasExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryEnable{};
    if (hasRayQuery && rayQuerySupported.rayQuery) {
      rayQueryEnable.rayQuery = vk::True;
    }

    // Chain the feature structures together (build pNext chain explicitly)
    // Base
    features.pNext = &timelineSemaphoreFeatures;
    timelineSemaphoreFeatures.pNext = &memoryModelFeatures;
    memoryModelFeatures.pNext = &bufferDeviceAddressFeatures;
    bufferDeviceAddressFeatures.pNext = &storage8BitFeatures;
    storage8BitFeatures.pNext = &vulkan11Features; // link 1.1 first
    vulkan11Features.pNext = &vulkan13Features; // then 1.3 features

    // Build tail chain starting at Vulkan 1.3 features
    void** tailNext = reinterpret_cast<void **>(&vulkan13Features.pNext);
    if (descriptorIndexingEnabled) {
      *tailNext = &indexingFeaturesEnable;
      tailNext = reinterpret_cast<void **>(&indexingFeaturesEnable.pNext);
    }
    if (hasRobust2) {
      *tailNext = &robust2Enable;
      tailNext = reinterpret_cast<void **>(&robust2Enable.pNext);
    }
#if !defined(PLATFORM_ANDROID)
    if (hasLocalRead) {
      *tailNext = &localReadEnable;
      tailNext = reinterpret_cast<void **>(&localReadEnable.pNext);
    }
    if (hasTileImage) {
      *tailNext = &tileImageEnable;
      tailNext = reinterpret_cast<void **>(&tileImageEnable.pNext);
    }
#endif
    if (hasAccelerationStructure) {
      *tailNext = &accelerationStructureEnable;
      tailNext = reinterpret_cast<void **>(&accelerationStructureEnable.pNext);
    }
    if (hasRayQuery) {
      *tailNext = &rayQueryEnable;
      tailNext = reinterpret_cast<void **>(&rayQueryEnable.pNext);
    }

    // Record which features ended up enabled (for runtime decisions/tutorial diagnostics)
    robustness2Enabled = hasRobust2 && (robust2Enable.robustBufferAccess2 == vk::True ||
      robust2Enable.robustImageAccess2 == vk::True ||
      robust2Enable.nullDescriptor == vk::True);
#if !defined(PLATFORM_ANDROID)
    dynamicRenderingLocalReadEnabled = hasLocalRead && (localReadEnable.dynamicRenderingLocalRead == vk::True);
    shaderTileImageEnabled = hasTileImage && (tileImageEnable.shaderTileImageColorReadAccess == vk::True ||
      tileImageEnable.shaderTileImageDepthReadAccess == vk::True ||
      tileImageEnable.shaderTileImageStencilReadAccess == vk::True);
#else
    dynamicRenderingLocalReadEnabled = false;
    shaderTileImageEnabled = false;
#endif
    accelerationStructureEnabled = hasAccelerationStructure && (accelerationStructureEnable.accelerationStructure == vk::True);
    rayQueryEnabled = hasRayQuery && (rayQueryEnable.rayQuery == vk::True);

    // One-time startup diagnostics (Ray Query + texture array indexing)
    static bool printedFeatureDiag = false;
    if (!printedFeatureDiag) {
      printedFeatureDiag = true;
      std::cout << "[DeviceFeatures] shaderSampledImageArrayDynamicIndexing="
          << (features.features.shaderSampledImageArrayDynamicIndexing == vk::True ? "ON" : "OFF")
          << ", shaderSampledImageArrayNonUniformIndexing="
          << (indexingFeaturesEnable.shaderSampledImageArrayNonUniformIndexing == vk::True ? "ON" : "OFF")
          << ", descriptorIndexingEnabled="
          << (descriptorIndexingEnabled ? "true" : "false")
          << "\n";
    }

    // Create a device. Device layers are deprecated and ignored, so we
    // only configure extensions and features here; validation is enabled
    // via instance layers.
    vk::DeviceCreateInfo createInfo{
      .pNext = &features,
      .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
      .pQueueCreateInfos = queueCreateInfos.data(),
      .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
      .ppEnabledExtensionNames = deviceExtensions.data(),
      .pEnabledFeatures = nullptr // Using pNext for features
    };

    // Create the logical device
    device = vk::raii::Device(physicalDevice, createInfo);

    // Get queue handles
    graphicsQueue = vk::raii::Queue(device, queueFamilyIndices.graphicsFamily.value(), 0);
    presentQueue = vk::raii::Queue(device, queueFamilyIndices.presentFamily.value(), 0);
    computeQueue = vk::raii::Queue(device, queueFamilyIndices.computeFamily.value(), 0);
    transferQueue = vk::raii::Queue(device, queueFamilyIndices.transferFamily.value(), 0);

    // Create global timeline semaphore for uploads early (needed before default texture creation)
    vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> timelineChain(
      {},
      {.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0});
    uploadsTimeline = vk::raii::Semaphore(device, timelineChain.get<vk::SemaphoreCreateInfo>());
    uploadTimelineLastSubmitted.store(0, std::memory_order_relaxed);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create logical device: " << e.what() << std::endl;
    return false;
  }
}

// Check validation layer support
bool Renderer::checkValidationLayerSupport() const {
  // Get available layers
  std::vector<vk::LayerProperties> availableLayers = context.enumerateInstanceLayerProperties();

  // Check if all requested layers are available
  for (const char* layerName : validationLayers) {
    bool layerFound = false;

    for (const auto& layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Vulkan Profile capability check — logs a compatibility matrix against the
// SimpleEngine baseline profile requirements defined in
// vulkan_profiles/SimpleEngine_baseline.json.
// Called once after device selection so problems are surfaced at startup.
// ---------------------------------------------------------------------------
void Renderer::checkAndLogVulkanProfile() {
  try {
    auto props = physicalDevice.getProperties();
    std::cout << "\n=== Vulkan Profile Check [VP_SIMPLEENGINE_BASELINE_1_0] ===" << std::endl;
    std::cout << "  Device : " << props.deviceName << std::endl;
    std::cout << "  API    : "
              << VK_VERSION_MAJOR(props.apiVersion) << "."
              << VK_VERSION_MINOR(props.apiVersion) << "."
              << VK_VERSION_PATCH(props.apiVersion) << std::endl;

    // Query all relevant feature structs in one call
    auto chain = physicalDevice.getFeatures2<
      vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan12Features,
      vk::PhysicalDeviceVulkan13Features>();

    auto& f10 = chain.get<vk::PhysicalDeviceFeatures2>().features;
    auto& f12 = chain.get<vk::PhysicalDeviceVulkan12Features>();
    auto& f13 = chain.get<vk::PhysicalDeviceVulkan13Features>();

    auto& limits = props.limits;

    bool ok = true;
    auto check = [&](const char* name, bool have, bool required) {
      std::cout << "  " << (have ? "[+]" : (required ? "[!]" : "[-]"))
                << " " << name << " : " << (have ? "OK" : (required ? "MISSING" : "optional")) << "\n";
      if (required && !have) ok = false;
    };

    check("Vulkan 1.3 API",            props.apiVersion >= VK_API_VERSION_1_3, true);
    check("samplerAnisotropy",         (bool)f10.samplerAnisotropy,   true);
    check("fillModeNonSolid",          (bool)f10.fillModeNonSolid,    true);
    check("shaderInt64",               (bool)f10.shaderInt64,         false);
    check("multiDrawIndirect",         (bool)f10.multiDrawIndirect,   false);
    check("descriptorIndexing",                        (bool)f12.descriptorIndexing,                        true);
    check("shaderSampledImageArrayNonUniformIndexing", (bool)f12.shaderSampledImageArrayNonUniformIndexing, true);
    check("runtimeDescriptorArray",                   (bool)f12.runtimeDescriptorArray,                    true);
    check("descriptorBindingPartiallyBound",           (bool)f12.descriptorBindingPartiallyBound,           true);
    check("descriptorBindingVariableDescriptorCount",  (bool)f12.descriptorBindingVariableDescriptorCount,  false);
    check("timelineSemaphore",                         (bool)f12.timelineSemaphore,                         true);
    check("bufferDeviceAddress",                       (bool)f12.bufferDeviceAddress,                       false);
    check("dynamicRendering",          (bool)f13.dynamicRendering,    true);
    check("synchronization2",          (bool)f13.synchronization2,    true);
    check("maxPushConstantsSize >= 128",          limits.maxPushConstantsSize >= 128,          true);
    check("maxDescriptorSetSamplers >= 512",      limits.maxDescriptorSetSamplers >= 512,      true);
    check("maxDescriptorSetSampledImages >= 512", limits.maxDescriptorSetSampledImages >= 512, true);
    check("maxDescriptorSetStorageBuffers >= 64", limits.maxDescriptorSetStorageBuffers >= 64, true);
    check("maxComputeWorkGroupInvocations >= 256",limits.maxComputeWorkGroupInvocations >= 256,true);

    // Optional ray-tracing profile
    auto availExts = physicalDevice.enumerateDeviceExtensionProperties();
    auto hasExt = [&](const char* name) {
      for (auto& e : availExts) if (strcmp(e.extensionName, name) == 0) return true;
      return false;
    };
    check("VK_KHR_acceleration_structure (RT optional)", hasExt("VK_KHR_acceleration_structure"), false);
    check("VK_KHR_ray_query (RT optional)",              hasExt("VK_KHR_ray_query"),              false);

    std::cout << "\n  Profile result: "
              << (ok ? "PASS — device meets VP_SIMPLEENGINE_BASELINE_1_0 requirements."
                     : "FAIL — one or more required capabilities are missing.")
              << "\n=========================================================\n" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "checkAndLogVulkanProfile: " << e.what() << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Debug label / object-name helpers (VK_EXT_debug_utils).
// No-ops when the extension loader has not resolved the entry points.
// ---------------------------------------------------------------------------

void Renderer::BeginDebugLabel(vk::raii::CommandBuffer& cmd, const char* name,
                               float r, float g, float b, float a) {
  if (!VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT) return;
  vk::DebugUtilsLabelEXT label{};
  label.pLabelName = name;
  label.color[0] = r; label.color[1] = g; label.color[2] = b; label.color[3] = a;
  cmd.beginDebugUtilsLabelEXT(label);
}

void Renderer::EndDebugLabel(vk::raii::CommandBuffer& cmd) {
  if (!VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndDebugUtilsLabelEXT) return;
  cmd.endDebugUtilsLabelEXT();
}

void Renderer::InsertDebugLabel(vk::raii::CommandBuffer& cmd, const char* name,
                                float r, float g, float b, float a) {
  if (!VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdInsertDebugUtilsLabelEXT) return;
  vk::DebugUtilsLabelEXT label{};
  label.pLabelName = name;
  label.color[0] = r; label.color[1] = g; label.color[2] = b; label.color[3] = a;
  cmd.insertDebugUtilsLabelEXT(label);
}

void Renderer::SetDebugName(vk::Buffer buffer, const char* name) {
  if (!VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT || !*device) return;
  vk::DebugUtilsObjectNameInfoEXT info{};
  info.objectType   = vk::ObjectType::eBuffer;
  info.objectHandle = reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer));
  info.pObjectName  = name;
  device.setDebugUtilsObjectNameEXT(info);
}

void Renderer::SetDebugName(vk::Image image, const char* name) {
  if (!VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT || !*device) return;
  vk::DebugUtilsObjectNameInfoEXT info{};
  info.objectType   = vk::ObjectType::eImage;
  info.objectHandle = reinterpret_cast<uint64_t>(static_cast<VkImage>(image));
  info.pObjectName  = name;
  device.setDebugUtilsObjectNameEXT(info);
}

void Renderer::SetDebugName(vk::Pipeline pipeline, const char* name) {
  if (!VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT || !*device) return;
  vk::DebugUtilsObjectNameInfoEXT info{};
  info.objectType   = vk::ObjectType::ePipeline;
  info.objectHandle = reinterpret_cast<uint64_t>(static_cast<VkPipeline>(pipeline));
  info.pObjectName  = name;
  device.setDebugUtilsObjectNameEXT(info);
}