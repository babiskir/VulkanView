/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan_raii.hpp>

// Forward declarations
class Renderer;
class CameraComponent;
class WSTessendorf;

/**
 * @brief Water simulation and rendering system.
 *
 * Owns the Tessendorf FFT ocean simulation, the GPU mesh, displacement/normal
 * textures, per-frame UBOs and descriptor sets, and the Vulkan pipeline.
 * The pipeline (waterPipeline / waterPipelineLayout / waterDescriptorSetLayout)
 * lives in the Renderer; WaterSystem stores only a back-pointer to Renderer.
 */
class WaterSystem {
public:
    // -------------------------------------------------------------------------
    // Water simulation / visual parameters (exposed for ImGui)
    float heightAmplitude = 2.5f;
    float choppiness      = 1.5f;
    float textureScale    = 0.5f;
    int   tileResolution  = 256;        // grid quads along one axis
    float windSpeed       = 30.0f;
    float windAngleDeg    = 45.0f;      // degrees, converted to direction on update
    float waterDepth      = 50.0f;      // height parameter passed to frag shader
    float skyIntensity        = 1.0f;
    float specularIntensity   = 1.0f;
    float specularHighlights  = 32.0f;
    float sunIntensity        = 1.0f;

    // -------------------------------------------------------------------------
    WaterSystem();
    ~WaterSystem();

    /**
     * @brief Allocate all GPU resources and prepare simulation.
     * @param renderer  Back-pointer to Renderer (not owned).
     * @return true on success.
     */
    bool Initialize(Renderer* renderer);

    /**
     * @brief Release all GPU resources.
     */
    void Cleanup();

    /**
     * @brief Advance simulation by deltaTime seconds and upload new textures.
     *  Must be called from the main / render thread.
     */
    void Update(float deltaTime, CameraComponent* camera);

    /**
     * @brief Upload wave textures and transition layouts — call OUTSIDE a renderpass.
     */
    void UploadTextures(vk::raii::CommandBuffer& cmd, uint32_t frameIndex);

    /**
     * @brief Bind pipeline and draw — call INSIDE a beginRendering block.
     */
    void Draw(vk::raii::CommandBuffer& cmd, uint32_t frameIndex);

    bool IsActive() const { return active; }
    void SetActive(bool a) { active = a; }

    // Structures that must match water.slang
    struct WaterVertexUBO {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
        float WSHeightAmp;
        float WSChoppy;
        float texScale;
        float _pad;
    };

    struct WaterFragUBO {
        alignas(16) glm::vec3 camPos;      float height;
        alignas(16) glm::vec3 absorpCoef;  float _p0;
        alignas(16) glm::vec3 scatterCoef; float _p1;
        alignas(16) glm::vec3 backscatterCoef; float _p2;
        alignas(16) glm::vec3 terrainColor; float _p3;
        float skyIntensity;
        float specularIntensity;
        float specularHighlights;
        uint32_t useEnvMap;   // 1 = sample EnvMap for reflections, 0 = Preetham
        alignas(16) glm::vec3 sunDirection; float sunIntensity;
        // Preetham sky
        alignas(16) glm::vec3 skyA; float turbidity;
        alignas(16) glm::vec3 skyB; float _p4;
        alignas(16) glm::vec3 skyC; float _p5;
        alignas(16) glm::vec3 skyD; float _p6;
        alignas(16) glm::vec3 skyE; float _p7;
        alignas(16) glm::vec3 ZenithLum;     float _p8;
        alignas(16) glm::vec3 ZeroThetaSun;  float _p9;
    };

    // Water vertex layout
    struct WaterVertex {
        glm::vec3 pos;
        glm::vec2 uv;
    };

private:
    bool active   = false;
    Renderer* renderer = nullptr;
    bool lastEnvTextureState = false;  // tracks HasEnvTexture() && GetUseEnvMapForSky() to refresh descriptors

    std::unique_ptr<WSTessendorf> simulation;

    // Per-frame GPU resources
    struct FrameData {
        // Displacement and normal maps (GPU images, vec4 RGBA32F)
        vk::raii::Image         dispImage{nullptr};
        vk::raii::DeviceMemory  dispMemory{nullptr};
        vk::raii::ImageView     dispView{nullptr};
        vk::raii::Image         normImage{nullptr};
        vk::raii::DeviceMemory  normMemory{nullptr};
        vk::raii::ImageView     normView{nullptr};
        // Shared staging buffer for both maps
        vk::raii::Buffer        stagingBuffer{nullptr};
        vk::raii::DeviceMemory  stagingMemory{nullptr};
        void*                   stagingMapped = nullptr;
        // Per-frame UBOs
        vk::raii::Buffer        vertUboBuffer{nullptr};
        vk::raii::DeviceMemory  vertUboMemory{nullptr};
        void*                   vertUboMapped = nullptr;
        vk::raii::Buffer        fragUboBuffer{nullptr};
        vk::raii::DeviceMemory  fragUboMemory{nullptr};
        void*                   fragUboMapped = nullptr;
        // Descriptor resources
        vk::raii::DescriptorPool descriptorPool{nullptr};
        vk::raii::DescriptorSet  descriptorSet{nullptr};
        // Track image layout so we know when to skip the transition
        vk::ImageLayout dispLayout = vk::ImageLayout::eUndefined;
        vk::ImageLayout normLayout = vk::ImageLayout::eUndefined;
    };
    std::vector<FrameData> frames;

    // Static water mesh
    vk::raii::Buffer        vertexBuffer{nullptr};
    vk::raii::DeviceMemory  vertexMemory{nullptr};
    vk::raii::Buffer        indexBuffer{nullptr};
    vk::raii::DeviceMemory  indexMemory{nullptr};
    uint32_t                indexCount = 0;

    vk::raii::Sampler textureSampler{nullptr};

    float simTime = 0.0f;

    // Camera state cached for UBO upload
    glm::mat4 cachedView{1.0f};
    glm::mat4 cachedProj{1.0f};
    glm::vec3 cachedCamPos{0.0f, 8.0f, -150.0f};

    // -------------------------------------------------------------------------
    // Private helpers

    bool createMesh(int resolution, float halfSize);
    bool createFrameResources(uint32_t framesInFlight);
    bool createTextureSampler();

    // Upload simulation output to GPU staging, then copy to device-local images
    void uploadWaveTextures(uint32_t frameIndex);

    // Record image layout transitions for wave textures
    void transitionWaveTexturesForSampling(vk::raii::CommandBuffer& cmd, uint32_t frameIndex);
    void transitionWaveTexturesForTransfer(vk::raii::CommandBuffer& cmd, uint32_t frameIndex);

    // Update a per-frame descriptor set with latest images
    void updateDescriptorSet(uint32_t frameIndex);

    // Vulkan helpers (delegate to renderer's public API)
    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
    allocateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags props);

    std::pair<vk::raii::Image, vk::raii::DeviceMemory>
    allocateImage(uint32_t w, uint32_t h, vk::Format fmt,
                  vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                  vk::MemoryPropertyFlags props);

    vk::raii::ImageView createImageView(vk::Image image, vk::Format fmt);

    void copyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size);
};
