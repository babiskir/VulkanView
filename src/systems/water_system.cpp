/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "water_system.h"

#include "tessendorf.h"
#include "renderer.h"
#include "camera_component.h"

#include <cmath>
#include <iostream>
#include <cstring>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Constructor / destructor

WaterSystem::WaterSystem()  = default;
WaterSystem::~WaterSystem() { Cleanup(); }

// ---------------------------------------------------------------------------
// Helpers – buffer / image allocation via raw Vulkan (using renderer's device)

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
WaterSystem::allocateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags props)
{
    vk::Device dev = renderer->GetDevice();
    vk::PhysicalDevice phys = renderer->GetPhysicalDevice();

    vk::BufferCreateInfo bufCI{
        .size        = size,
        .usage       = usage,
        .sharingMode = vk::SharingMode::eExclusive
    };
    vk::raii::Buffer buf(renderer->GetRaiiDevice(), bufCI);

    auto memReq = buf.getMemoryRequirements();
    auto memProps = phys.getMemoryProperties();

    // Find suitable memory type
    uint32_t memTypeIdx = renderer->FindMemoryType(memReq.memoryTypeBits, props);

    vk::MemoryAllocateInfo allocInfo{
        .allocationSize  = memReq.size,
        .memoryTypeIndex = memTypeIdx
    };
    vk::raii::DeviceMemory mem(renderer->GetRaiiDevice(), allocInfo);
    buf.bindMemory(*mem, 0);

    return { std::move(buf), std::move(mem) };
}

std::pair<vk::raii::Image, vk::raii::DeviceMemory>
WaterSystem::allocateImage(uint32_t w, uint32_t h, vk::Format fmt,
                            vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                            vk::MemoryPropertyFlags props)
{
    vk::PhysicalDevice phys = renderer->GetPhysicalDevice();

    vk::ImageCreateInfo imgCI{
        .imageType   = vk::ImageType::e2D,
        .format      = fmt,
        .extent      = {w, h, 1},
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = vk::SampleCountFlagBits::e1,
        .tiling      = tiling,
        .usage       = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined
    };
    vk::raii::Image img(renderer->GetRaiiDevice(), imgCI);

    auto memReq = img.getMemoryRequirements();
    uint32_t memTypeIdx = renderer->FindMemoryType(memReq.memoryTypeBits, props);

    vk::MemoryAllocateInfo allocInfo{
        .allocationSize  = memReq.size,
        .memoryTypeIndex = memTypeIdx
    };
    vk::raii::DeviceMemory mem(renderer->GetRaiiDevice(), allocInfo);
    img.bindMemory(*mem, 0);

    return { std::move(img), std::move(mem) };
}

vk::raii::ImageView WaterSystem::createImageView(vk::Image image, vk::Format fmt)
{
    vk::ImageViewCreateInfo viewCI{
        .image    = image,
        .viewType = vk::ImageViewType::e2D,
        .format   = fmt,
        .subresourceRange = {
            .aspectMask     = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1
        }
    };
    return vk::raii::ImageView(renderer->GetRaiiDevice(), viewCI);
}

// ---------------------------------------------------------------------------
// Initialize

bool WaterSystem::Initialize(Renderer* rend)
{
    renderer = rend;
    if (!renderer) return false;

    // Create and prepare Tessendorf simulation
    simulation = std::make_unique<WSTessendorf>(
        static_cast<uint32_t>(tileResolution),
        1000.0f
    );
    float angle = static_cast<float>(windAngleDeg * M_PI / 180.0);
    simulation->SetWindDirection(glm::vec2(std::cos(angle), std::sin(angle)));
    simulation->SetWindSpeed(windSpeed);
    simulation->SetLambda(-1.0f); // choppy factor sign convention
    simulation->Prepare();

    const uint32_t framesInFlight = renderer->GetMaxFramesInFlight();

    if (!createTextureSampler())     return false;
    if (!createFrameResources(framesInFlight)) return false;
    if (!createMesh(tileResolution, 500.0f))   return false;

    active = true;
    std::cout << "[WaterSystem] Initialized successfully\n";
    return true;
}

// ---------------------------------------------------------------------------
// Create texture sampler

bool WaterSystem::createTextureSampler()
{
    vk::SamplerCreateInfo samplerCI{
        .magFilter    = vk::Filter::eLinear,
        .minFilter    = vk::Filter::eLinear,
        .mipmapMode   = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .anisotropyEnable = vk::False,
        .maxAnisotropy    = 1.0f,
        .compareEnable    = vk::False,
        .minLod  = 0.0f,
        .maxLod  = 0.0f
    };
    textureSampler = vk::raii::Sampler(renderer->GetRaiiDevice(), samplerCI);
    return true;
}

// ---------------------------------------------------------------------------
// Per-frame GPU resources

bool WaterSystem::createFrameResources(uint32_t framesInFlight)
{
    const uint32_t N       = static_cast<uint32_t>(tileResolution);
    const uint32_t texSize = N * N * sizeof(glm::vec4); // RGBA32F per pixel
    const vk::Format mapFmt = vk::Format::eR32G32B32A32Sfloat;

    frames.resize(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; ++i)
    {
        auto& f = frames[i];

        // Displacement image
        {
            auto [img, mem] = allocateImage(N, N, mapFmt,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            f.dispImage  = std::move(img);
            f.dispMemory = std::move(mem);
            f.dispView   = createImageView(*f.dispImage, mapFmt);
            f.dispLayout = vk::ImageLayout::eUndefined;
        }

        // Normal image
        {
            auto [img, mem] = allocateImage(N, N, mapFmt,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            f.normImage  = std::move(img);
            f.normMemory = std::move(mem);
            f.normView   = createImageView(*f.normImage, mapFmt);
            f.normLayout = vk::ImageLayout::eUndefined;
        }

        // Staging buffer for both textures (2 * N*N * sizeof(vec4))
        {
            const vk::DeviceSize stagingSize = 2 * texSize;
            auto [buf, mem] = allocateBuffer(stagingSize,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            f.stagingBuffer = std::move(buf);
            f.stagingMemory = std::move(mem);
            f.stagingMapped = f.stagingMemory.mapMemory(0, stagingSize);
        }

        // Vertex UBO
        {
            auto [buf, mem] = allocateBuffer(sizeof(WaterVertexUBO),
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            f.vertUboBuffer = std::move(buf);
            f.vertUboMemory = std::move(mem);
            f.vertUboMapped = f.vertUboMemory.mapMemory(0, sizeof(WaterVertexUBO));
        }

        // Fragment UBO
        {
            auto [buf, mem] = allocateBuffer(sizeof(WaterFragUBO),
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            f.fragUboBuffer = std::move(buf);
            f.fragUboMemory = std::move(mem);
            f.fragUboMapped = f.fragUboMemory.mapMemory(0, sizeof(WaterFragUBO));
        }

        // Descriptor pool (5 descriptors: 2 UBOs + 3 samplers — disp, normal, env)
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {{
            {vk::DescriptorType::eUniformBuffer,        2},
            {vk::DescriptorType::eCombinedImageSampler, 3}
        }};
        vk::DescriptorPoolCreateInfo poolCI{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = 1,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes    = poolSizes.data()
        };
        f.descriptorPool = vk::raii::DescriptorPool(renderer->GetRaiiDevice(), poolCI);

        // Allocate descriptor set
        vk::DescriptorSetLayout dsl = renderer->GetWaterDescriptorSetLayout();
        vk::DescriptorSetAllocateInfo dsAllocInfo{
            .descriptorPool     = *f.descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &dsl
        };
        vk::raii::DescriptorSets dsets(renderer->GetRaiiDevice(), dsAllocInfo);
        f.descriptorSet = std::move(dsets[0]);

        // Write descriptor set
        updateDescriptorSet(i);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Update a descriptor set with current images

void WaterSystem::updateDescriptorSet(uint32_t idx)
{
    auto& f = frames[idx];

    vk::DescriptorBufferInfo vertUboInfo{
        .buffer = *f.vertUboBuffer,
        .offset = 0,
        .range  = sizeof(WaterVertexUBO)
    };
    vk::DescriptorBufferInfo fragUboInfo{
        .buffer = *f.fragUboBuffer,
        .offset = 0,
        .range  = sizeof(WaterFragUBO)
    };
    vk::DescriptorImageInfo dispImgInfo{
        .sampler     = *textureSampler,
        .imageView   = *f.dispView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };
    vk::DescriptorImageInfo normImgInfo{
        .sampler     = *textureSampler,
        .imageView   = *f.normView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };

    // Binding 4: environment map for reflections (use default if not loaded yet)
    vk::ImageView  envView    = renderer->GetEnvImageView();
    vk::Sampler    envSmplr   = renderer->GetEnvSampler();
    // Fall back to displacement map view/sampler when env texture not loaded
    if (!envView)    envView  = *f.dispView;
    if (!envSmplr)   envSmplr = *textureSampler;
    vk::DescriptorImageInfo envImgInfo{
        .sampler     = envSmplr,
        .imageView   = envView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };

    std::array<vk::WriteDescriptorSet, 5> writes = {{
        {.dstSet = *f.descriptorSet, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .pBufferInfo = &vertUboInfo},
        {.dstSet = *f.descriptorSet, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .pBufferInfo = &fragUboInfo},
        {.dstSet = *f.descriptorSet, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &dispImgInfo},
        {.dstSet = *f.descriptorSet, .dstBinding = 3, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &normImgInfo},
        {.dstSet = *f.descriptorSet, .dstBinding = 4, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &envImgInfo},
    }};
    renderer->GetRaiiDevice().updateDescriptorSets(writes, {});
}

// ---------------------------------------------------------------------------
// Create water mesh grid

bool WaterSystem::createMesh(int resolution, float halfSize)
{
    // (resolution+1)^2 vertices
    const int verts = (resolution + 1);
    std::vector<WaterVertex> vertices;
    vertices.reserve(static_cast<size_t>(verts * verts));

    const float step = (2.0f * halfSize) / static_cast<float>(resolution);

    for (int z = 0; z <= resolution; ++z)
    for (int x = 0; x <= resolution; ++x)
    {
        float px = -halfSize + x * step;
        float pz = -halfSize + z * step;
        float u  = static_cast<float>(x) / static_cast<float>(resolution);
        float v  = static_cast<float>(z) / static_cast<float>(resolution);
        vertices.push_back({ glm::vec3(px, 0.0f, pz), glm::vec2(u, v) });
    }

    std::vector<uint32_t> indices;
    indices.reserve(static_cast<size_t>(resolution * resolution * 6));

    for (int z = 0; z < resolution; ++z)
    for (int x = 0; x < resolution; ++x)
    {
        uint32_t tl = z * verts + x;
        uint32_t tr = tl + 1;
        uint32_t bl = (z + 1) * verts + x;
        uint32_t br = bl + 1;
        indices.push_back(tl); indices.push_back(bl); indices.push_back(tr);
        indices.push_back(tr); indices.push_back(bl); indices.push_back(br);
    }
    indexCount = static_cast<uint32_t>(indices.size());

    // Upload vertex buffer
    {
        const vk::DeviceSize vbSize = vertices.size() * sizeof(WaterVertex);
        auto [staging, stagingMem] = allocateBuffer(vbSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        void* mapped = stagingMem.mapMemory(0, vbSize);
        std::memcpy(mapped, vertices.data(), vbSize);
        stagingMem.unmapMemory();

        auto [vb, vbMem] = allocateBuffer(vbSize,
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        vertexBuffer = std::move(vb);
        vertexMemory = std::move(vbMem);

        // Single-shot copy via a temp command pool
        copyBuffer(*staging, *vertexBuffer, vbSize);
    }

    // Upload index buffer
    {
        const vk::DeviceSize ibSize = indexCount * sizeof(uint32_t);
        auto [staging, stagingMem] = allocateBuffer(ibSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        void* mapped = stagingMem.mapMemory(0, ibSize);
        std::memcpy(mapped, indices.data(), ibSize);
        stagingMem.unmapMemory();

        auto [ib, ibMem] = allocateBuffer(ibSize,
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        indexBuffer = std::move(ib);
        indexMemory = std::move(ibMem);

        copyBuffer(*staging, *indexBuffer, ibSize);
    }

    std::cout << "[WaterSystem] Mesh: " << vertices.size() << " vertices, " << indexCount << " indices\n";
    return true;
}

// Single-shot buffer copy using graphics queue
void WaterSystem::copyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size)
{
    const auto& dev = renderer->GetRaiiDevice();

    vk::CommandPoolCreateInfo cpCI{
        .flags            = vk::CommandPoolCreateFlagBits::eTransient,
        .queueFamilyIndex = renderer->GetGraphicsQueueFamilyIndex()
    };
    vk::raii::CommandPool pool(dev, cpCI);

    vk::CommandBufferAllocateInfo cbAI{
        .commandPool        = *pool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    vk::raii::CommandBuffers cbs(dev, cbAI);
    auto& cb = cbs[0];

    cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::BufferCopy region{.size = size};
    cb.copyBuffer(src, dst, region);
    cb.end();

    vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
    vk::raii::Fence fence(dev, vk::FenceCreateInfo{});
    renderer->GetGraphicsQueue().submit(si, *fence);
    (void) dev.waitForFences(*fence, vk::True, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// Cleanup

void WaterSystem::Cleanup()
{
    if (!renderer) return;

    // Wait for device idle before destroying resources
    renderer->WaitIdle();

    frames.clear();

    vertexBuffer = vk::raii::Buffer(nullptr);
    vertexMemory = vk::raii::DeviceMemory(nullptr);
    indexBuffer  = vk::raii::Buffer(nullptr);
    indexMemory  = vk::raii::DeviceMemory(nullptr);
    textureSampler = vk::raii::Sampler(nullptr);

    simulation.reset();
    active   = false;
    renderer = nullptr;
}

// ---------------------------------------------------------------------------
// Update

void WaterSystem::Update(float deltaTime, CameraComponent* camera)
{
    if (!active || !simulation) return;

    simTime += deltaTime;

    // Re-initialize simulation if parameters changed
    static float lastWindSpeed = -1.0f;
    static float lastWindAngle = -999.0f;
    if (std::abs(windSpeed - lastWindSpeed) > 0.01f || std::abs(windAngleDeg - lastWindAngle) > 0.5f) {
        lastWindSpeed = windSpeed;
        lastWindAngle = windAngleDeg;
        float angle = static_cast<float>(windAngleDeg * M_PI / 180.0);
        simulation->SetWindDirection(glm::vec2(std::cos(angle), std::sin(angle)));
        simulation->SetWindSpeed(windSpeed);
        simulation->Prepare();
    }

    // Advance simulation
    simulation->ComputeWaves(simTime);

    // Cache camera state
    if (camera) {
        cachedView   = camera->GetViewMatrix();
        cachedProj   = camera->GetProjectionMatrix();
        cachedProj[1][1] *= -1.0f;  // Vulkan Y-flip (matches main renderer UBO convention)
        cachedCamPos = camera->GetPosition();
    }

    // Refresh descriptor sets when env texture availability or the sky-mode toggle changes
    bool curEnvState = renderer->HasEnvTexture() && renderer->GetUseEnvMapForSky();
    if (curEnvState != lastEnvTextureState) {
        lastEnvTextureState = curEnvState;
        for (uint32_t i = 0; i < static_cast<uint32_t>(frames.size()); ++i)
            updateDescriptorSet(i);
    }
}

// ---------------------------------------------------------------------------
// Image layout transitions (inline in command buffer)

void WaterSystem::transitionWaveTexturesForTransfer(vk::raii::CommandBuffer& cmd, uint32_t frameIndex)
{
    auto& f = frames[frameIndex];

    auto mkBarrier = [](vk::Image img, vk::ImageLayout oldL, vk::ImageLayout newL,
                        vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                        vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
    {
        return vk::ImageMemoryBarrier2{
            .srcStageMask  = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask  = dstStage,
            .dstAccessMask = dstAccess,
            .oldLayout     = oldL,
            .newLayout     = newL,
            .image         = img,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
        };
    };

    vk::PipelineStageFlags2 fragRead = vk::PipelineStageFlagBits2::eFragmentShader;
    vk::PipelineStageFlags2 transfer = vk::PipelineStageFlagBits2::eTransfer;
    vk::AccessFlags2 shaderR  = vk::AccessFlagBits2::eShaderRead;
    vk::AccessFlags2 transferW = vk::AccessFlagBits2::eTransferWrite;

    std::array<vk::ImageMemoryBarrier2, 2> barriers = {{
        mkBarrier(*f.dispImage, f.dispLayout, vk::ImageLayout::eTransferDstOptimal,
                  (f.dispLayout == vk::ImageLayout::eUndefined) ? vk::PipelineStageFlagBits2::eTopOfPipe : fragRead,
                  (f.dispLayout == vk::ImageLayout::eUndefined) ? vk::AccessFlags2{} : shaderR,
                  transfer, transferW),
        mkBarrier(*f.normImage, f.normLayout, vk::ImageLayout::eTransferDstOptimal,
                  (f.normLayout == vk::ImageLayout::eUndefined) ? vk::PipelineStageFlagBits2::eTopOfPipe : fragRead,
                  (f.normLayout == vk::ImageLayout::eUndefined) ? vk::AccessFlags2{} : shaderR,
                  transfer, transferW),
    }};

    vk::DependencyInfo depInfo{
        .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers    = barriers.data()
    };
    cmd.pipelineBarrier2(depInfo);

    f.dispLayout = vk::ImageLayout::eTransferDstOptimal;
    f.normLayout = vk::ImageLayout::eTransferDstOptimal;
}

void WaterSystem::transitionWaveTexturesForSampling(vk::raii::CommandBuffer& cmd, uint32_t frameIndex)
{
    auto& f = frames[frameIndex];

    vk::ImageMemoryBarrier2 dispBarrier{
        .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eVertexShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout     = vk::ImageLayout::eTransferDstOptimal,
        .newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image         = *f.dispImage,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    };
    vk::ImageMemoryBarrier2 normBarrier = dispBarrier;
    normBarrier.image = *f.normImage;

    std::array<vk::ImageMemoryBarrier2, 2> barriers = {dispBarrier, normBarrier};
    vk::DependencyInfo depInfo{
        .imageMemoryBarrierCount = 2,
        .pImageMemoryBarriers    = barriers.data()
    };
    cmd.pipelineBarrier2(depInfo);

    f.dispLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    f.normLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
}

// ---------------------------------------------------------------------------
// Upload wave textures to GPU (via pre-recorded command buffer inline)

void WaterSystem::uploadWaveTextures(uint32_t frameIndex)
{
    if (!simulation) return;

    auto& f   = frames[frameIndex];
    const uint32_t N = static_cast<uint32_t>(tileResolution);
    const vk::DeviceSize texSize = N * N * sizeof(glm::vec4);

    // Copy CPU data to staging buffer
    const auto& displacements = simulation->GetDisplacements();
    const auto& normals       = simulation->GetNormals();

    uint8_t* mapped = static_cast<uint8_t*>(f.stagingMapped);
    std::memcpy(mapped,           displacements.data(), texSize);
    std::memcpy(mapped + texSize, normals.data(),       texSize);

    // The actual vkCmdCopyBufferToImage will be issued in Render() from the
    // main command buffer, so we just need the data to be in staging.
}

// ---------------------------------------------------------------------------
// Render

// ---------------------------------------------------------------------------
// UploadTextures — must be called OUTSIDE a renderpass
// ---------------------------------------------------------------------------
void WaterSystem::UploadTextures(vk::raii::CommandBuffer& cmd, uint32_t frameIndex)
{
    if (!active || frames.empty() || frameIndex >= frames.size()) return;

    auto& f   = frames[frameIndex];
    const uint32_t N = static_cast<uint32_t>(tileResolution);
    const vk::DeviceSize texSize = N * N * sizeof(glm::vec4);

    // Copy CPU simulation output to staging buffer
    uploadWaveTextures(frameIndex);

    // Transition to TransferDst, copy, transition to ShaderReadOnly
    transitionWaveTexturesForTransfer(cmd, frameIndex);

    vk::BufferImageCopy dispRegion{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageOffset       = {0, 0, 0},
        .imageExtent       = {N, N, 1}
    };
    vk::BufferImageCopy normRegion = dispRegion;
    normRegion.bufferOffset = texSize;

    cmd.copyBufferToImage(*f.stagingBuffer, *f.dispImage,
                          vk::ImageLayout::eTransferDstOptimal, dispRegion);
    cmd.copyBufferToImage(*f.stagingBuffer, *f.normImage,
                          vk::ImageLayout::eTransferDstOptimal, normRegion);

    transitionWaveTexturesForSampling(cmd, frameIndex);

    // Update UBOs while we're still outside the renderpass
    {
        WaterVertexUBO vubo{};
        vubo.model        = glm::mat4(1.0f);
        vubo.view         = cachedView;
        vubo.proj         = cachedProj;
        vubo.WSHeightAmp  = heightAmplitude * simulation->GetMaxHeight();
        vubo.WSChoppy     = choppiness;
        vubo.texScale     = textureScale;
        vubo._pad         = 0.0f;
        std::memcpy(f.vertUboMapped, &vubo, sizeof(vubo));
    }
    {
        WaterFragUBO fubo{};
        fubo.camPos          = cachedCamPos;
        fubo.height          = waterDepth;
        fubo.absorpCoef      = glm::vec3(0.420f, 0.063f, 0.019f);
        fubo.scatterCoef     = glm::vec3(0.037f * (-0.00113f * glm::vec3(680,550,440) + 1.62517f)
                                         / (-0.00113f * 514.0f + 1.62517f));
        fubo.backscatterCoef = 0.01829f * fubo.scatterCoef + 0.00006f;
        fubo.terrainColor    = glm::vec3(0.964f, 1.0f, 0.824f);
        fubo.skyIntensity       = skyIntensity;
        fubo.specularIntensity  = specularIntensity;
        fubo.specularHighlights = specularHighlights;
        fubo.useEnvMap          = (renderer->HasEnvTexture() && renderer->GetUseEnvMapForSky()) ? 1u : 0u;
        fubo.sunDirection    = glm::normalize(renderer->GetSunDirection());
        fubo.sunIntensity    = sunIntensity;

        // ---- Preetham sky model (per-channel Y/x/y coefficients) ----
        // Matches WaterSurfaceRendering SkyPreetham::ComputePerezDistribution()
        const float t = renderer->GetSunTurbidity();  // driven by renderer's sun settings
        fubo.turbidity = t;

        // Each vec3 is (Y-luminance, x-chromaticity, y-chromaticity) for that coefficient
        fubo.skyA = glm::vec3( 0.1787f*t - 1.4630f,  -0.0193f*t - 0.2592f,  -0.0167f*t - 0.2608f);
        fubo.skyB = glm::vec3(-0.3554f*t + 0.4275f,  -0.0665f*t + 0.0008f,  -0.0950f*t + 0.0092f);
        fubo.skyC = glm::vec3(-0.0227f*t + 5.3251f,  -0.0004f*t + 0.2125f,  -0.0079f*t + 0.2102f);
        fubo.skyD = glm::vec3( 0.1206f*t - 2.5771f,  -0.0641f*t - 0.8989f,  -0.0441f*t - 1.6537f);
        fubo.skyE = glm::vec3(-0.0670f*t + 0.3703f,  -0.0033f*t + 0.0452f,  -0.0109f*t + 0.0529f);

        // Zenith luminance Yxy — from SkyPreetham::ComputeZenithLuminanceYxy()
        const glm::vec3 UP(0.f, 1.f, 0.f);
        const float thetaSun = std::acos(glm::clamp(glm::dot(fubo.sunDirection, UP), 0.f, 1.f));
        const float t2 = t * t;
        const float th2 = thetaSun * thetaSun, th3 = th2 * thetaSun;
        const float chi = (4.f/9.f - t/120.f) * (static_cast<float>(M_PI) - 2.f*thetaSun);
        const float Yz = std::max((4.0453f*t - 4.9710f) * std::tan(chi) - 0.2155f*t + 2.4192f, 0.f);
        const float xz = ( 0.00165f*th3 - 0.00375f*th2 + 0.00209f*thetaSun         )*t2
                       + (-0.02903f*th3 + 0.06377f*th2 - 0.03202f*thetaSun + 0.00394f)*t
                       + ( 0.11693f*th3 - 0.21196f*th2 + 0.06052f*thetaSun + 0.25886f);
        const float yz = ( 0.00275f*th3 - 0.00610f*th2 + 0.00317f*thetaSun          )*t2
                       + (-0.04214f*th3 + 0.08970f*th2 - 0.04153f*thetaSun + 0.00516f)*t
                       + ( 0.15346f*th3 - 0.26756f*th2 + 0.06670f*thetaSun + 0.26688f);
        fubo.ZenithLum = glm::vec3(Yz, xz, yz);

        // ZeroThetaSun — Perez function evaluated at (theta=0, gamma=thetaSun)
        // = (1 + A*exp(B)) * (1 + C*exp(D*thetaSun) + E*cos²(thetaSun))
        const float cosGamma = std::cos(thetaSun);
        fubo.ZeroThetaSun = (1.f + fubo.skyA * glm::exp(fubo.skyB))
                          * (1.f + fubo.skyC * glm::exp(fubo.skyD * thetaSun)
                                 + fubo.skyE * (cosGamma * cosGamma));
        std::memcpy(f.fragUboMapped, &fubo, sizeof(fubo));
    }
}

// ---------------------------------------------------------------------------
// Draw — must be called INSIDE a beginRendering block
// ---------------------------------------------------------------------------
void WaterSystem::Draw(vk::raii::CommandBuffer& cmd, uint32_t frameIndex)
{
    if (!active || frames.empty() || frameIndex >= frames.size()) return;

    vk::Pipeline wp = renderer->GetWaterPipeline();
    if (!wp) return;

    auto& f = frames[frameIndex];

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, wp);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           renderer->GetWaterPipelineLayout(),
                           0, {*f.descriptorSet}, {});

    vk::Buffer vb = *vertexBuffer;
    vk::DeviceSize offset = 0;
    cmd.bindVertexBuffers(0, vb, offset);
    cmd.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
    cmd.drawIndexed(indexCount, 1, 0, 0, 0);
}
