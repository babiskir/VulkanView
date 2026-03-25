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
#include "mesh_component.h"
#include "renderer.h"
#include <array>
#include <fstream>
#include <iostream>

// This file contains pipeline-related methods from the Renderer class

// Create a descriptor set layout
bool Renderer::createDescriptorSetLayout() {
  try {
    // Create binding for a uniform buffer
    vk::DescriptorSetLayoutBinding uboLayoutBinding{
      .binding = 0,
      .descriptorType = vk::DescriptorType::eUniformBuffer,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
      .pImmutableSamplers = nullptr
    };

    // Create binding for texture sampler
    vk::DescriptorSetLayoutBinding samplerLayoutBinding{
      .binding = 1,
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eFragment,
      .pImmutableSamplers = nullptr
    };

    // Create a descriptor set layout
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

    // Descriptor indexing: set per-binding flags for UPDATE_AFTER_BIND if enabled
    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    std::array<vk::DescriptorBindingFlags, 2> bindingFlags{};
    if (descriptorIndexingEnabled) {
      bindingFlags[0] = vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
      bindingFlags[1] = vk::DescriptorBindingFlagBits::eUpdateAfterBind | vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
      bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
      bindingFlagsInfo.pBindingFlags = bindingFlags.data();
    }

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (descriptorIndexingEnabled) {
      layoutInfo.flags |= vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
      layoutInfo.pNext = &bindingFlagsInfo;
    }

    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create descriptor set layout: " << e.what() << std::endl;
    return false;
  }
}

// Create PBR descriptor set layout
bool Renderer::createPBRDescriptorSetLayout() {
  try {
    // Create descriptor set layout bindings for PBR shader
    std::array bindings = {
      // Binding 0: Uniform buffer (UBO)
      vk::DescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 1: Base color map and sampler
      vk::DescriptorSetLayoutBinding{
        .binding = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 2: Metallic roughness map and sampler
      vk::DescriptorSetLayoutBinding{
        .binding = 2,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 3: Normal map and sampler
      vk::DescriptorSetLayoutBinding{
        .binding = 3,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 4: Occlusion map and sampler
      vk::DescriptorSetLayoutBinding{
        .binding = 4,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 5: Emissive map and sampler
      vk::DescriptorSetLayoutBinding{
        .binding = 5,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 6: Light storage buffer (shadows removed)
      vk::DescriptorSetLayoutBinding{
        .binding = 6,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 7: Forward+ tile headers SSBO
      vk::DescriptorSetLayoutBinding{
        .binding = 7,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 8: Forward+ tile light indices SSBO
      vk::DescriptorSetLayoutBinding{
        .binding = 8,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 9: Fragment debug output buffer (optional)
      vk::DescriptorSetLayoutBinding{
        .binding = 9,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 10: Reflection texture (planar reflections)
      vk::DescriptorSetLayoutBinding{
        .binding = 10,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 11: TLAS (ray-query shadows in raster fragment shader)
      vk::DescriptorSetLayoutBinding{
        .binding = 11,
        .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 12: Ray-query geometry info buffer (per-instance addresses + material indices)
      vk::DescriptorSetLayoutBinding{
        .binding = 12,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 13: Ray-query material buffer (PBR material properties)
      vk::DescriptorSetLayoutBinding{
        .binding = 13,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 14: CSM shadow maps (3 cascades, comparison sampler)
      vk::DescriptorSetLayoutBinding{
        .binding = 14,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = CSM_CASCADE_COUNT,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      },
      // Binding 15: CSM UBO (light space matrices + cascade splits + shadow params)
      vk::DescriptorSetLayoutBinding{
        .binding = 15,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr
      }
    };

    // Create a descriptor set layout
    // Descriptor indexing: set per-binding flags for UPDATE_AFTER_BIND on UBO (0) and sampled images (1..5)
    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    std::array<vk::DescriptorBindingFlags, 16> bindingFlags{};
    if (descriptorIndexingEnabled) {
      bindingFlags[0] = vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
      bindingFlags[1] = vk::DescriptorBindingFlagBits::eUpdateAfterBind | vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
      bindingFlags[10] = vk::DescriptorBindingFlagBits::eUpdateAfterBind | vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
      // Binding 11 (TLAS) is partially bound: on non-RT hardware we skip writing it,
      // which is spec-legal only when ePartiallyBound is set.
      bindingFlags[11] = vk::DescriptorBindingFlagBits::ePartiallyBound;
      // Bindings 14 and 15 (CSM) do NOT use eUpdateAfterBind for Maxwell compatibility
      bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
      bindingFlagsInfo.pBindingFlags = bindingFlags.data();
    }

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (descriptorIndexingEnabled) {
      layoutInfo.flags |= vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
      layoutInfo.pNext = &bindingFlagsInfo;
    }

    pbrDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    // Binding 7: transparent passes input
    // Layout for Set 1: Just the scene color texture
    vk::DescriptorSetLayoutBinding sceneColorBinding{
      .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment
    };
    vk::DescriptorSetLayoutCreateInfo transparentLayoutInfo{.bindingCount = 1, .pBindings = &sceneColorBinding};
    if (descriptorIndexingEnabled) {
      // Make this sampler binding update-after-bind safe as well (optional)
      vk::DescriptorSetLayoutBindingFlagsCreateInfo transBindingFlagsInfo{};
      vk::DescriptorBindingFlags transFlags = vk::DescriptorBindingFlagBits::eUpdateAfterBind | vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
      transBindingFlagsInfo.bindingCount = 1;
      transBindingFlagsInfo.pBindingFlags = &transFlags;
      transparentLayoutInfo.flags |= vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
      transparentLayoutInfo.pNext = &transBindingFlagsInfo;

      // Create the layout while the pNext chain is still valid (avoid dangling pointer)
      transparentDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, transparentLayoutInfo);
    } else {
      // Create without extra binding flags
      transparentDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, transparentLayoutInfo);
    }

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create PBR descriptor set layout: " << e.what() << std::endl;
    return false;
  }
}

// Create a graphics pipeline
bool Renderer::createGraphicsPipeline() {
  try {
    // Read shader code
    auto shaderCode = readFile("shaders/texturedMesh.spv");

    // Create shader modules
    vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

    // Create shader stage info
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = *shaderModule,
      .pName = "main"
    };

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = *shaderModule,
      .pName = "main"
    };

    // Fragment entry point specialized for architectural glass
    vk::PipelineShaderStageCreateInfo fragGlassStageInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = *shaderModule,
      .pName = "main"
    };

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // Create vertex input info with instancing support
    auto vertexBindingDescription = Vertex::getBindingDescription();
    auto instanceBindingDescription = InstanceData::getBindingDescription();
    std::array<vk::VertexInputBindingDescription, 2> bindingDescriptions = {
      vertexBindingDescription,
      instanceBindingDescription
    };

    auto vertexAttributeDescriptions = Vertex::getAttributeDescriptions();
    auto instanceAttributeDescriptions = InstanceData::getAttributeDescriptions();

    // Combine all attribute descriptions (no duplicates)
    std::vector<vk::VertexInputAttributeDescription> allAttributeDescriptions;
    allAttributeDescriptions.insert(allAttributeDescriptions.end(), vertexAttributeDescriptions.begin(), vertexAttributeDescriptions.end());
    allAttributeDescriptions.insert(allAttributeDescriptions.end(), instanceAttributeDescriptions.begin(), instanceAttributeDescriptions.end());

    // Note: materialIndex attribute (Location 11) is not used by current shaders

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size()),
      .pVertexBindingDescriptions = bindingDescriptions.data(),
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(allAttributeDescriptions.size()),
      .pVertexAttributeDescriptions = allAttributeDescriptions.data()
    };

    // Create input assembly info
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
      .topology = vk::PrimitiveTopology::eTriangleList,
      .primitiveRestartEnable = VK_FALSE
    };

    // Create viewport state info
    vk::PipelineViewportStateCreateInfo viewportState{
      .viewportCount = 1,
      .scissorCount = 1
    };

    // Create rasterization state info
    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eNone,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = VK_FALSE,
      .lineWidth = 1.0f
    };

    // Create multisample state info
    vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = VK_FALSE
    };

    // Create depth stencil state info
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      // Use LessOrEqual so that the main shading pass works after a depth pre-pass
      .depthCompareOp = vk::CompareOp::eLessOrEqual,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE
    };

    // Create a color blend attachment state
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable = VK_FALSE,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    // Create color blend state info
    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .logicOpEnable = VK_FALSE,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment
    };

    // Create dynamic state info
    std::vector dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()
    };

    // Create pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
      .setLayoutCount = 1,
      .pSetLayouts = &*descriptorSetLayout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr
    };

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    // Create pipeline rendering info
    vk::Format depthFormat = findDepthFormat();
    std::cout << "Creating main graphics pipeline with depth format: " << static_cast<int>(depthFormat) << std::endl;

    // Initialize member variable for proper lifetime management
    mainPipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &swapChainImageFormat,
      .depthAttachmentFormat = depthFormat,
      .stencilAttachmentFormat = vk::Format::eUndefined
    };

    // Create the graphics pipeline
    vk::PipelineRasterizationStateCreateInfo rasterizerBack = rasterizer;
    // Disable back-face culling for opaque PBR to avoid disappearing geometry when
    // instance/model transforms flip winding (ensures PASS 1 actually shades pixels)
    rasterizerBack.cullMode = vk::CullModeFlagBits::eNone;

    vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext = &mainPipelineRenderingCreateInfo,
      .flags = vk::PipelineCreateFlags{},
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizerBack,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = *pipelineLayout,
      .renderPass = nullptr,
      .subpass = 0,
      .basePipelineHandle = nullptr,
      .basePipelineIndex = -1
    };

    graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create graphics pipeline: " << e.what() << std::endl;
    return false;
  }
}

// Create PBR pipeline
bool Renderer::createPBRPipeline() {
  try {
    // Create PBR descriptor set layout
    if (!createPBRDescriptorSetLayout()) {
      return false;
    }

    // Each PBR entry point is compiled to its own SPV to avoid two Fragment
    // entry points sharing the name "main" in one module (invalid SPIR-V).
    vk::raii::ShaderModule vertShaderModule  = createShaderModule(readFile("shaders/pbr_vsmain.spv"));
    vk::raii::ShaderModule fragShaderModule  = createShaderModule(readFile("shaders/pbr_psmain.spv"));
    vk::raii::ShaderModule glassShaderModule = createShaderModule(readFile("shaders/pbr_glasspsmain.spv"));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = *vertShaderModule,
      .pName = "main"
    };

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = *fragShaderModule,
      .pName = "main"
    };

    // Fragment entry point specialized for architectural glass
    vk::PipelineShaderStageCreateInfo fragGlassStageInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = *glassShaderModule,
      .pName = "main"
    };

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // Define vertex and instance binding descriptions
    auto vertexBindingDescription = Vertex::getBindingDescription();
    auto instanceBindingDescription = InstanceData::getBindingDescription();
    std::array<vk::VertexInputBindingDescription, 2> bindingDescriptions = {
      vertexBindingDescription,
      instanceBindingDescription
    };

    // Define vertex and instance attribute descriptions
    auto vertexAttributeDescriptions = Vertex::getAttributeDescriptions();
    auto instanceModelMatrixAttributes = InstanceData::getModelMatrixAttributeDescriptions();
    auto instanceNormalMatrixAttributes = InstanceData::getNormalMatrixAttributeDescriptions();

    // Combine all attribute descriptions
    std::vector<vk::VertexInputAttributeDescription> allAttributeDescriptions;
    allAttributeDescriptions.insert(allAttributeDescriptions.end(), vertexAttributeDescriptions.begin(), vertexAttributeDescriptions.end());
    allAttributeDescriptions.insert(allAttributeDescriptions.end(), instanceModelMatrixAttributes.begin(), instanceModelMatrixAttributes.end());
    allAttributeDescriptions.insert(allAttributeDescriptions.end(), instanceNormalMatrixAttributes.begin(), instanceNormalMatrixAttributes.end());

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size()),
      .pVertexBindingDescriptions = bindingDescriptions.data(),
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(allAttributeDescriptions.size()),
      .pVertexAttributeDescriptions = allAttributeDescriptions.data()
    };

    // Create input assembly info
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
      .topology = vk::PrimitiveTopology::eTriangleList,
      .primitiveRestartEnable = VK_FALSE
    };

    // Create viewport state info
    vk::PipelineViewportStateCreateInfo viewportState{
      .viewportCount = 1,
      .scissorCount = 1
    };

    // Create rasterization state info
    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eNone,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = VK_FALSE,
      .lineWidth = 1.0f
    };

    // Create multisample state info
    vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = VK_FALSE
    };

    // Create depth stencil state info
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE
    };

    // Create a color blend attachment state
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable = VK_FALSE,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    // Create color blend state info
    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .logicOpEnable = VK_FALSE,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment
    };

    // Create dynamic state info
    std::vector dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()
    };

    // Create push constant range for material properties
    vk::PushConstantRange pushConstantRange{
      .stageFlags = vk::ShaderStageFlagBits::eFragment,
      .offset = 0,
      .size = sizeof(MaterialProperties)
    };

    std::array<vk::DescriptorSetLayout, 2> transparentSetLayouts = {*pbrDescriptorSetLayout, *transparentDescriptorSetLayout};
    // Create a pipeline layout for opaque PBR with only the PBR descriptor set (set 0)
    std::array<vk::DescriptorSetLayout, 1> pbrOnlySetLayouts = {*pbrDescriptorSetLayout};
    // Create BOTH pipeline layouts with two descriptor sets (PBR set 0 + scene color set 1)
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
      .setLayoutCount = static_cast<uint32_t>(transparentSetLayouts.size()),
      .pSetLayouts = transparentSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange
    };

    pbrPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    // Transparent PBR layout uses the same two-set layout
    vk::PipelineLayoutCreateInfo transparentPipelineLayoutInfo{.setLayoutCount = static_cast<uint32_t>(transparentSetLayouts.size()), .pSetLayouts = transparentSetLayouts.data(), .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange};
    pbrTransparentPipelineLayout = vk::raii::PipelineLayout(device, transparentPipelineLayoutInfo);

    // Create pipeline rendering info
    vk::Format depthFormat = findDepthFormat();

    // Initialize member variable for proper lifetime management
    pbrPipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &swapChainImageFormat,
      .depthAttachmentFormat = depthFormat,
      .stencilAttachmentFormat = vk::Format::eUndefined
    };

    // 1) Opaque PBR pipeline (no blending, depth writes enabled)
    vk::PipelineColorBlendAttachmentState opaqueBlendAttachment = colorBlendAttachment;
    opaqueBlendAttachment.blendEnable = VK_FALSE;
    vk::PipelineColorBlendStateCreateInfo colorBlendingOpaque{
      .logicOpEnable = VK_FALSE,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &opaqueBlendAttachment
    };
    vk::PipelineDepthStencilStateCreateInfo depthStencilOpaque = depthStencil;
    depthStencilOpaque.depthWriteEnable = VK_TRUE;

    vk::PipelineRasterizationStateCreateInfo rasterizerBack = rasterizer;
    rasterizerBack.cullMode = vk::CullModeFlagBits::eBack;

    // For architectural glass we often want to see both the inner and outer
    // walls of thin shells (e.g., bar glasses viewed from above). Use
    // no culling for the glass pipeline to render both sides, while
    // keeping back-face culling for the generic PBR pipelines.
    vk::PipelineRasterizationStateCreateInfo rasterizerGlass = rasterizer;
    rasterizerGlass.cullMode = vk::CullModeFlagBits::eNone;

    vk::GraphicsPipelineCreateInfo opaquePipelineInfo{

      .pNext = &pbrPipelineRenderingCreateInfo,
      .flags = vk::PipelineCreateFlags{},
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizerBack,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencilOpaque,
      .pColorBlendState = &colorBlendingOpaque,
      .pDynamicState = &dynamicState,
      .layout = *pbrPipelineLayout,
      .renderPass = nullptr,
      .subpass = 0,
      .basePipelineHandle = nullptr,
      .basePipelineIndex = -1
    };
    pbrGraphicsPipeline = vk::raii::Pipeline(device, nullptr, opaquePipelineInfo);

    // 1b) Opaque PBR pipeline variant for color pass after a depth pre-pass.
    // Depth writes disabled (read-only) and compare against pre-pass depth.
    vk::PipelineDepthStencilStateCreateInfo depthStencilAfterPrepass = depthStencil;
    depthStencilAfterPrepass.depthTestEnable = VK_TRUE;
    depthStencilAfterPrepass.depthWriteEnable = VK_FALSE;
    depthStencilAfterPrepass.depthCompareOp = vk::CompareOp::eEqual;

    vk::GraphicsPipelineCreateInfo opaqueAfterPrepassInfo{

      .pNext = &pbrPipelineRenderingCreateInfo,
      .flags = vk::PipelineCreateFlags{},
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizerBack,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencilAfterPrepass,
      .pColorBlendState = &colorBlendingOpaque,
      .pDynamicState = &dynamicState,
      .layout = *pbrPipelineLayout,
      .renderPass = nullptr,
      .subpass = 0,
      .basePipelineHandle = nullptr,
      .basePipelineIndex = -1
    };
    pbrPrepassGraphicsPipeline = vk::raii::Pipeline(device, nullptr, opaqueAfterPrepassInfo);

    // 1c) Reflection PBR pipeline for mirrored off-screen pass (cull none to avoid winding issues)
    vk::PipelineRasterizationStateCreateInfo rasterizerReflection = rasterizer;
    rasterizerReflection.cullMode = vk::CullModeFlagBits::eNone;
    vk::GraphicsPipelineCreateInfo reflectionPipelineInfo{

      .pNext = &pbrPipelineRenderingCreateInfo,
      .flags = vk::PipelineCreateFlags{},
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizerReflection,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencilOpaque,
      .pColorBlendState = &colorBlendingOpaque,
      .pDynamicState = &dynamicState,
      .layout = *pbrPipelineLayout,
      .renderPass = nullptr,
      .subpass = 0,
      .basePipelineHandle = nullptr,
      .basePipelineIndex = -1
    };
    pbrReflectionGraphicsPipeline = vk::raii::Pipeline(device, nullptr, reflectionPipelineInfo);

    // 2) Blended PBR pipeline (straight alpha blending, depth writes disabled for translucency)
    vk::PipelineColorBlendAttachmentState blendedAttachment = colorBlendAttachment;
    blendedAttachment.blendEnable = VK_TRUE;
    // Straight alpha blending: out.rgb = src.rgb*src.a + dst.rgb*(1-src.a)
    blendedAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    blendedAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    // Alpha channel keeps destination scaled by inverse src alpha
    blendedAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    blendedAttachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    vk::PipelineColorBlendStateCreateInfo colorBlendingBlended{.attachmentCount = 1, .pAttachments = &blendedAttachment};
    vk::PipelineDepthStencilStateCreateInfo depthStencilBlended = depthStencil;
    depthStencilBlended.depthWriteEnable = VK_FALSE;
    depthStencilBlended.depthCompareOp = vk::CompareOp::eLessOrEqual;

    vk::GraphicsPipelineCreateInfo blendedPipelineInfo{

      .pNext = &pbrPipelineRenderingCreateInfo,
      .flags = vk::PipelineCreateFlags{},
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      // Use back-face culling for the blended (glass) pipeline to avoid
      // rendering both front and back faces of thin glass geometry, which
      // can cause flickering as the camera rotates due to overlapping
      // transparent surfaces passing the depth test.
      .pRasterizationState = &rasterizerBack,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencilBlended,
      .pColorBlendState = &colorBlendingBlended,
      .pDynamicState = &dynamicState,
      .layout = *pbrTransparentPipelineLayout,
      .renderPass = nullptr,
      .subpass = 0,
      .basePipelineHandle = nullptr,
      .basePipelineIndex = -1
    };
    pbrBlendGraphicsPipeline = vk::raii::Pipeline(device, nullptr, blendedPipelineInfo);

    // 3) Glass pipeline (architectural glass) - uses the same vertex input and
    // descriptor layouts, but a dedicated fragment shader entry point
    // (GlassPSMain) for more stable glass shading.
    vk::PipelineShaderStageCreateInfo glassStages[] = {vertShaderStageInfo, fragGlassStageInfo};

    vk::GraphicsPipelineCreateInfo glassPipelineInfo{

      .pNext = &pbrPipelineRenderingCreateInfo,
      .flags = vk::PipelineCreateFlags{},
      .stageCount = 2,
      .pStages = glassStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizerGlass,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencilBlended,
      .pColorBlendState = &colorBlendingBlended,
      .pDynamicState = &dynamicState,
      .layout = *pbrTransparentPipelineLayout,
      .renderPass = nullptr,
      .subpass = 0,
      .basePipelineHandle = nullptr,
      .basePipelineIndex = -1
    };
    glassGraphicsPipeline = vk::raii::Pipeline(device, nullptr, glassPipelineInfo);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create PBR pipeline: " << e.what() << std::endl;
    return false;
  }
}

// Create fullscreen composite pipeline (samples off-screen color and writes to swapchain)
bool Renderer::createCompositePipeline() {
  try {
    // Reuse the transparent descriptor set layout (binding 0 = combined image sampler)
    if (*transparentDescriptorSetLayout == nullptr) {
      // Ensure PBR pipeline path created it
      if (!createPBRPipeline()) {
        return false;
      }
    }

    // Read composite shader code
    auto shaderCode = readFile("shaders/composite.spv");
    vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

    // Shader stages
    vk::PipelineShaderStageCreateInfo vert{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = *shaderModule,
      .pName = "main"
    };
    vk::PipelineShaderStageCreateInfo frag{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = *shaderModule,
      .pName = "main"
    };
    vk::PipelineShaderStageCreateInfo stages[] = {vert, frag};

    // No vertex inputs (fullscreen triangle via SV_VertexID)
    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{.polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone, .frontFace = vk::FrontFace::eCounterClockwise, .lineWidth = 1.0f};
    vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    // No depth
    vk::PipelineDepthStencilStateCreateInfo depthStencil{.depthTestEnable = VK_FALSE, .depthWriteEnable = VK_FALSE};
    // No blending (we clear swapchain before this and blend transparents later)
    vk::PipelineColorBlendAttachmentState attachment{
      .blendEnable = VK_FALSE,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{.attachmentCount = 1, .pAttachments = &attachment};
    std::array dynStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynStates.size()), .pDynamicStates = dynStates.data()};

    // Pipeline layout: single set (combined image sampler) + push constants for exposure/gamma/srgb flag
    vk::DescriptorSetLayout setLayouts[] = {*transparentDescriptorSetLayout};
    vk::PushConstantRange pushRange{.stageFlags = vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = 16}; // matches struct Push in composite.slang
    vk::PipelineLayoutCreateInfo plInfo{.setLayoutCount = 1, .pSetLayouts = setLayouts, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange};
    compositePipelineLayout = vk::raii::PipelineLayout(device, plInfo);

    // Dynamic rendering info
    compositePipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{

      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &swapChainImageFormat,
      .depthAttachmentFormat = vk::Format::eUndefined,
      .stencilAttachmentFormat = vk::Format::eUndefined
    };

    vk::GraphicsPipelineCreateInfo pipeInfo{

      .pNext = &compositePipelineRenderingCreateInfo,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertexInput,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = *compositePipelineLayout,
      .renderPass = nullptr,
      .subpass = 0
    };

    compositePipeline = vk::raii::Pipeline(device, nullptr, pipeInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create composite pipeline: " << e.what() << std::endl;
    return false;
  }
}

// Create Depth Pre-pass pipeline (depth-only)
bool Renderer::createDepthPrepassPipeline() {
  try {
    // Use the same descriptor set layout and pipeline layout as PBR for UBOs and instancing
    if (*pbrDescriptorSetLayout == nullptr || *pbrPipelineLayout == nullptr) {
      if (!createPBRPipeline()) {
        return false;
      }
    }

    // Read PBR vertex shader (Slang entry point: vsmain)
    auto shaderCode = readFile("shaders/pbr_vsmain.spv");
    vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

    // Stages: Vertex only (depth pre-pass writes depth, no color output)
    vk::PipelineShaderStageCreateInfo vertStage{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = *shaderModule,
      .pName = "main"
    };

    // Vertex/instance bindings & attributes same as PBR
    auto vertexBindingDescription = Vertex::getBindingDescription();
    auto instanceBindingDescription = InstanceData::getBindingDescription();
    std::array<vk::VertexInputBindingDescription, 2> bindingDescriptions = {
      vertexBindingDescription,
      instanceBindingDescription
    };

    auto vertexAttributeDescriptions = Vertex::getAttributeDescriptions();
    auto instanceModelMatrixAttributes = InstanceData::getModelMatrixAttributeDescriptions();
    auto instanceNormalMatrixAttributes = InstanceData::getNormalMatrixAttributeDescriptions();
    std::vector<vk::VertexInputAttributeDescription> allAttributes;
    allAttributes.insert(allAttributes.end(), vertexAttributeDescriptions.begin(), vertexAttributeDescriptions.end());
    allAttributes.insert(allAttributes.end(), instanceModelMatrixAttributes.begin(), instanceModelMatrixAttributes.end());
    allAttributes.insert(allAttributes.end(), instanceNormalMatrixAttributes.begin(), instanceNormalMatrixAttributes.end());

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size()),
      .pVertexBindingDescriptions = bindingDescriptions.data(),
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(allAttributes.size()),
      .pVertexAttributeDescriptions = allAttributes.data()
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
      .topology = vk::PrimitiveTopology::eTriangleList,
      .primitiveRestartEnable = VK_FALSE
    };

    // Dummy viewport/scissor (dynamic)
    vk::PipelineViewportStateCreateInfo viewportState{
      .viewportCount = 1,
      .scissorCount = 1
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = VK_FALSE,
      .lineWidth = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = vk::CompareOp::eLessOrEqual,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE
    };

    // No color attachments
    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .logicOpEnable = VK_FALSE,
      .attachmentCount = 0,
      .pAttachments = nullptr
    };

    std::array dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()
    };

    vk::Format depthFormat = findDepthFormat();
    vk::PipelineRenderingCreateInfo renderingInfo{
      .colorAttachmentCount = 0,
      .pColorAttachmentFormats = nullptr,
      .depthAttachmentFormat = depthFormat
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext = &renderingInfo,
      .stageCount = 1,
      .pStages = &vertStage,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = *pbrPipelineLayout
    };

    depthPrepassPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create depth pre-pass pipeline: " << e.what() << std::endl;
    return false;
  }
}

// Create a lighting pipeline
bool Renderer::createLightingPipeline() {
  try {
    // Read shader code
    auto shaderCode = readFile("shaders/lighting.spv");

    // Create shader modules
    vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

    // Create shader stage info
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = *shaderModule,
      .pName = "main"
    };

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = *shaderModule,
      .pName = "main"
    };

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // Create vertex input info
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDescription,
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
      .pVertexAttributeDescriptions = attributeDescriptions.data()
    };

    // Create input assembly info
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
      .topology = vk::PrimitiveTopology::eTriangleList,
      .primitiveRestartEnable = VK_FALSE
    };

    // Create viewport state info
    vk::PipelineViewportStateCreateInfo viewportState{
      .viewportCount = 1,
      .scissorCount = 1
    };

    // Create rasterization state info
    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eNone,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = VK_FALSE,
      .lineWidth = 1.0f
    };

    // Create multisample state info
    vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = VK_FALSE
    };

    // Create depth stencil state info
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE
    };

    // Create a color blend attachment state
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
      .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = vk::BlendFactor::eOne,
      .dstAlphaBlendFactor = vk::BlendFactor::eZero,
      .alphaBlendOp = vk::BlendOp::eAdd,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    // Create color blend state info
    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .logicOpEnable = VK_FALSE,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment
    };

    // Create dynamic state info
    std::vector dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()
    };

    // Create push constant range for material properties
    vk::PushConstantRange pushConstantRange{
      .stageFlags = vk::ShaderStageFlagBits::eFragment,
      .offset = 0,
      .size = sizeof(MaterialProperties)
    };

    // Create pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
      .setLayoutCount = 1,
      .pSetLayouts = &*descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange
    };

    lightingPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    // Create pipeline rendering info
    vk::Format depthFormat = findDepthFormat();

    // Initialize member variable for proper lifetime management
    lightingPipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo{

      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &swapChainImageFormat,
      .depthAttachmentFormat = depthFormat,
      .stencilAttachmentFormat = vk::Format::eUndefined
    };

    // Create a graphics pipeline
    vk::PipelineRasterizationStateCreateInfo rasterizerBack = rasterizer;
    rasterizerBack.cullMode = vk::CullModeFlagBits::eBack;

    vk::GraphicsPipelineCreateInfo pipelineInfo{

      .pNext = &lightingPipelineRenderingCreateInfo,
      .flags = vk::PipelineCreateFlags{},
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizerBack,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = *lightingPipelineLayout,
      .renderPass = nullptr,
      .subpass = 0,
      .basePipelineHandle = nullptr,
      .basePipelineIndex = -1
    };

    lightingPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create lighting pipeline: " << e.what() << std::endl;
    return false;
  }
}

// Push material properties to the pipeline
void Renderer::pushMaterialProperties(vk::CommandBuffer commandBuffer, const MaterialProperties& material) const {
  commandBuffer.pushConstants(*pbrPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(MaterialProperties), &material);
}

bool Renderer::createRayQueryDescriptorSetLayout() {
  // Production layout: 7 bindings (0..6), no debug buffer at 7
  std::array<vk::DescriptorSetLayoutBinding, 7> bindings{};

  // Binding 0: UBO (UniformBufferObject)
  bindings[0].binding = 0;
  bindings[0].descriptorType = vk::DescriptorType::eUniformBuffer;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

  // Binding 1: TLAS (Top-Level Acceleration Structure)
  bindings[1].binding = 1;
  bindings[1].descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

  // Binding 2: Output image (storage image)
  bindings[2].binding = 2;
  bindings[2].descriptorType = vk::DescriptorType::eStorageImage;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = vk::ShaderStageFlagBits::eCompute;

  // Binding 3: Light buffer (storage buffer)
  bindings[3].binding = 3;
  bindings[3].descriptorType = vk::DescriptorType::eStorageBuffer;
  bindings[3].descriptorCount = 1;
  bindings[3].stageFlags = vk::ShaderStageFlagBits::eCompute;

  // Binding 4: Geometry info buffer (maps BLAS geometry index to vertex/index buffer addresses)
  bindings[4].binding = 4;
  bindings[4].descriptorType = vk::DescriptorType::eStorageBuffer;
  bindings[4].descriptorCount = 1;
  bindings[4].stageFlags = vk::ShaderStageFlagBits::eCompute;

  // Binding 5: Material buffer (array of material properties)
  bindings[5].binding = 5;
  bindings[5].descriptorType = vk::DescriptorType::eStorageBuffer;
  bindings[5].descriptorCount = 1;
  bindings[5].stageFlags = vk::ShaderStageFlagBits::eCompute;

  // Binding 6: BaseColor textures array (combined image samplers)
  bindings[6].binding = 6;
  bindings[6].descriptorType = vk::DescriptorType::eCombinedImageSampler;
  bindings[6].descriptorCount = RQ_MAX_TEX; // large static array
  bindings[6].stageFlags = vk::ShaderStageFlagBits::eCompute;

  // Descriptor indexing / update-after-bind support:
  // The ray query shader indexes a large `eCombinedImageSampler` array with a per-pixel varying index.
  // On some drivers this requires descriptor indexing features + layout binding flags to avoid the
  // array collapsing to slot 0 (resulting in "no textures" even when `texIndex>0`).
  std::array<vk::DescriptorBindingFlags, 7> bindingFlags{};
  if (descriptorIndexingEnabled) {
    // Binding 6 is the large sampled texture array.
    bindingFlags[6] = vk::DescriptorBindingFlagBits::eUpdateAfterBind |
        vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending |
        vk::DescriptorBindingFlagBits::ePartiallyBound;
  }

  vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
  if (descriptorIndexingEnabled) {
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();
  }

  vk::DescriptorSetLayoutCreateInfo layoutInfo{};
  if (descriptorIndexingEnabled) {
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
  }
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  try {
    rayQueryDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create ray query descriptor set layout: " << e.what() << std::endl;
    return false;
  }
}

bool Renderer::createRayQueryPipeline() {
  // Check if ray query is supported on this device
  if (!rayQueryEnabled || !accelerationStructureEnabled) {
    std::cout << "Ray query rendering not available on this device (missing VK_KHR_ray_query or VK_KHR_acceleration_structure support)\n";
    return true; // Not an error - just skip ray query pipeline creation
  }

  // Load compiled shader module
  auto shaderCode = readFile("shaders/ray_query.spv");
  if (shaderCode.empty()) {
    std::cerr << "Failed to load ray query shader\n";
    return false;
  }

  vk::ShaderModuleCreateInfo createInfo{};
  createInfo.codeSize = shaderCode.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());

  vk::raii::ShaderModule shaderModule(device, createInfo);

  vk::PipelineShaderStageCreateInfo shaderStage{};
  shaderStage.stage = vk::ShaderStageFlagBits::eCompute;
  shaderStage.module = *shaderModule;
  shaderStage.pName = "main";

  // Create pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &(*rayQueryDescriptorSetLayout);

  rayQueryPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

  // Create compute pipeline
  vk::ComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.stage = shaderStage;
  pipelineInfo.layout = *rayQueryPipelineLayout;

  try {
    rayQueryPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create ray query pipeline: " << e.what() << std::endl;
    return false;
  }
}

bool Renderer::createRayQueryResources() {
  try {
    // Create output image using memory pool (storage image for compute shader)
    // Use an HDR-capable format for Ray Query so PBR lighting can accumulate in linear space
    // before composite applies exposure/gamma.
    // Fall back to R8G8B8A8_UNORM if the device does not support storage-image usage.
    vk::Format rqFormat = vk::Format::eR16G16B16A16Sfloat; {
      auto props = physicalDevice.getFormatProperties(rqFormat);
      if (!(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage)) {
        rqFormat = vk::Format::eR8G8B8A8Unorm;
      }
    }
    auto [image, allocation] = createVmaImage(
      swapChainExtent.width,
      swapChainExtent.height,
      rqFormat,
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
      VMA_MEMORY_USAGE_GPU_ONLY,
      1,
      // mipLevels
      vk::SharingMode::eExclusive,
      {} // queueFamilies
    );

    rayQueryOutputImage = std::move(image);
    rayQueryOutputImageAllocation = allocation;

    // Create image view
    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = *rayQueryOutputImage;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = rqFormat;
    viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    rayQueryOutputImageView = vk::raii::ImageView(device, viewInfo);

    // Transition output image to GENERAL layout for compute shader writes
    transitionImageLayout(*rayQueryOutputImage,
                          rqFormat,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eGeneral,
                          1);

    // Allocate descriptor sets (one per frame in flight)
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *rayQueryDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();

    // Allocate into a temporary owning container, then move the individual RAII sets into our vector.
    // (Avoid assigning `vk::raii::DescriptorSets` directly into `std::vector<vk::raii::DescriptorSet>`.)
    {
      auto sets = vk::raii::DescriptorSets(device, allocInfo);
      rayQueryDescriptorSets.clear();
      rayQueryDescriptorSets.reserve(sets.size());
      for (auto& s : sets) {
        rayQueryDescriptorSets.emplace_back(std::move(s));
      }
    }

    // Create descriptor sets for composite pass to sample the rayQueryOutputImage
    // Reuse the transparentDescriptorSetLayout (binding 0 = combined image sampler)
    if (*transparentDescriptorSetLayout == nullptr) {
      // Ensure it exists (created by PBR path);
      createPBRPipeline();
    }
    if (*transparentDescriptorSetLayout != nullptr) {
      // Ensure we have a valid sampler for sampling the ray-query output image
      if (*rqCompositeSampler == nullptr) {
        vk::SamplerCreateInfo sci{
          .magFilter = vk::Filter::eLinear,
          .minFilter = vk::Filter::eLinear,
          .mipmapMode = vk::SamplerMipmapMode::eNearest,
          .addressModeU = vk::SamplerAddressMode::eClampToEdge,
          .addressModeV = vk::SamplerAddressMode::eClampToEdge,
          .addressModeW = vk::SamplerAddressMode::eClampToEdge,
          .mipLodBias = 0.0f,
          .anisotropyEnable = VK_FALSE,
          .maxAnisotropy = 1.0f,
          .compareEnable = VK_FALSE,
          .compareOp = vk::CompareOp::eAlways,
          .minLod = 0.0f,
          .maxLod = 0.0f,
          .borderColor = vk::BorderColor::eIntOpaqueBlack,
          .unnormalizedCoordinates = VK_FALSE
        };
        rqCompositeSampler = vk::raii::Sampler(device, sci);
      }
      std::vector<vk::DescriptorSetLayout> rqLayouts(MAX_FRAMES_IN_FLIGHT, *transparentDescriptorSetLayout);
      vk::DescriptorSetAllocateInfo rqAllocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts = rqLayouts.data()
      }; {
        auto sets = vk::raii::DescriptorSets(device, rqAllocInfo);
        rqCompositeDescriptorSets.clear();
        rqCompositeDescriptorSets.reserve(sets.size());
        for (auto& s : sets) {
          rqCompositeDescriptorSets.emplace_back(std::move(s));
        }
      }

      // Update each set to sample the rayQueryOutputImage
      for (size_t i = 0; i < rqCompositeDescriptorSets.size(); ++i) {
        // Use a dedicated sampler to avoid null sampler issues during early init
        vk::Sampler samplerHandle = *rqCompositeSampler;
        vk::DescriptorImageInfo imgInfo{
          .sampler = samplerHandle,
          .imageView = *rayQueryOutputImageView,
          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };
        vk::WriteDescriptorSet write{
          .dstSet = *rqCompositeDescriptorSets[i],
          .dstBinding = 0,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .pImageInfo = &imgInfo
        };
        device.updateDescriptorSets({write}, {});
      }
    }

    // Create dedicated UBO buffers for ray query (one per frame in flight)
    rayQueryUniformBuffers.clear();
    rayQueryUniformAllocations.clear();
    rayQueryUniformBuffersMapped.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      auto [uboBuffer, uboAlloc] = createVmaBuffer(
        sizeof(RayQueryUniformBufferObject),
        vk::BufferUsageFlagBits::eUniformBuffer,
        VMA_MEMORY_USAGE_CPU_TO_GPU,
        VMA_ALLOCATION_CREATE_MAPPED_BIT);

      rayQueryUniformBuffers.push_back(std::move(uboBuffer));
      rayQueryUniformAllocations.push_back(uboAlloc);
      {
        VmaAllocationInfo info{};
        vmaGetAllocationInfo(vmaAllocator, uboAlloc, &info);
        rayQueryUniformBuffersMapped.push_back(info.pMappedData);
      }
    }

    std::cout << "Ray query resources created successfully (including " << MAX_FRAMES_IN_FLIGHT << " dedicated UBOs)\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create ray query resources: " << e.what() << std::endl;
    return false;
  }
}
bool Renderer::createSkyPipeline() {
  try {
    // Destroy old sky descriptor resources in reverse-dependency order before recreating.
    // (The RAII pool destructor fires while the set RAII still holds a handle to that pool,
    //  causing vkFreeDescriptorSets on an already-destroyed pool on the second call.)
    skyDescriptorSet         = vk::raii::DescriptorSet(nullptr);
    skyDescriptorPool        = vk::raii::DescriptorPool(nullptr);
    skyDescriptorSetLayout   = vk::raii::DescriptorSetLayout(nullptr);
    skyPipelineLayout        = vk::raii::PipelineLayout(nullptr);
    skyPipeline              = vk::raii::Pipeline(nullptr);

    // --- Descriptor set layout: binding 0 = combined image sampler (env map) ---
    vk::DescriptorSetLayoutBinding envBinding{
      .binding         = 0,
      .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = 1,
      .stageFlags      = vk::ShaderStageFlagBits::eFragment
    };
    vk::DescriptorSetLayoutCreateInfo dslCI{
      .bindingCount = 1,
      .pBindings    = &envBinding
    };
    skyDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, dslCI);

    // --- Descriptor pool (1 set, 1 combined image sampler) ---
    vk::DescriptorPoolSize poolSize{
      .type            = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = 1
    };
    vk::DescriptorPoolCreateInfo poolCI{
      .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets       = 1,
      .poolSizeCount = 1,
      .pPoolSizes    = &poolSize
    };
    skyDescriptorPool = vk::raii::DescriptorPool(device, poolCI);

    // --- Allocate descriptor set ---
    vk::DescriptorSetLayout dslHandle = *skyDescriptorSetLayout;
    vk::DescriptorSetAllocateInfo dsAllocInfo{
      .descriptorPool     = *skyDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &dslHandle
    };
    vk::raii::DescriptorSets dsets(device, dsAllocInfo);
    skyDescriptorSet = std::move(dsets[0]);

    // Write descriptor set with the default 1x1 texture as placeholder until env map loads
    {
      vk::DescriptorImageInfo imgInfo{
        .sampler     = *defaultTextureResources.textureSampler,
        .imageView   = *defaultTextureResources.textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
      };
      vk::WriteDescriptorSet write{
        .dstSet          = *skyDescriptorSet,
        .dstBinding      = 0,
        .descriptorCount = 1,
        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo      = &imgInfo
      };
      device.updateDescriptorSets(write, {});
    }

    // Push constant range: SkyPushConstants (mat4=64 + 48 = 112 bytes total)
    vk::PushConstantRange pushRange{
      .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
      .offset = 0,
      .size = sizeof(SkyPushConstants)
    };
    vk::DescriptorSetLayout skyDslHandle = *skyDescriptorSetLayout;
    vk::PipelineLayoutCreateInfo layoutInfo{
      .setLayoutCount         = 1,
      .pSetLayouts            = &skyDslHandle,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushRange
    };
    skyPipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);

    auto vertCode = readFile("shaders/sky_vsmain.spv");
    auto fragCode = readFile("shaders/sky_psmain.spv");
    vk::raii::ShaderModule vertMod = createShaderModule(vertCode);
    vk::raii::ShaderModule fragMod = createShaderModule(fragCode);

    vk::PipelineShaderStageCreateInfo stages[2] = {
      {.stage = vk::ShaderStageFlagBits::eVertex,   .module = *vertMod, .pName = "main"},
      {.stage = vk::ShaderStageFlagBits::eFragment, .module = *fragMod, .pName = "main"}
    };

    vk::PipelineVertexInputStateCreateInfo vertexInput{
      .vertexBindingDescriptionCount   = 0,
      .vertexAttributeDescriptionCount = 0
    };
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
      .topology = vk::PrimitiveTopology::eTriangleList
    };
    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode    = vk::CullModeFlagBits::eNone,
      .frontFace   = vk::FrontFace::eCounterClockwise,
      .lineWidth   = 1.0f
    };
    vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1
    };
    // Depth test EQUAL (only draw sky on far plane pixels), no depth write
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable  = vk::True,
      .depthWriteEnable = vk::False,
      .depthCompareOp   = vk::CompareOp::eEqual
    };
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable    = vk::False,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .attachmentCount = 1,
      .pAttachments    = &colorBlendAttachment
    };
    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = (uint32_t)dynamicStates.size(),
      .pDynamicStates    = dynamicStates.data()
    };

    vk::Format depthFmt = findDepthFormat();
    vk::PipelineRenderingCreateInfo renderingInfo{
      .colorAttachmentCount    = 1,
      .pColorAttachmentFormats = &swapChainImageFormat,
      .depthAttachmentFormat   = depthFmt
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext               = &renderingInfo,
      .stageCount          = 2,
      .pStages             = stages,
      .pVertexInputState   = &vertexInput,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState      = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState   = &multisampling,
      .pDepthStencilState  = &depthStencil,
      .pColorBlendState    = &colorBlending,
      .pDynamicState       = &dynamicState,
      .layout              = *skyPipelineLayout
    };
    skyPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    std::cout << "[Sky] Sky pipeline created successfully\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create sky pipeline: " << e.what() << std::endl;
    return false;
  }
}

bool Renderer::createWaterPipeline() {
  try {
    // ---- Descriptor set layout ----
    // binding 0: WaterVertexUBO (vertex)
    // binding 1: WaterFragUBO   (fragment)
    // binding 2: displacement map (frag + vert sampler)
    // binding 3: normal map (frag + vert sampler)
    // binding 4: env/sky equirectangular map (fragment — reflections)
    std::array<vk::DescriptorSetLayoutBinding, 5> dslBindings = {{
      {.binding=0, .descriptorType=vk::DescriptorType::eUniformBuffer,
       .descriptorCount=1, .stageFlags=vk::ShaderStageFlagBits::eVertex},
      {.binding=1, .descriptorType=vk::DescriptorType::eUniformBuffer,
       .descriptorCount=1, .stageFlags=vk::ShaderStageFlagBits::eFragment},
      {.binding=2, .descriptorType=vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount=1, .stageFlags=vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
      {.binding=3, .descriptorType=vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount=1, .stageFlags=vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
      {.binding=4, .descriptorType=vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount=1, .stageFlags=vk::ShaderStageFlagBits::eFragment},
    }};
    vk::DescriptorSetLayoutCreateInfo dslInfo{
      .bindingCount = static_cast<uint32_t>(dslBindings.size()),
      .pBindings    = dslBindings.data()
    };
    waterDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, dslInfo);

    vk::DescriptorSetLayout dslHandle = *waterDescriptorSetLayout;
    vk::PipelineLayoutCreateInfo layoutCI{
      .setLayoutCount = 1,
      .pSetLayouts    = &dslHandle
    };
    waterPipelineLayout = vk::raii::PipelineLayout(device, layoutCI);

    // ---- Shaders ----
    auto vertCode = readFile("shaders/water_vsmain.spv");
    auto fragCode = readFile("shaders/water_psmain.spv");
    vk::raii::ShaderModule vertMod = createShaderModule(vertCode);
    vk::raii::ShaderModule fragMod = createShaderModule(fragCode);

    vk::PipelineShaderStageCreateInfo stages[2] = {
      {.stage = vk::ShaderStageFlagBits::eVertex,   .module = *vertMod, .pName = "main"},
      {.stage = vk::ShaderStageFlagBits::eFragment, .module = *fragMod, .pName = "main"}
    };

    // ---- Vertex input: WaterVertex { vec3 pos, vec2 uv } ----
    vk::VertexInputBindingDescription vtxBinding{
      .binding   = 0,
      .stride    = sizeof(float) * 5, // vec3 + vec2
      .inputRate = vk::VertexInputRate::eVertex
    };
    std::array<vk::VertexInputAttributeDescription, 2> vtxAttribs = {{
      {.location=0, .binding=0, .format=vk::Format::eR32G32B32Sfloat, .offset=0},
      {.location=1, .binding=0, .format=vk::Format::eR32G32Sfloat,    .offset=sizeof(float)*3},
    }};
    vk::PipelineVertexInputStateCreateInfo vertexInput{
      .vertexBindingDescriptionCount   = 1,
      .pVertexBindingDescriptions      = &vtxBinding,
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(vtxAttribs.size()),
      .pVertexAttributeDescriptions    = vtxAttribs.data()
    };
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount=1, .scissorCount=1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode    = vk::CullModeFlagBits::eNone,
      .frontFace   = vk::FrontFace::eCounterClockwise,
      .lineWidth   = 1.0f
    };
    vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1
    };
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable  = vk::True,
      .depthWriteEnable = vk::True,
      .depthCompareOp   = vk::CompareOp::eLess
    };
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable    = vk::False,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .attachmentCount = 1,
      .pAttachments    = &colorBlendAttachment
    };
    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = (uint32_t)dynamicStates.size(),
      .pDynamicStates    = dynamicStates.data()
    };
    vk::Format depthFmt = findDepthFormat();
    vk::PipelineRenderingCreateInfo renderingInfo{
      .colorAttachmentCount    = 1,
      .pColorAttachmentFormats = &swapChainImageFormat,
      .depthAttachmentFormat   = depthFmt
    };
    vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext               = &renderingInfo,
      .stageCount          = 2,
      .pStages             = stages,
      .pVertexInputState   = &vertexInput,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState      = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState   = &multisampling,
      .pDepthStencilState  = &depthStencil,
      .pColorBlendState    = &colorBlending,
      .pDynamicState       = &dynamicState,
      .layout              = *waterPipelineLayout
    };
    waterPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    std::cout << "[Water] Water pipeline created successfully\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create water pipeline: " << e.what() << std::endl;
    return false;
  }
}

bool Renderer::createCSMDepthPipeline() {
  try {
    // Descriptor set layout: binding 0 = CsmDepthUBO (vertex stage)
    vk::DescriptorSetLayoutBinding uboBinding{
      .binding         = 0,
      .descriptorType  = vk::DescriptorType::eUniformBuffer,
      .descriptorCount = 1,
      .stageFlags      = vk::ShaderStageFlagBits::eVertex
    };
    vk::DescriptorSetLayoutCreateInfo dslInfo{
      .bindingCount = 1,
      .pBindings    = &uboBinding
    };
    csmDepthDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, dslInfo);

    // Push constants: model matrix (64) + cascade index (4) + pad (12)
    vk::PushConstantRange pushRange{
      .stageFlags = vk::ShaderStageFlagBits::eVertex,
      .offset     = 0,
      .size       = 80  // 64 + 4 + 12 pad
    };
    vk::DescriptorSetLayout dslHandle = *csmDepthDescriptorSetLayout;
    vk::PipelineLayoutCreateInfo layoutInfo{
      .setLayoutCount         = 1,
      .pSetLayouts            = &dslHandle,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushRange
    };
    csmDepthPipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);

    auto vertCode = readFile("shaders/csm_depth_vsmain.spv");
    vk::raii::ShaderModule vertMod = createShaderModule(vertCode);
    vk::PipelineShaderStageCreateInfo stage{
      .stage  = vk::ShaderStageFlagBits::eVertex,
      .module = *vertMod,
      .pName  = "main"
    };

    // Vertex input: position only (first attribute, binding 0)
    vk::VertexInputBindingDescription vtxBinding{
      .binding   = 0,
      .stride    = sizeof(Vertex),
      .inputRate = vk::VertexInputRate::eVertex
    };
    vk::VertexInputAttributeDescription vtxAttr{
      .location = 0,
      .binding  = 0,
      .format   = vk::Format::eR32G32B32Sfloat,
      .offset   = offsetof(Vertex, position)
    };
    vk::PipelineVertexInputStateCreateInfo vtxInput{
      .vertexBindingDescriptionCount   = 1,
      .pVertexBindingDescriptions      = &vtxBinding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions    = &vtxAttr
    };
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .polygonMode     = vk::PolygonMode::eFill,
      .cullMode        = vk::CullModeFlagBits::eBack,
      .frontFace       = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = vk::True,
      .lineWidth       = 1.0f
    };
    vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable  = vk::True,
      .depthWriteEnable = vk::True,
      .depthCompareOp   = vk::CompareOp::eLessOrEqual
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{.attachmentCount = 0};
    std::vector<vk::DynamicState> dynamicStates = {
      vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eDepthBias
    };
    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = (uint32_t)dynamicStates.size(),
      .pDynamicStates    = dynamicStates.data()
    };
    vk::Format depthFmt = vk::Format::eD32Sfloat;
    vk::PipelineRenderingCreateInfo renderingInfo{
      .depthAttachmentFormat = depthFmt
    };
    vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext               = &renderingInfo,
      .stageCount          = 1,
      .pStages             = &stage,
      .pVertexInputState   = &vtxInput,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState      = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState   = &multisampling,
      .pDepthStencilState  = &depthStencil,
      .pColorBlendState    = &colorBlending,
      .pDynamicState       = &dynamicState,
      .layout              = *csmDepthPipelineLayout
    };
    csmDepthPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    std::cout << "[CSM] CSM depth pipeline created successfully\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create CSM depth pipeline: " << e.what() << std::endl;
    return false;
  }
}

// =============================================================================
// Bloom pipelines [Jimenez 2014] — dual-Kawase downsample + tent upsample
// =============================================================================
bool Renderer::createBloomPipelines() {
  try {
    bloomDownsamplePipeline = vk::raii::Pipeline(nullptr);
    bloomUpsamplePipeline   = vk::raii::Pipeline(nullptr);
    bloomPipelineLayout     = vk::raii::PipelineLayout(nullptr);
    bloomDescriptorSetLayout = vk::raii::DescriptorSetLayout(nullptr);

    // --- Descriptor set layout ---
    // binding 0 = source image (srcImage), binding 1 = add image (addImage / unused in downsample)
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
      {.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
      {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment}
    }};
    vk::DescriptorSetLayoutCreateInfo dslCI{
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings    = bindings.data()
    };
    bloomDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, dslCI);

    // Push constant: BloomPush (32 bytes)
    vk::PushConstantRange pushRange{
      .stageFlags = vk::ShaderStageFlagBits::eFragment,
      .offset = 0,
      .size   = 32  // float2 texelSize + int mipLevel + float threshold + float strength + float3 _pad
    };
    vk::DescriptorSetLayout dslHandle = *bloomDescriptorSetLayout;
    vk::PipelineLayoutCreateInfo plCI{
      .setLayoutCount         = 1,
      .pSetLayouts            = &dslHandle,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushRange
    };
    bloomPipelineLayout = vk::raii::PipelineLayout(device, plCI);

    // --- Shared pipeline state ---
    vk::PipelineVertexInputStateCreateInfo vertexInput{
      .vertexBindingDescriptionCount   = 0,
      .vertexAttributeDescriptionCount = 0
    };
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
      .topology = vk::PrimitiveTopology::eTriangleList
    };
    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode    = vk::CullModeFlagBits::eNone,
      .frontFace   = vk::FrontFace::eCounterClockwise,
      .lineWidth   = 1.0f
    };
    vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1
    };
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable  = vk::False,
      .depthWriteEnable = vk::False
    };
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable    = vk::False,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .attachmentCount = 1,
      .pAttachments    = &colorBlendAttachment
    };
    std::vector<vk::DynamicState> dynStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = (uint32_t)dynStates.size(),
      .pDynamicStates    = dynStates.data()
    };

    // Bloom renders into R16G16B16A16_SFLOAT off-screen images
    vk::Format bloomFmt = vk::Format::eR16G16B16A16Sfloat;
    vk::PipelineRenderingCreateInfo renderingInfo{
      .colorAttachmentCount    = 1,
      .pColorAttachmentFormats = &bloomFmt,
      .depthAttachmentFormat   = vk::Format::eUndefined
    };

    // Shared vertex shader
    auto vertCode = readFile("shaders/bloom_vsmain.spv");
    vk::raii::ShaderModule vertMod = createShaderModule(vertCode);

    // --- Downsample pipeline ---
    {
      auto fragCode = readFile("shaders/bloom_downsampleps.spv");
      vk::raii::ShaderModule fragMod = createShaderModule(fragCode);
      vk::PipelineShaderStageCreateInfo stages[2] = {
        {.stage = vk::ShaderStageFlagBits::eVertex,   .module = *vertMod, .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment, .module = *fragMod, .pName = "main"}
      };
      vk::GraphicsPipelineCreateInfo pipeCI{
        .pNext               = &renderingInfo,
        .stageCount          = 2,
        .pStages             = stages,
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState   = &multisampling,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &colorBlending,
        .pDynamicState       = &dynamicState,
        .layout              = *bloomPipelineLayout
      };
      bloomDownsamplePipeline = vk::raii::Pipeline(device, nullptr, pipeCI);
    }

    // --- Upsample pipeline ---
    {
      auto fragCode = readFile("shaders/bloom_upsampleps.spv");
      vk::raii::ShaderModule fragMod = createShaderModule(fragCode);
      vk::PipelineShaderStageCreateInfo stages[2] = {
        {.stage = vk::ShaderStageFlagBits::eVertex,   .module = *vertMod, .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment, .module = *fragMod, .pName = "main"}
      };
      vk::GraphicsPipelineCreateInfo pipeCI{
        .pNext               = &renderingInfo,
        .stageCount          = 2,
        .pStages             = stages,
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState   = &multisampling,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &colorBlending,
        .pDynamicState       = &dynamicState,
        .layout              = *bloomPipelineLayout
      };
      bloomUpsamplePipeline = vk::raii::Pipeline(device, nullptr, pipeCI);
    }

    // --- Additive blend pipeline: renders bloom contribution into opaqueSceneColorImages ---
    // This is for additively compositing the final bloom mip into the HDR scene before tonemapping.
    // The render target is R16G16B16A16_SFLOAT (same format as bloomMips).
    {
      auto fragCode = readFile("shaders/bloom_addps.spv");
      vk::raii::ShaderModule fragMod = createShaderModule(fragCode);
      vk::PipelineShaderStageCreateInfo stages[2] = {
        {.stage = vk::ShaderStageFlagBits::eVertex,   .module = *vertMod, .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment, .module = *fragMod, .pName = "main"}
      };
      // Additive blending: output = dst + src * strength
      vk::PipelineColorBlendAttachmentState addBlend{
        .blendEnable         = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eOne,
        .colorBlendOp        = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp        = vk::BlendOp::eAdd,
        .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                               vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
      };
      vk::PipelineColorBlendStateCreateInfo addBlending{.attachmentCount = 1, .pAttachments = &addBlend};
      // The render target for the bloom add pass is opaqueSceneColorImages (same format as swapchain)
      vk::PipelineRenderingCreateInfo hdrRenderingInfo{
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &swapChainImageFormat,
        .depthAttachmentFormat   = vk::Format::eUndefined
      };
      vk::GraphicsPipelineCreateInfo pipeCI{
        .pNext               = &hdrRenderingInfo,
        .stageCount          = 2,
        .pStages             = stages,
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState   = &multisampling,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &addBlending,
        .pDynamicState       = &dynamicState,
        .layout              = *bloomPipelineLayout
      };
      bloomAddPipeline = vk::raii::Pipeline(device, nullptr, pipeCI);
    }

    std::cout << "[Bloom] Bloom pipelines created.\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[Bloom] Failed to create bloom pipelines: " << e.what() << "\n";
    return false;
  }
}

// =============================================================================
// Histogram auto-exposure compute pipelines [Chapelle 2018]
// =============================================================================
bool Renderer::createExposurePipelines() {
  try {
    exposureHistogramPipeline  = vk::raii::Pipeline(nullptr);
    exposureReducePipeline     = vk::raii::Pipeline(nullptr);
    exposurePipelineLayout     = vk::raii::PipelineLayout(nullptr);
    exposureDescriptorSetLayout = vk::raii::DescriptorSetLayout(nullptr);

    // --- Descriptor set layout ---
    // binding 0 = sceneColor (combined image sampler, read-only)
    // binding 1 = histogram SSBO (RWStructuredBuffer<uint>)
    // binding 2 = exposureBuffer SSBO (RWStructuredBuffer<float>)
    std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
      {.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
       .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
      {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer,
       .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
      {.binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer,
       .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    vk::DescriptorSetLayoutCreateInfo dslCI{
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings    = bindings.data()
    };
    exposureDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, dslCI);

    // Push constant: ExposurePush (48 bytes)
    vk::PushConstantRange pushRange{
      .stageFlags = vk::ShaderStageFlagBits::eCompute,
      .offset = 0,
      .size   = 48
    };
    vk::DescriptorSetLayout dslHandle = *exposureDescriptorSetLayout;
    vk::PipelineLayoutCreateInfo plCI{
      .setLayoutCount         = 1,
      .pSetLayouts            = &dslHandle,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushRange
    };
    exposurePipelineLayout = vk::raii::PipelineLayout(device, plCI);

    // BuildHistogramCS
    {
      auto code = readFile("shaders/exposure_buildhistogramcs.spv");
      vk::raii::ShaderModule mod = createShaderModule(code);
      vk::PipelineShaderStageCreateInfo stage{
        .stage  = vk::ShaderStageFlagBits::eCompute,
        .module = *mod,
        .pName  = "main"
      };
      vk::ComputePipelineCreateInfo pipeCI{.stage = stage, .layout = *exposurePipelineLayout};
      exposureHistogramPipeline = vk::raii::Pipeline(device, nullptr, pipeCI);
    }

    // ReduceHistogramCS
    {
      auto code = readFile("shaders/exposure_reducehistogramcs.spv");
      vk::raii::ShaderModule mod = createShaderModule(code);
      vk::PipelineShaderStageCreateInfo stage{
        .stage  = vk::ShaderStageFlagBits::eCompute,
        .module = *mod,
        .pName  = "main"
      };
      vk::ComputePipelineCreateInfo pipeCI{.stage = stage, .layout = *exposurePipelineLayout};
      exposureReducePipeline = vk::raii::Pipeline(device, nullptr, pipeCI);
    }

    std::cout << "[Exposure] Auto-exposure compute pipelines created.\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[Exposure] Failed to create exposure pipelines: " << e.what() << "\n";
    return false;
  }
}
