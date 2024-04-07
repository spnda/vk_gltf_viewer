#include <cassert>
#include <fstream>

#include <tracy/Tracy.hpp>

#include <vulkan/pipeline_builder.hpp>

VkResult vk::loadShaderModule(std::filesystem::path filePath, VkDevice device, VkShaderModule *pShaderModule) {
    std::error_code error;
    auto length = static_cast<std::streamsize>(std::filesystem::file_size(filePath, error));
    if (error) {
        return VK_ERROR_UNKNOWN;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return VK_ERROR_UNKNOWN;
    }

    auto buffer = std::make_unique_for_overwrite<std::uint32_t[]>(length / sizeof(std::uint32_t));
    file.read((char*)buffer.get(), length);

    const VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = static_cast<std::uint32_t>(length),
        .pCode = buffer.get(),
    };

    return vkCreateShaderModule(device, &createInfo, VK_NULL_HANDLE, pShaderModule);
}

vk::PipelineBuilder::PipelineBuilder(VkDevice device)
        : device(device) {}


vk::PipelineBuilder& vk::PipelineBuilder::setPipelineCache(VkPipelineCache cache) {
    pipelineCache = cache;
    return *this;
}

vk::ComputePipelineBuilder::ComputePipelineBuilder(VkDevice device, std::uint32_t count)
        : PipelineBuilder(device) {
	assert(count != 0);
	pipelineInfos.resize(count);
    for (auto& pipelineInfo : pipelineInfos) {
        pipelineInfo = {
        	.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        };
    }
}

VkResult vk::ComputePipelineBuilder::build(VkPipeline* pipeline) noexcept {
    ZoneScoped;
    if (pipelineInfos.empty()) {
        return VK_ERROR_NOT_PERMITTED_KHR;
    }
    assert(pipelineInfos.size() < std::numeric_limits<std::uint32_t>::max());

    return vkCreateComputePipelines(device, pipelineCache, static_cast<std::uint32_t>(pipelineInfos.size()), pipelineInfos.data(), nullptr, pipeline);
}

vk::ComputePipelineBuilder& vk::ComputePipelineBuilder::pushPNext(std::uint32_t idx, const void* pNext) {
    assert(idx < pipelineInfos.size());
    ((VkBaseOutStructure*)pNext)->pNext = (VkBaseOutStructure*)pipelineInfos[idx].pNext;
    pipelineInfos[idx].pNext = pNext;
    return *this;
}

vk::ComputePipelineBuilder& vk::ComputePipelineBuilder::setPipelineLayout(std::uint32_t idx, VkPipelineLayout layout) {
    assert(idx < pipelineInfos.size());
    pipelineInfos[idx].layout = layout;
    return *this;
}

vk::ComputePipelineBuilder& vk::ComputePipelineBuilder::setPipelineFlags(std::uint32_t idx, VkPipelineCreateFlags flags) {
    assert(idx < pipelineInfos.size());
    pipelineInfos[idx].flags = flags;
    return *this;
}

vk::ComputePipelineBuilder& vk::ComputePipelineBuilder::setShaderStage(std::uint32_t idx, VkShaderStageFlagBits stage, VkShaderModule module, std::string_view name) {
    assert(idx < pipelineInfos.size());
    pipelineInfos[idx].stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = stage,
            .module = module,
            .pName = name.data(),
    };
    return *this;
}

vk::GraphicsPipelineBuilder::GraphicsPipelineBuilder(VkDevice device, std::uint32_t count)
        : PipelineBuilder(device) {
	assert(count != 0);
    pipelineInfos.resize(static_cast<std::size_t>(count));
    pipelineBuildInfos.resize(static_cast<std::size_t>(count));

    for (std::size_t i = 0; i < pipelineInfos.size(); ++i) {
        pipelineInfos[i] = {
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        };
        pipelineBuildInfos[i].blendStateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        };
        pipelineBuildInfos[i].vertexInputState = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        };
        pipelineBuildInfos[i].inputAssemblyState = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        };
        pipelineBuildInfos[i].tessellationState = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        };
        pipelineBuildInfos[i].viewportState = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        };
        pipelineBuildInfos[i].multisampleState = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        };
    }
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::addDynamicState(std::uint32_t idx, VkDynamicState state) {
    assert(idx < pipelineBuildInfos.size());
    pipelineBuildInfos[idx].dynamicStateValues.emplace_back(state);
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::addShaderStage(std::uint32_t idx, VkShaderStageFlagBits stage, VkShaderModule module,
                                                                         std::string_view name, const VkSpecializationInfo* specInfo) {
    assert(idx < pipelineBuildInfos.size());
    pipelineBuildInfos[idx].stages.emplace_back(VkPipelineShaderStageCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.flags = 0,
            .stage = stage,
            .module = module,
            .pName = name.data(),
			.pSpecializationInfo = specInfo,
    });
    return *this;
}

VkResult vk::GraphicsPipelineBuilder::build(VkPipeline* pipeline) noexcept {
    ZoneScoped;
    if (pipelineInfos.empty()) {
        return VK_ERROR_NOT_PERMITTED_KHR;
    }
    assert(pipelineInfos.size() < std::numeric_limits<std::uint32_t>::max());

    std::vector<VkPipelineDynamicStateCreateInfo> dynamicStateInfos(pipelineInfos.size());
    for (std::size_t i = 0; i < pipelineInfos.size(); ++i) {
        auto& info = pipelineInfos[i];
        auto& buildInfo = pipelineBuildInfos[i];

        info.pVertexInputState = &buildInfo.vertexInputState;
        info.pInputAssemblyState = &buildInfo.inputAssemblyState;
        info.pTessellationState = &buildInfo.tessellationState;
        info.pViewportState = &buildInfo.viewportState;
        info.pRasterizationState = &buildInfo.rasterState;
        info.pMultisampleState = &buildInfo.multisampleState;
        info.pDepthStencilState = &buildInfo.depthState;
        info.pColorBlendState = &buildInfo.blendStateInfo;

        // Update the stage counts & data pointers.
        info.stageCount = static_cast<std::uint32_t>(buildInfo.stages.size());
        info.pStages = buildInfo.stages.data();

        // Update the dynamic state counts & data pointers.
        dynamicStateInfos[i] = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = static_cast<std::uint32_t>(buildInfo.dynamicStateValues.size()),
                .pDynamicStates = buildInfo.dynamicStateValues.data(),
        };
        info.pDynamicState = &dynamicStateInfos[i];

        if (buildInfo.multisampleState.rasterizationSamples == 0) {
            buildInfo.multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        }
    }

    return vkCreateGraphicsPipelines(
            device, pipelineCache, static_cast<std::uint32_t>(pipelineInfos.size()), pipelineInfos.data(), nullptr, pipeline);
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::pushPNext(std::uint32_t idx, const void* pNext) {
    assert(idx < pipelineInfos.size());
    ((VkBaseOutStructure*)pNext)->pNext = (VkBaseOutStructure*)pipelineInfos[idx].pNext;
    pipelineInfos[idx].pNext = pNext;
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setBlendAttachment(std::uint32_t idx,
                                                                             const VkPipelineColorBlendAttachmentState* state) {
   	assert(idx < pipelineBuildInfos.size());
    pipelineBuildInfos[idx].blendStateInfo.attachmentCount = 1;
    pipelineBuildInfos[idx].blendStateInfo.pAttachments = state;
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setBlendAttachments(std::uint32_t idx,
                                                                              std::span<VkPipelineColorBlendAttachmentState> states) {
    assert(idx < pipelineBuildInfos.size());
    pipelineBuildInfos[idx].blendStateInfo.attachmentCount = static_cast<std::uint32_t>(states.size());
    pipelineBuildInfos[idx].blendStateInfo.pAttachments = states.data();
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setBlendConstants(std::uint32_t idx, std::array<float, 4>& constants) {
    assert(idx < pipelineBuildInfos.size());
    std::copy(constants.begin(), constants.end(), std::begin(pipelineBuildInfos[idx].blendStateInfo.blendConstants));
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setDepthState(std::uint32_t idx, VkBool32 depthTestEnable, VkBool32 depthWriteEnable,
                                                                        VkCompareOp depthCompareOp) {
    assert(idx < pipelineBuildInfos.size());
    pipelineBuildInfos[idx].depthState = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = depthTestEnable,
            .depthWriteEnable = depthWriteEnable,
            .depthCompareOp = depthCompareOp,
			.depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE, // TODO: Expose.
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f,
    };
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setMultisampleCount(std::uint32_t idx, VkSampleCountFlagBits samples) {
    assert(idx < pipelineBuildInfos.size());
    pipelineBuildInfos[idx].multisampleState.rasterizationSamples = samples;
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setPipelineCache(VkPipelineCache cache) {
	pipelineCache = cache;
	return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setPipelineLayout(std::uint32_t idx, VkPipelineLayout layout) {
    assert(idx < pipelineInfos.size());
    pipelineInfos[idx].layout = layout;
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setPipelineFlags(std::uint32_t idx, VkPipelineCreateFlags flags) {
    assert(idx < pipelineInfos.size());
    pipelineInfos[idx].flags = flags;
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setRasterState(std::uint32_t idx, VkPolygonMode polygonMode,
                                                                         VkCullModeFlags cullMode, VkFrontFace frontFace,
                                                                         float lineWidth) {
    assert(idx < pipelineBuildInfos.size());
    pipelineBuildInfos[idx].rasterState = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .polygonMode = polygonMode,
            .cullMode = cullMode,
            .frontFace = frontFace,
            .lineWidth = lineWidth,
    };
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setScissorCount(std::uint32_t idx, std::uint32_t scissorCount) {
    assert(idx < pipelineInfos.size());
    pipelineBuildInfos[idx].viewportState.scissorCount = scissorCount;
    pipelineBuildInfos[idx].viewportState.pScissors = nullptr;
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setScissors(std::uint32_t idx, std::span<const VkRect2D> scissors) {
    assert(idx < pipelineInfos.size() && scissors.size() < std::numeric_limits<std::uint32_t>::max());
    pipelineBuildInfos[idx].viewportState.scissorCount = static_cast<std::uint32_t>(scissors.size());
    pipelineBuildInfos[idx].viewportState.pScissors = scissors.data();
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setTopology(std::uint32_t idx, VkPrimitiveTopology topology,
                                                                      VkBool32 restartEnable) {
    assert(idx < pipelineInfos.size());
    pipelineBuildInfos[idx].inputAssemblyState.topology = topology;
    pipelineBuildInfos[idx].inputAssemblyState.primitiveRestartEnable = restartEnable;
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setViewportCount(std::uint32_t idx, std::uint32_t viewportCount) {
    assert(idx < pipelineInfos.size());
    pipelineBuildInfos[idx].viewportState.viewportCount = viewportCount;
    pipelineBuildInfos[idx].viewportState.pViewports = nullptr;
    return *this;
}

vk::GraphicsPipelineBuilder& vk::GraphicsPipelineBuilder::setViewports(std::uint32_t idx, std::span<const VkViewport> viewports) {
    assert(idx < pipelineInfos.size() && viewports.size() < std::numeric_limits<std::uint32_t>::max());
    pipelineBuildInfos[idx].viewportState.viewportCount = static_cast<std::uint32_t>(viewports.size());
    pipelineBuildInfos[idx].viewportState.pViewports = viewports.data();
    return *this;
}