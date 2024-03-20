#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "TaskScheduler.h"

#include <vulkan/vk.hpp>

namespace vk {
    VkResult loadShaderModule(std::filesystem::path filePath, VkDevice device, VkShaderModule* pShaderModule);

    class PipelineBuilder {
    protected:
        VkDevice device;
        VkPhysicalDevice physicalDevice;

        VkPipelineCache pipelineCache = nullptr;

#ifdef _MSC_VER
#pragma pack (push, 1)
#endif
        struct [[gnu::packed]] PipelineCacheHeader {
            std::uint32_t headerSize;
            VkPipelineCacheHeaderVersion headerVersion;
            std::uint32_t vendorID;
            std::uint32_t deviceID;
            std::array<std::uint8_t, VK_UUID_SIZE> pipelineCacheUUID; // Same size as normal array on MSVC, Clang, and GCC.
        };
        static_assert(sizeof(PipelineCacheHeader) == 32, "Vulkan spec requires header to be 32 bytes");
        static_assert(sizeof(std::array<std::uint8_t, VK_UUID_SIZE>) == sizeof(std::uint8_t[VK_UUID_SIZE]));
#ifdef _MSC_VER
#pragma pack(pop)
#endif

        explicit PipelineBuilder(VkDevice device, VkPhysicalDevice physicalDevice);

        virtual VkResult build(VkPipeline* pipeline) noexcept = 0;
        virtual PipelineBuilder& pushPNext(std::uint32_t idx, const void* pNext) = 0;
        /**
         * Pipeline caches of multiple pipelines have to be merged together to be used here, as Vulkan
         * only accepts a single pipeline cache handle.
         */
        virtual PipelineBuilder& setPipelineCache(VkPipelineCache cache);
        virtual PipelineBuilder& setPipelineCount(std::uint32_t count) = 0;
        virtual PipelineBuilder& setPipelineFlags(std::uint32_t idx, VkPipelineCreateFlags flags) = 0;
        virtual PipelineBuilder& setPipelineLayout(std::uint32_t idx, VkPipelineLayout layout) = 0;
    };

    class ComputePipelineBuilder final : PipelineBuilder {
        std::vector<VkComputePipelineCreateInfo> pipelineInfos;

    public:
        explicit ComputePipelineBuilder(VkDevice device, VkPhysicalDevice physicalDevice);

        VkResult build(VkPipeline* pipeline) noexcept override;
        ComputePipelineBuilder& pushPNext(std::uint32_t idx, const void* pNext) override;
        ComputePipelineBuilder& setPipelineCount(std::uint32_t count) override;
        ComputePipelineBuilder& setPipelineLayout(std::uint32_t idx, VkPipelineLayout layout) override;
        ComputePipelineBuilder& setPipelineFlags(std::uint32_t idx, VkPipelineCreateFlags flags) override;
        ComputePipelineBuilder& setShaderStage(std::uint32_t idx, VkShaderStageFlagBits stage, VkShaderModule module, std::string_view name);
    };

    class GraphicsPipelineBuilder final : PipelineBuilder {
        struct PipelineBuildInfos {
            VkPipelineVertexInputStateCreateInfo vertexInputState;
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyState;
            VkPipelineTessellationStateCreateInfo tessellationState;
            VkPipelineViewportStateCreateInfo viewportState;
            VkPipelineRasterizationStateCreateInfo rasterState;
            VkPipelineMultisampleStateCreateInfo multisampleState;
            VkPipelineDepthStencilStateCreateInfo depthState;
            VkPipelineColorBlendStateCreateInfo blendStateInfo;
            std::vector<VkPipelineShaderStageCreateInfo> stages;
            std::vector<VkDynamicState> dynamicStateValues;
        };

        std::vector<VkGraphicsPipelineCreateInfo> pipelineInfos;
        std::vector<PipelineBuildInfos> pipelineBuildInfos;

    public:
        explicit GraphicsPipelineBuilder(VkDevice device, VkPhysicalDevice physicalDevice);

        GraphicsPipelineBuilder& addDynamicState(std::uint32_t idx, VkDynamicState state);
        GraphicsPipelineBuilder& addShaderStage(std::uint32_t idx, VkShaderStageFlagBits stage, VkShaderModule module, std::string_view name, const VkSpecializationInfo* specInfo = nullptr);
        VkResult build(VkPipeline* pipeline) noexcept override;
        /**
         * This updates the pNext member of the VkGraphicsPipelineCreateInfo to point to the given
         * pNext. It also writes the previous pNext member of VkGraphicsPipelineCreateInfo to the pNext
         * of the given structure, overwriting any previous assignments.
         */
        GraphicsPipelineBuilder& pushPNext(std::uint32_t idx, const void* pNext) override;
        GraphicsPipelineBuilder& setBlendAttachment(std::uint32_t idx, const VkPipelineColorBlendAttachmentState* state);
        GraphicsPipelineBuilder& setBlendAttachments(std::uint32_t idx, std::span<VkPipelineColorBlendAttachmentState> states);
        GraphicsPipelineBuilder& setBlendConstants(std::uint32_t idx, std::array<float, 4>& constants);
        GraphicsPipelineBuilder& setDepthState(std::uint32_t idx, VkBool32 depthTestEnable, VkBool32 depthWriteEnable,
                                               VkCompareOp depthCompareOp);
        GraphicsPipelineBuilder& setMultisampleCount(std::uint32_t idx, VkSampleCountFlagBits samples);
		GraphicsPipelineBuilder& setPipelineCache(VkPipelineCache cache) override;
        /**
         * This creates everything needed to build the pipeline. Note that this is required to be
         * called for every pipeline builder with a count of at least 1.
         */
        GraphicsPipelineBuilder& setPipelineCount(std::uint32_t count) override;
        GraphicsPipelineBuilder& setPipelineLayout(std::uint32_t idx, VkPipelineLayout layout) override;
        GraphicsPipelineBuilder& setPipelineFlags(std::uint32_t idx, VkPipelineCreateFlags flags) override;
        GraphicsPipelineBuilder& setRasterState(std::uint32_t idx, VkPolygonMode polygonMode, VkCullModeFlags cullMode, VkFrontFace frontFace,
                                                float lineWidth = 1.0F);
        GraphicsPipelineBuilder& setScissorCount(std::uint32_t idx, std::uint32_t scissorCount);
        GraphicsPipelineBuilder& setScissors(std::uint32_t idx, std::span<const VkRect2D> scissors);
        GraphicsPipelineBuilder& setTopology(std::uint32_t idx, VkPrimitiveTopology topology, VkBool32 restartEnable = VK_FALSE);
        GraphicsPipelineBuilder& setViewportCount(std::uint32_t idx, std::uint32_t viewportCount);
        GraphicsPipelineBuilder& setViewports(std::uint32_t idx, std::span<const VkViewport> viewports);
    };
} // namespace vk
