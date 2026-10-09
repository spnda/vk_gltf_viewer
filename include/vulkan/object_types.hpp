#if !defined(VK_OBJECT_TYPE_CASE)
#error "Must define VK_OBJECT_TYPE_CASE before including this header"
#endif

#if !defined(VK_OBJECT_TYPE_CASE_FIRST)
#define VK_OBJECT_TYPE_CASE_FIRST(id, newObjectType) VK_OBJECT_TYPE_CASE(id, newObjectType)
#endif

VK_OBJECT_TYPE_CASE_FIRST(VkInstance, VK_OBJECT_TYPE_INSTANCE)
VK_OBJECT_TYPE_CASE(VkPhysicalDevice, VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_OBJECT_TYPE_CASE(VkDevice, VK_OBJECT_TYPE_DEVICE)
VK_OBJECT_TYPE_CASE(VkQueue, VK_OBJECT_TYPE_QUEUE)
VK_OBJECT_TYPE_CASE(VkSemaphore, VK_OBJECT_TYPE_SEMAPHORE)
VK_OBJECT_TYPE_CASE(VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER)
VK_OBJECT_TYPE_CASE(VkFence, VK_OBJECT_TYPE_FENCE)
VK_OBJECT_TYPE_CASE(VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_OBJECT_TYPE_CASE(VkBuffer, VK_OBJECT_TYPE_BUFFER)
VK_OBJECT_TYPE_CASE(VkImage, VK_OBJECT_TYPE_IMAGE)
VK_OBJECT_TYPE_CASE(VkEvent, VK_OBJECT_TYPE_EVENT)
VK_OBJECT_TYPE_CASE(VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL)
VK_OBJECT_TYPE_CASE(VkBufferView, VK_OBJECT_TYPE_BUFFER_VIEW)
VK_OBJECT_TYPE_CASE(VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW)
VK_OBJECT_TYPE_CASE(VkShaderModule, VK_OBJECT_TYPE_SHADER_MODULE)
VK_OBJECT_TYPE_CASE(VkPipelineCache, VK_OBJECT_TYPE_PIPELINE_CACHE)
VK_OBJECT_TYPE_CASE(VkPipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT)
VK_OBJECT_TYPE_CASE(VkRenderPass, VK_OBJECT_TYPE_RENDER_PASS)
VK_OBJECT_TYPE_CASE(VkPipeline, VK_OBJECT_TYPE_PIPELINE)
VK_OBJECT_TYPE_CASE(VkDescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VK_OBJECT_TYPE_CASE(VkSampler, VK_OBJECT_TYPE_SAMPLER)
VK_OBJECT_TYPE_CASE(VkDescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL)
VK_OBJECT_TYPE_CASE(VkDescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET)
VK_OBJECT_TYPE_CASE(VkFramebuffer, VK_OBJECT_TYPE_FRAMEBUFFER)
VK_OBJECT_TYPE_CASE(VkCommandPool, VK_OBJECT_TYPE_COMMAND_POOL)
VK_OBJECT_TYPE_CASE(VkSamplerYcbcrConversion, VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION)
VK_OBJECT_TYPE_CASE(VkDescriptorUpdateTemplate, VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE)
VK_OBJECT_TYPE_CASE(VkPrivateDataSlot, VK_OBJECT_TYPE_PRIVATE_DATA_SLOT)
VK_OBJECT_TYPE_CASE(VkSurfaceKHR, VK_OBJECT_TYPE_SURFACE_KHR)
VK_OBJECT_TYPE_CASE(VkSwapchainKHR, VK_OBJECT_TYPE_SWAPCHAIN_KHR)
VK_OBJECT_TYPE_CASE(VkDisplayKHR, VK_OBJECT_TYPE_DISPLAY_KHR)
VK_OBJECT_TYPE_CASE(VkDisplayModeKHR, VK_OBJECT_TYPE_DISPLAY_MODE_KHR)
VK_OBJECT_TYPE_CASE(VkDebugReportCallbackEXT, VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT)
VK_OBJECT_TYPE_CASE(VkVideoSessionKHR, VK_OBJECT_TYPE_VIDEO_SESSION_KHR)
VK_OBJECT_TYPE_CASE(VkVideoSessionParametersKHR, VK_OBJECT_TYPE_VIDEO_SESSION_PARAMETERS_KHR)
VK_OBJECT_TYPE_CASE(VkCuModuleNVX, VK_OBJECT_TYPE_CU_MODULE_NVX)
VK_OBJECT_TYPE_CASE(VkCuFunctionNVX, VK_OBJECT_TYPE_CU_FUNCTION_NVX)
VK_OBJECT_TYPE_CASE(VkDebugUtilsMessengerEXT, VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT)
VK_OBJECT_TYPE_CASE(VkAccelerationStructureKHR, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR)
VK_OBJECT_TYPE_CASE(VkValidationCacheEXT, VK_OBJECT_TYPE_VALIDATION_CACHE_EXT)
VK_OBJECT_TYPE_CASE(VkAccelerationStructureNV, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV)
VK_OBJECT_TYPE_CASE(VkPerformanceConfigurationINTEL, VK_OBJECT_TYPE_PERFORMANCE_CONFIGURATION_INTEL)
VK_OBJECT_TYPE_CASE(VkDeferredOperationKHR, VK_OBJECT_TYPE_DEFERRED_OPERATION_KHR)
VK_OBJECT_TYPE_CASE(VkIndirectCommandsLayoutNV, VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV)
VK_OBJECT_TYPE_CASE(VkCudaModuleNV, VK_OBJECT_TYPE_CUDA_MODULE_NV)
VK_OBJECT_TYPE_CASE(VkCudaFunctionNV, VK_OBJECT_TYPE_CUDA_FUNCTION_NV)
#ifdef VK_FUCHSIA_buffer_collection
VK_OBJECT_TYPE_CASE(VkBufferCollectionFUCHSIA, VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA)
#endif
VK_OBJECT_TYPE_CASE(VkMicromapEXT, VK_OBJECT_TYPE_MICROMAP_EXT)
VK_OBJECT_TYPE_CASE(VkOpticalFlowSessionNV, VK_OBJECT_TYPE_OPTICAL_FLOW_SESSION_NV)
VK_OBJECT_TYPE_CASE(VkShaderEXT, VK_OBJECT_TYPE_SHADER_EXT)

#ifndef VK_OBJECT_TYPES_HPP
#define VK_OBJECT_TYPES_HPP

namespace vk {
	template <typename T>
	struct VulkanObjectType;

	template <> struct VulkanObjectType<VkInstance> { constexpr static auto VALUE = VK_OBJECT_TYPE_INSTANCE; };
	template <> struct VulkanObjectType<VkPhysicalDevice> { constexpr static auto VALUE = VK_OBJECT_TYPE_PHYSICAL_DEVICE; };
	template <> struct VulkanObjectType<VkDevice> { constexpr static auto VALUE = VK_OBJECT_TYPE_DEVICE; };
	template <> struct VulkanObjectType<VkQueue> { constexpr static auto VALUE = VK_OBJECT_TYPE_QUEUE; };
	template <> struct VulkanObjectType<VkSemaphore> { constexpr static auto VALUE = VK_OBJECT_TYPE_SEMAPHORE; };
	template <> struct VulkanObjectType<VkCommandBuffer> { constexpr static auto VALUE = VK_OBJECT_TYPE_COMMAND_BUFFER; };
	template <> struct VulkanObjectType<VkFence> { constexpr static auto VALUE = VK_OBJECT_TYPE_FENCE; };
	template <> struct VulkanObjectType<VkDeviceMemory> { constexpr static auto VALUE = VK_OBJECT_TYPE_DEVICE_MEMORY; };
	template <> struct VulkanObjectType<VkBuffer> { constexpr static auto VALUE = VK_OBJECT_TYPE_BUFFER; };
	template <> struct VulkanObjectType<VkImage> { constexpr static auto VALUE = VK_OBJECT_TYPE_IMAGE; };
	template <> struct VulkanObjectType<VkEvent> { constexpr static auto VALUE = VK_OBJECT_TYPE_EVENT; };
	template <> struct VulkanObjectType<VkQueryPool> { constexpr static auto VALUE = VK_OBJECT_TYPE_QUERY_POOL; };
	template <> struct VulkanObjectType<VkBufferView> { constexpr static auto VALUE = VK_OBJECT_TYPE_BUFFER_VIEW; };
	template <> struct VulkanObjectType<VkImageView> { constexpr static auto VALUE = VK_OBJECT_TYPE_IMAGE_VIEW; };
	template <> struct VulkanObjectType<VkShaderModule> { constexpr static auto VALUE = VK_OBJECT_TYPE_SHADER_MODULE; };
	template <> struct VulkanObjectType<VkPipelineCache> { constexpr static auto VALUE = VK_OBJECT_TYPE_PIPELINE_CACHE; };
	template <> struct VulkanObjectType<VkPipelineLayout> { constexpr static auto VALUE = VK_OBJECT_TYPE_PIPELINE_LAYOUT; };
	template <> struct VulkanObjectType<VkRenderPass> { constexpr static auto VALUE = VK_OBJECT_TYPE_RENDER_PASS; };
	template <> struct VulkanObjectType<VkPipeline> { constexpr static auto VALUE = VK_OBJECT_TYPE_PIPELINE; };
	template <> struct VulkanObjectType<VkDescriptorSetLayout> { constexpr static auto VALUE = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT; };
	template <> struct VulkanObjectType<VkSampler> { constexpr static auto VALUE = VK_OBJECT_TYPE_SAMPLER; };
	template <> struct VulkanObjectType<VkDescriptorPool> { constexpr static auto VALUE = VK_OBJECT_TYPE_DESCRIPTOR_POOL; };
	template <> struct VulkanObjectType<VkDescriptorSet> { constexpr static auto VALUE = VK_OBJECT_TYPE_DESCRIPTOR_SET; };
	template <> struct VulkanObjectType<VkFramebuffer> { constexpr static auto VALUE = VK_OBJECT_TYPE_FRAMEBUFFER; };
	template <> struct VulkanObjectType<VkCommandPool> { constexpr static auto VALUE = VK_OBJECT_TYPE_COMMAND_POOL; };
	template <> struct VulkanObjectType<VkSamplerYcbcrConversion> { constexpr static auto VALUE = VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION; };
	template <> struct VulkanObjectType<VkDescriptorUpdateTemplate> { constexpr static auto VALUE = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE; };
	template <> struct VulkanObjectType<VkPrivateDataSlot> { constexpr static auto VALUE = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT; };

	template <> struct VulkanObjectType<VkSurfaceKHR> { constexpr static auto VALUE = VK_OBJECT_TYPE_SURFACE_KHR; };
	template <> struct VulkanObjectType<VkSwapchainKHR> { constexpr static auto VALUE = VK_OBJECT_TYPE_SWAPCHAIN_KHR; };
	template <> struct VulkanObjectType<VkDisplayKHR> { constexpr static auto VALUE = VK_OBJECT_TYPE_DISPLAY_KHR; };
	template <> struct VulkanObjectType<VkDisplayModeKHR> { constexpr static auto VALUE = VK_OBJECT_TYPE_DISPLAY_MODE_KHR; };
	template <> struct VulkanObjectType<VkDebugReportCallbackEXT> { constexpr static auto VALUE = VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT; };
	template <> struct VulkanObjectType<VkVideoSessionKHR> { constexpr static auto VALUE = VK_OBJECT_TYPE_VIDEO_SESSION_KHR; };
	template <> struct VulkanObjectType<VkVideoSessionParametersKHR> { constexpr static auto VALUE = VK_OBJECT_TYPE_VIDEO_SESSION_PARAMETERS_KHR; };
	template <> struct VulkanObjectType<VkCuModuleNVX> { constexpr static auto VALUE = VK_OBJECT_TYPE_CU_MODULE_NVX; };
	template <> struct VulkanObjectType<VkCuFunctionNVX> { constexpr static auto VALUE = VK_OBJECT_TYPE_CU_FUNCTION_NVX; };
	template <> struct VulkanObjectType<VkDebugUtilsMessengerEXT> { constexpr static auto VALUE = VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT; };
	template <> struct VulkanObjectType<VkAccelerationStructureKHR> { constexpr static auto VALUE = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR; };
	template <> struct VulkanObjectType<VkValidationCacheEXT> { constexpr static auto VALUE = VK_OBJECT_TYPE_VALIDATION_CACHE_EXT; };
	template <> struct VulkanObjectType<VkAccelerationStructureNV> { constexpr static auto VALUE = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV; };
	template <> struct VulkanObjectType<VkPerformanceConfigurationINTEL> { constexpr static auto VALUE = VK_OBJECT_TYPE_PERFORMANCE_CONFIGURATION_INTEL; };
	template <> struct VulkanObjectType<VkDeferredOperationKHR> { constexpr static auto VALUE = VK_OBJECT_TYPE_DEFERRED_OPERATION_KHR; };
	template <> struct VulkanObjectType<VkIndirectCommandsLayoutNV> { constexpr static auto VALUE = VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV; };
#ifdef VK_ENABLE_BETA_EXTENSIONS
	template <> struct VulkanObjectType<VkCudaModuleNV> { constexpr static auto VALUE = VK_OBJECT_TYPE_CUDA_MODULE_NV; };
	template <> struct VulkanObjectType<VkCudaFunctionNV> { constexpr static auto VALUE = VK_OBJECT_TYPE_CUDA_FUNCTION_NV; };
#endif
#ifdef VK_FUCHSIA_buffer_collection
	template <> struct VulkanObjectType<VkBufferCollectionFUCHSIA> { constexpr static auto VALUE = VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA; };
#endif
	template <> struct VulkanObjectType<VkMicromapEXT> { constexpr static auto VALUE = VK_OBJECT_TYPE_MICROMAP_EXT; };
	template <> struct VulkanObjectType<VkOpticalFlowSessionNV> { constexpr static auto VALUE = VK_OBJECT_TYPE_OPTICAL_FLOW_SESSION_NV; };
	template <> struct VulkanObjectType<VkShaderEXT> { constexpr static auto VALUE = VK_OBJECT_TYPE_SHADER_EXT; };

	template <typename T>
	constexpr auto VulkanObjectTypeV = VulkanObjectType<T>::VALUE;
}

#endif // VK_OBJECT_TYPES_HPP
