#pragma once

#include <vector>

#include <glm/glm.hpp>
#include <imgui.h>
#include <TaskScheduler.h>

#include <vulkan/vk.hpp>
#include <vulkan/vma.hpp>

#include <glfw/glfw3.h>

struct Viewer;

namespace imgui {
	struct PushConstants {
		glm::fvec2 scale = {};
		glm::fvec2 translate = {};
		VkDeviceAddress vertexBufferAddress = 0;
	};

	struct PerFrameBuffers {
		VkBuffer vertexBuffer = VK_NULL_HANDLE;
		VmaAllocation vertexAllocation = VK_NULL_HANDLE;
		VkDeviceSize vertexBufferSize = 0;
		VkDeviceAddress vertexBufferAddress = 0;

		VkBuffer indexBuffer = VK_NULL_HANDLE;
		VmaAllocation indexAllocation = VK_NULL_HANDLE;
		VkDeviceSize indexBufferSize = 0;
	};

	class Renderer final {
		friend class ShaderLoadTask;

		Viewer* viewer = nullptr;

		PushConstants pushConstants = {};
		std::vector<PerFrameBuffers> buffers;

		VkBuffer fontAtlasStagingBuffer = VK_NULL_HANDLE;
		VmaAllocation fontAtlasStagingAllocation = VK_NULL_HANDLE;

		VkImage fontAtlas = VK_NULL_HANDLE;
		VmaAllocation fontAtlasAllocation = VK_NULL_HANDLE;
		VkImageView fontAtlasView = VK_NULL_HANDLE;
		VkSampler fontAtlasSampler = VK_NULL_HANDLE;
		glm::u32vec2 fontAtlasExtent = {};

		VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
		VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
		VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipelineCache pipelineCache = VK_NULL_HANDLE;
		VkShaderModule fragmentShader = VK_NULL_HANDLE;
		VkShaderModule vertexShader = VK_NULL_HANDLE;

		VkResult createGeometryBuffers(std::size_t index, VkDeviceSize vertexSize, VkDeviceSize indexSize);

	public:
		/**
		 * Creates the sampler, texture, and shader parameter for the font atlas. If the font data
		 * changes, this will upload the data again and resize the texture if necessary. This will
		 * also update the descriptor set, meaning this cannot be called while any frame is still
		 * in flight.
		 */
		void createFontAtlas();
		void destroy();
		void draw(VkCommandBuffer commandBuffer, VkImageView swapchainImageView, glm::u32vec2 framebufferSize, std::size_t currentFrame);
		auto init(Viewer* viewer) -> VkResult;
		auto initFrameData(std::uint32_t frameCount) -> VkResult;
		void newFrame();
	};
} // namespace imgui
