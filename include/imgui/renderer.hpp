#pragma once

#include <vector>

#include <ui/ui.h.glsl>
#include <resource_table.h.glsl>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <imgui.h>
#include <TaskScheduler.h>

#include <vulkan/vk.hpp>
#include <vulkan/vma.hpp>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/image.hpp>
#include <vk_gltf_viewer/buffer.hpp>

namespace imgui {
	struct PerFrameBuffers {
		std::unique_ptr<ScopedBuffer> vertexBuffer;
		VkDeviceAddress vertexBufferAddress = 0;

		std::unique_ptr<ScopedBuffer> indexBuffer;
	};

	class Renderer final {
		friend class ShaderLoadTask;

		std::reference_wrapper<Device> device;

		std::vector<PerFrameBuffers> buffers;

		VkBuffer fontAtlasStagingBuffer = VK_NULL_HANDLE;
		VmaAllocation fontAtlasStagingAllocation = VK_NULL_HANDLE;

		std::unique_ptr<ScopedImage> fontAtlas;
		VkImageView fontAtlasView = VK_NULL_HANDLE;
		VkSampler fontAtlasSampler = VK_NULL_HANDLE;
		glm::u32vec2 fontAtlasExtent = {};
		glsl::ResourceTableHandle fontAtlasHandle = glsl::invalidHandle;

		VkPipeline pipeline = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipelineCache pipelineCache = VK_NULL_HANDLE;

		void createGeometryBuffers(std::size_t index, VkDeviceSize vertexSize, VkDeviceSize indexSize);

	public:
		explicit Renderer(Device& device, GLFWwindow* window, VkFormat swapchainImageFormat);
		~Renderer();

		/**
		 * Creates the sampler, texture, and shader parameter for the font atlas. If the font data
		 * changes, this will upload the data again and resize the texture if necessary. This will
		 * also update the descriptor set, meaning this cannot be called while any frame is still
		 * in flight.
		 */
		void createFontAtlas();
		void draw(VkCommandBuffer commandBuffer, VkImageView swapchainImageView, glm::u32vec2 framebufferSize, std::size_t currentFrame);
		auto initFrameData(std::uint32_t frameCount) -> VkResult;
		void newFrame();
	};
} // namespace imgui
