#pragma once

#include <cstdint>
#include <queue>
#include <span>

#include <vulkan/vk.hpp>

namespace vk {
	/** A command pool wrapper */
	class CommandPool {
		VkCommandPool pool = VK_NULL_HANDLE;
		std::uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED;
		VkDevice device = VK_NULL_HANDLE;

		std::queue<VkCommandBuffer> availableCommandBuffers;

	public:
		void create(VkDevice nDevice, std::uint32_t queueFamilyIndex) {
			device = nDevice;
			queueFamily = queueFamilyIndex;

			const VkCommandPoolCreateInfo commandPoolInfo {
				.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
				.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
				.queueFamilyIndex = queueFamilyIndex,
			};
			auto createResult = vkCreateCommandPool(device, &commandPoolInfo, nullptr, &pool);
			vk::checkResult(createResult, "Failed to allocate command pool: {}");
		}

		void destroy() noexcept {
			vkDestroyCommandPool(device, pool, nullptr);
		}

		[[nodiscard]] VkCommandPool handle() const noexcept {
			return pool;
		}

		[[nodiscard]] std::uint32_t queue_family() const noexcept {
			return queueFamily;
		}

		[[nodiscard]] VkCommandBuffer allocate() {
			VkCommandBuffer handle;
			allocate(handle);
			return handle;
		}

		void allocate(VkCommandBuffer& handle) {
			if (!availableCommandBuffers.empty()) {
				handle = availableCommandBuffers.front();
				availableCommandBuffers.pop();
				return;
			}

			const VkCommandBufferAllocateInfo allocateInfo{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};
			auto allocateResult = vkAllocateCommandBuffers(device, &allocateInfo, &handle);
			vk::checkResult(allocateResult, "Failed to allocate command buffer: {}");
		}

		void allocate(std::span<VkCommandBuffer> handles) {
			if (availableCommandBuffers.size() >= handles.size()) {
				for (auto& handle: handles) {
					handle = availableCommandBuffers.front();
					availableCommandBuffers.pop();
				}
				return;
			}

			const VkCommandBufferAllocateInfo allocateInfo{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = static_cast<std::uint32_t>(handles.size()),
			};
			auto allocateResult = vkAllocateCommandBuffers(device, &allocateInfo, handles.data());
			vk::checkResult(allocateResult, "Failed to allocate command buffer: {}");
		}

		/** Makes the command buffer available again, and resets it */
		void reset_and_free(VkCommandBuffer handle) {
			vkResetCommandBuffer(handle, 0);
			availableCommandBuffers.emplace(handle);
		}
	};
} // namespace vk
