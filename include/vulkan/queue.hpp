#pragma once

#include <mutex>
#include <span>
#include <utility>

#include <tracy/Tracy.hpp>

#include <vulkan/vk.hpp>

namespace vk {
	struct Queue {
		VkQueue handle = VK_NULL_HANDLE;
		std::unique_ptr<std::mutex> lock; // Can't hold the object in a vector otherwise.

		VkResult submit(const VkSubmitInfo2& submit, VkFence fence) const {
			ZoneScoped;
			std::lock_guard guard(*lock);
			return vkQueueSubmit2(handle, 1, &submit, fence);
		}

		VkResult submit(std::span<const VkSubmitInfo2> submits, VkFence fence) const {
			ZoneScoped;
			std::lock_guard guard(*lock);
			return vkQueueSubmit2(handle, submits.size(), submits.data(), fence);
		}

		VkResult present(const VkPresentInfoKHR& presentInfo) const {
			ZoneScoped;
			std::lock_guard guard(*lock);
			return vkQueuePresentKHR(handle, &presentInfo);
		}

		operator VkQueue() const noexcept {
			return handle;
		}
	};
}
