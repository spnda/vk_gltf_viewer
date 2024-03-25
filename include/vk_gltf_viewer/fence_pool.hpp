#pragma once

#include <deque>
#include <mutex>

#include <vulkan/vk.hpp>

struct Fence {
	VkDevice device = VK_NULL_HANDLE;
	VkFence handle = VK_NULL_HANDLE;

	Fence(VkDevice nDevice, VkFenceCreateFlags flags) {
		device = nDevice;
		const VkFenceCreateInfo fenceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = flags,
		};
		auto result = vkCreateFence(device, &fenceCreateInfo, nullptr, &handle);
		vk::checkResult(result, "Failed to create fence");
	}
	~Fence() {
		 vkDestroyFence(device, handle, nullptr);
	}
};

/** A pool for ref-counted fences */
class FencePool {
	VkDevice device;

	std::deque<std::shared_ptr<Fence>> availableFences;
	std::mutex fenceMutex;

public:
	void init(VkDevice nDevice) {
		device = nDevice;
	}

	void destroy() {
		for (auto& fence : availableFences) {
			fence.reset();
		}
	}

	std::shared_ptr<Fence> acquire() {
		std::lock_guard lock(fenceMutex);
		if (availableFences.empty()) {
			return std::make_shared<Fence>(device, VK_FENCE_CREATE_SIGNALED_BIT);
		} else {
			auto fence = availableFences.front();
			availableFences.pop_front();
			return fence;
		}
	}

	void free(std::shared_ptr<Fence>& fence) {
		std::lock_guard lock(fenceMutex);
		availableFences.emplace_back(std::move(fence));
	}

	void wait_and_free(std::shared_ptr<Fence>& fence) {
		vkWaitForFences(device, 1, &fence->handle, VK_TRUE, 9999999999);
		free(fence);
	}
};
