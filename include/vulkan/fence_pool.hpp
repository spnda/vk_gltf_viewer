#pragma once

#include <queue>
#include <mutex>

#include <vulkan/vk.hpp>

struct Fence {
	VkDevice device = VK_NULL_HANDLE;
	VkFence handle = VK_NULL_HANDLE;

	explicit Fence(VkDevice nDevice, VkFenceCreateFlags flags = 0) {
		device = nDevice;
		const VkFenceCreateInfo fenceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = flags,
		};
		auto result = vkCreateFence(device, &fenceCreateInfo, nullptr, &handle);
		vk::checkResult(result, "Failed to create fence");
	}
	~Fence() noexcept {
		 vkDestroyFence(device, handle, nullptr);
	}
};

/** A pool for ref-counted fences */
class FencePool {
	VkDevice device;

	std::queue<std::shared_ptr<Fence>> availableFences;
	std::mutex fenceMutex;

public:
	void init(VkDevice nDevice) noexcept {
		device = nDevice;
	}

	void destroy() noexcept {
		for (; !availableFences.empty(); availableFences.pop()) {
			availableFences.front().reset();
		}
	}

	[[nodiscard]] std::shared_ptr<Fence> acquire() {
		std::lock_guard lock(fenceMutex);
		if (availableFences.empty()) {
			return std::make_shared<Fence>(device);
		}

		auto fence = availableFences.front();
		availableFences.pop();
		return fence;
	}

	/** This will reset the fence and the passed shared_ptr, as this is a reference. Only call this after having waited on the fence */
	void free(std::shared_ptr<Fence>& fence) {
		vk::checkResult(vkResetFences(device, 1, &fence->handle), "Failed to reset fence");
		std::lock_guard lock(fenceMutex);
		availableFences.emplace(std::move(fence));
	}

	void wait_and_free(std::shared_ptr<Fence>& fence) {
		vk::checkResult(vkWaitForFences(device, 1, &fence->handle, VK_TRUE, 9999999999), "Failed to wait for fence");
		free(fence);
	}
};
