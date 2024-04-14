#pragma once

#include <queue>
#include <mutex>

#include <vulkan/vk.hpp>

#include <tracy/Tracy.hpp>

struct Fence {
	VkDevice device = VK_NULL_HANDLE;
	VkFence handle = VK_NULL_HANDLE;

	explicit Fence(VkDevice nDevice, VkFenceCreateFlags flags = 0) {
		ZoneScoped;
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

	[[nodiscard]] VkResult wait(std::uint64_t timeout = std::numeric_limits<std::uint64_t>::max()) noexcept {
		ZoneScoped;
		return vkWaitForFences(device, 1, &handle, VK_TRUE, timeout);
	}
};

/** A pool for ref-counted fences */
class FencePool {
	VkDevice device = VK_NULL_HANDLE;

	std::queue<std::shared_ptr<Fence>> availableFences;
	std::mutex availabilityMutex;

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
		assert(device != VK_NULL_HANDLE);
		std::lock_guard lock(availabilityMutex);
		if (availableFences.empty()) {
			return std::make_shared<Fence>(device);
		}

		auto fence = availableFences.front();
		availableFences.pop();
		return fence;
	}

	/** This will reset the fence and the passed shared_ptr, as this is a reference. Only call this after having waited on the fence */
	void free(std::shared_ptr<Fence>& fence) {
		ZoneScoped;
		vk::checkResult(vkResetFences(device, 1, &fence->handle), "Failed to reset fence");
		std::lock_guard lock(availabilityMutex);
		availableFences.emplace(std::move(fence));
	}

	void wait_and_free(std::shared_ptr<Fence>& fence) {
		vk::checkResult(fence->wait(), "Failed to wait for fence");
		free(fence);
	}
};

struct Semaphore {
	VkDevice device = VK_NULL_HANDLE;
	VkSemaphore handle = VK_NULL_HANDLE;

	explicit Semaphore(VkDevice nDevice) {
		ZoneScoped;
		device = nDevice;
		const VkSemaphoreCreateInfo semaphoreInfo {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
		auto result = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &handle);
		vk::checkResult(result, "Failed to create semaphore");
	}
	~Semaphore() noexcept {
		 vkDestroySemaphore(device, handle, nullptr);
	}
};

/** A pool for ref-counted semaphores */
class SemaphorePool {
	VkDevice device = VK_NULL_HANDLE;

	std::queue<std::shared_ptr<Semaphore>> availableSemaphores;
	std::mutex availabilityMutex;

public:
	void init(VkDevice nDevice) noexcept {
		device = nDevice;
	}

	void destroy() noexcept {
		for (; !availableSemaphores.empty(); availableSemaphores.pop()) {
			availableSemaphores.front().reset();
		}
	}

	[[nodiscard]] std::shared_ptr<Semaphore> acquire() {
		assert(device != VK_NULL_HANDLE);
		std::lock_guard lock(availabilityMutex);

		if (availableSemaphores.empty()) {
			return std::make_shared<Semaphore>(device);
		}

		auto semaphore = availableSemaphores.front();
		availableSemaphores.pop();
		return semaphore;
	}

	void free(std::shared_ptr<Semaphore>& semaphore) {
		std::lock_guard lock(availabilityMutex);
		availableSemaphores.emplace(std::move(semaphore));
	}
};
