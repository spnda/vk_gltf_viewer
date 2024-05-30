#pragma once

#include <vulkan/vk.hpp>
#include <vulkan/vma.hpp>
#include <VkBootstrap.h>

#include <tracy/TracyVulkan.hpp>

#include <vulkan/queue.hpp>
#include <vulkan/command_pool.hpp>
#include <vulkan/sync_pools.hpp>

#include <vk_gltf_viewer/deletion_queue.hpp>
#include <vk_gltf_viewer/resource_table.hpp>
#if defined(VKV_NV_AFTERMATH)
#include <vk_gltf_viewer/nvidia/aftermath.hpp>
#endif

class ScopedBuffer;

struct Instance {
	vkb::Instance instance;

	explicit Instance();
	~Instance() noexcept;

	[[nodiscard]] operator VkInstance() const noexcept {
		return VkInstance(instance);
	}
};

struct Device {
	vkb::PhysicalDevice physicalDevice;
	vkb::Device device;
	VmaAllocator allocator = VK_NULL_HANDLE;

	VkPhysicalDeviceVulkan12Properties vulkan12Properties = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES
	};

	TracyVkCtx tracyCtx = nullptr;

#if defined(VKV_NV_AFTERMATH)
	std::unique_ptr<AftermathCrashTracker> aftermathCrashTracker;
#endif

	/** A timeline deletion queue for the frame rendering */
	std::unique_ptr<TimelineDeletionQueue> timelineDeletionQueue;

	std::unique_ptr<ResourceTable> resourceTable;

	/** The main graphics & present queue */
	std::uint32_t graphicsQueueFamily = VK_QUEUE_FAMILY_IGNORED;
	vk::Queue graphicsQueue;

	// TODO: Should the transfer queue stuff be kept as part of Device?

	/** Dedicated transfer queues */
	std::uint32_t transferQueueFamily = VK_QUEUE_FAMILY_IGNORED;
	std::vector<vk::Queue> transferQueues;

	/** A command pool for each thread for the dedicated transfer queues */
	std::vector<vk::CommandPool> uploadCommandPools;

	/** A global pool of fences */
	vk::FencePool globalFencePool;

	explicit Device(const Instance& instance, VkSurfaceKHR surface);
	~Device() noexcept;

	[[nodiscard]] decltype(auto) getNextTransferQueueHandle() {
		static std::atomic<std::size_t> idx = 0;
		return transferQueues[idx++ % transferQueues.size()];
	}

	[[nodiscard]] std::unique_ptr<ScopedBuffer> createHostStagingBuffer(std::size_t byteSize) const noexcept;

	void immediateSubmit(vk::Queue& queue, vk::CommandPool& commandPool, std::function<void(VkCommandBuffer)> commands);

	[[nodiscard]] operator VkDevice() const noexcept {
		return VkDevice(device);
	}
};
