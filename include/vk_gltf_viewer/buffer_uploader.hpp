#pragma once

#include <span>

#include <vulkan/vk.hpp>
#include <vulkan/vma.hpp>

/** Simple class that contains functions to copy any buffer into DEVICE_LOCAL memory through staging buffers */
class BufferUploader {
	VkDevice device;
	VmaAllocator allocator;

	std::uint32_t transferQueueIndex;
	VkQueue transferQueue;

	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
	VkFence fence;

	VkBuffer stagingBuffer;
	VmaAllocation stagingAllocation;

	BufferUploader() = default;

	// Any better value than this? I don't want to flood the DEVICE_LOCAL and HOST_COHERENT memory as there's other uses for it.
	// But I also don't want tiny staging buffers, so 16MB (2^24) it is for now?
	static constexpr std::size_t maxStagingBufferSize = 16777216;

public:
	static BufferUploader& getInstance() {
		static BufferUploader uploader;
		return uploader;
	}

	bool init(VkDevice device, VmaAllocator allocator, VkQueue transferQueue, std::uint32_t transferQueueIndex);
	void destroy();

	bool uploadToBuffer(std::span<const std::byte> data, VkBuffer buffer, VmaAllocation allocation);
};
