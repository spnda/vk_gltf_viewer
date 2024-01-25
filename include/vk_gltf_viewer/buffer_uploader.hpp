#pragma once

#include <mutex>
#include <span>

#include <vulkan/vk.hpp>
#include <vulkan/vma.hpp>

#include <TaskScheduler.h>

class BufferUploadTask : public enki::ITaskSet {
	std::span<const std::byte> data;
	VkBuffer destinationBuffer;

public:
	explicit BufferUploadTask(std::span<const std::byte> data, VkBuffer destinationBuffer);

	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

class ImageUploadTask : public enki::ITaskSet {
	std::span<const std::byte> data;
	VkImage destinationImage;
	VkExtent3D imageExtent;
	VkImageLayout destinationLayout;

	enki::Dependency dependency;

public:
	explicit ImageUploadTask(std::span<const std::byte> data, VkImage destinationImage, VkExtent3D imageExtent, VkImageLayout destinationLayout);

	void SetDependency(enki::ITaskSet* task) {
		ITaskSet::SetDependency(dependency, task);
	}

	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override;

	std::span<const std::byte> getData() {
		return data;
	}
};

/** Simple class that contains functions to copy any buffer into DEVICE_LOCAL memory through staging buffers */
class BufferUploader {
	friend class BufferUploadTask;
	friend class ImageUploadTask;

	VkDevice device = VK_NULL_HANDLE;
	VmaAllocator allocator = VK_NULL_HANDLE;

	struct TransferQueue {
		VkQueue handle;
		std::unique_ptr<std::mutex> lock; // Can't hold the object in a vector otherwise.
	};

	std::uint32_t transferQueueIndex = VK_QUEUE_FAMILY_IGNORED;
	std::vector<TransferQueue> transferQueues;

	VkCommandPool commandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkFence> fences;

	struct StagingBuffer {
		VkBuffer handle;
		VmaAllocation allocation;
	};

	std::vector<StagingBuffer> stagingBuffers;

	BufferUploader() = default;

	std::size_t stagingBufferSize = 0;

	TransferQueue& getNextQueueHandle() {
		static std::size_t idx = 0;
		return transferQueues[idx++ % transferQueues.size()];
	}

public:
	static BufferUploader& getInstance() {
		static BufferUploader uploader;
		return uploader;
	}

	[[nodiscard]] std::size_t getStagingBufferSize() const {
		return stagingBufferSize;
	}

	bool init(VkDevice device, VmaAllocator allocator, std::uint32_t transferQueueIndex);
	void destroy();

	[[nodiscard]] std::unique_ptr<BufferUploadTask> uploadToBuffer(std::span<const std::byte> data, VkBuffer buffer, enki::TaskScheduler& taskScheduler);
};
