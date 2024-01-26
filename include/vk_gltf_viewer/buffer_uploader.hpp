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
	enki::Dependency dependency;

	std::span<const std::byte> data;
	VkImage destinationImage;
	VkExtent3D imageExtent;
	VkImageLayout destinationLayout;

public:
	explicit ImageUploadTask(std::span<const std::byte> data, VkImage destinationImage, VkExtent3D imageExtent, VkImageLayout destinationLayout);

	void SetDependency(enki::ICompletable* task) {
		ITaskSet::SetDependency(dependency, task);
	}

	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override;
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

	struct CommandPool {
		VkCommandPool pool;
		// TODO: Find some mechanism to allow using multiple command buffers on one thread.
		//       Perhaps we can use up to N command buffers and fences for multiple submits from a single thread.
		VkCommandBuffer buffer;
	};

	std::vector<CommandPool> commandPools;
	std::vector<VkFence> fences;

	struct StagingBuffer {
		VkBuffer handle;
		VmaAllocation allocation;
	};

	std::vector<StagingBuffer> stagingBuffers;

	BufferUploader() = default;

	std::size_t stagingBufferSize = 0;

	TransferQueue& getNextQueueHandle() {
		// Generally it shouldn't matter if we don't guard the idx variable, as then we might just use the same queue twice in succession.
		// However, just to be completely correct we'll use this.
		static std::atomic<std::size_t> idx = 0;
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

	bool init(VkDevice device, VmaAllocator allocator, std::uint32_t transferQueueIndex, std::size_t transferQueueCount);
	void destroy();

	[[nodiscard]] std::unique_ptr<BufferUploadTask> uploadToBuffer(std::span<const std::byte> data, VkBuffer buffer);
};
