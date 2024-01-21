#include <tracy/Tracy.hpp>

#include <vulkan/debug_utils.hpp>

#include <vk_gltf_viewer/util.hpp>
#include <vk_gltf_viewer/buffer_uploader.hpp>

BufferUploadTask::BufferUploadTask(std::span<const std::byte> data, VkBuffer destinationBuffer) : data(data), destinationBuffer(destinationBuffer) {
	// This is required so that every task's range has this size to fit with the staging buffers.
	auto& uploader = BufferUploader::getInstance();
	m_SetSize = (data.size_bytes() + uploader.getStagingBufferSize() - 1) / uploader.getStagingBufferSize();
}

void BufferUploadTask::ExecuteRange(enki::TaskSetPartition range, uint32_t threadnum) {
	assert(!BufferUploader::getInstance().stagingBuffers.empty());
	ZoneScoped;
	for (auto i = range.start; i < range.end; ++i) {
		auto &uploader = BufferUploader::getInstance();
		auto stagingBufferSize = uploader.getStagingBufferSize();

		// Get the subspan for this execution
		auto subLength = min(data.size_bytes() - i * stagingBufferSize, stagingBufferSize);
		auto sub = data.subspan(i * stagingBufferSize, subLength);

		auto &stagingBuffer = uploader.stagingBuffers[threadnum];
		// Copy the memory chunk into the staging buffer
		{
			vk::ScopedMap map(uploader.allocator, stagingBuffer.allocation);
			std::memcpy(map.get(), sub.data(), sub.size_bytes());
		}

		auto cmd = uploader.commandBuffers[threadnum];

		// We perform a partial copy with vkCmdCopyBuffer on the transfer queue.
		auto fence = uploader.fences[threadnum];
		vkResetFences(uploader.device, 1, &fence);
		vkResetCommandBuffer(cmd, 0);

		const VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		vkBeginCommandBuffer(cmd, &beginInfo);

		const VkBufferCopy region{
			.srcOffset = 0,
			.dstOffset = i * stagingBufferSize,
			.size = sub.size_bytes(),
		};
		vkCmdCopyBuffer(cmd, stagingBuffer.handle, destinationBuffer, 1, &region);

		vkEndCommandBuffer(cmd);

		{
			auto& queue = uploader.getNextQueueHandle();

			// We need to guard the vkQueueSubmit call
			std::lock_guard lock(*queue.lock);

			// Submit the command buffer
			const VkPipelineStageFlags submitWaitStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			const VkSubmitInfo submitInfo{
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.pWaitDstStageMask = &submitWaitStages,
				.commandBufferCount = 1,
				.pCommandBuffers = &cmd,
			};
			auto submitResult = vkQueueSubmit(queue.handle, 1, &submitInfo, fence);
			vk::checkResult(submitResult, "Failed to submit buffer copy: {}");
		}

		vkWaitForFences(uploader.device, 1, &fence, VK_TRUE, 9999999999);
	}
}

bool BufferUploader::init(VkDevice nDevice, VmaAllocator nAllocator, std::uint32_t nTransferQueueIndex) {
	ZoneScoped;
	device = nDevice;
	allocator = nAllocator;
	transferQueueIndex = nTransferQueueIndex;

	transferQueues.resize(1);// TODO: Get queue count
	for (std::size_t i = 0; auto& transferQueue : transferQueues) {
		transferQueue.lock = std::make_unique<std::mutex>();
		vkGetDeviceQueue(device, transferQueueIndex, i++, &transferQueue.handle);
	}

	auto threadCount = std::thread::hardware_concurrency();

	// Set the staging buffer size. We only want to use 80% of the DEVICE_LOCAL | HOST_VISIBLE memory.
	// TODO: Use actual Vulkan heap size.
	std::size_t totalSize = alignDown(static_cast<std::size_t>(224395264U * 0.5), static_cast<std::size_t>(threadCount));
	stagingBufferSize = totalSize / threadCount;

	// Create the command pool
	const VkCommandPoolCreateInfo commandPoolInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = transferQueueIndex,
	};
	auto createResult = vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool);
	vk::checkResult(createResult, "Failed to allocate buffer upload command pool: {}");

	const VkCommandBufferAllocateInfo allocateInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = threadCount,
	};
	commandBuffers.resize(allocateInfo.commandBufferCount);
	auto allocateResult = vkAllocateCommandBuffers(device, &allocateInfo, commandBuffers.data());
	vk::checkResult(allocateResult, "Failed to allocate buffer upload command buffers: {}");

	fences.resize(threadCount);
	for (auto& fence : fences) {
		// Create the submit fence
		const VkFenceCreateInfo fenceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		auto fenceResult = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
		vk::checkResult(fenceResult, "Failed to create buffer upload fence: {}");
	}

	stagingBuffers.resize(std::thread::hardware_concurrency());
	for (auto& stagingBuffer : stagingBuffers) {
		// Create the staging buffer
		const VmaAllocationCreateInfo allocationInfo{
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
							 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		};
		const VkBufferCreateInfo bufferCreateInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = stagingBufferSize,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		};

		auto result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationInfo,
									  &stagingBuffer.handle, &stagingBuffer.allocation, VK_NULL_HANDLE);
		vk::checkResult(result, "Failed to allocate staging buffer: {}");
		vk::setDebugUtilsName(device, stagingBuffer.handle, "Staging buffer {}");
	}
	return true;
}

void BufferUploader::destroy() {
	for (auto& stagingBuffer : stagingBuffers) {
		vmaDestroyBuffer(allocator, stagingBuffer.handle, stagingBuffer.allocation);
	}
	for (auto& fence : fences) {
		vkDestroyFence(device, fence, VK_NULL_HANDLE);
	}
	vkDestroyCommandPool(device, commandPool, VK_NULL_HANDLE);
}

std::unique_ptr<BufferUploadTask> BufferUploader::uploadToBuffer(std::span<const std::byte> data, VkBuffer buffer, enki::TaskScheduler& taskScheduler) {
	auto task = std::make_unique<BufferUploadTask>(data, buffer);
	taskScheduler.AddTaskSetToPipe(task.get());
	return task;
}
