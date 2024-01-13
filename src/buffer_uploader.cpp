#include <vulkan/debug_utils.hpp>

#include <vk_gltf_viewer/util.hpp>
#include <vk_gltf_viewer/buffer_uploader.hpp>

bool BufferUploader::init(VkDevice nDevice, VmaAllocator nAllocator, VkQueue nTransferQueue, std::uint32_t nTransferQueueIndex) {
	device = nDevice;
	allocator = nAllocator;
	transferQueue = nTransferQueue;
	transferQueueIndex = nTransferQueueIndex;

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
		.commandBufferCount = 1,
	};
	auto allocateResult = vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer);
	vk::checkResult(allocateResult, "Failed to allocate buffer upload command buffers: {}");

	// Create the submit fence
	const VkFenceCreateInfo fenceCreateInfo {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT,
	};
	auto fenceResult = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
	vk::checkResult(fenceResult, "Failed to create buffer upload fence: {}");

	// Create the staging buffer
	const VmaAllocationCreateInfo allocationInfo {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
		.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	};
	const VkBufferCreateInfo bufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = maxStagingBufferSize,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	};

	auto result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationInfo,
								  &stagingBuffer, &stagingAllocation, VK_NULL_HANDLE);
	vk::checkResult(result, "Failed to allocate staging buffer: {}");
	vk::setDebugUtilsName(device, stagingBuffer, "Staging buffer");
	return true;
}

void BufferUploader::destroy() {
	vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
	vkDestroyFence(device, fence, VK_NULL_HANDLE);
	vkDestroyCommandPool(device, commandPool, VK_NULL_HANDLE);
}

bool BufferUploader::uploadToBuffer(std::span<const std::byte> data, VkBuffer destinationBuffer, VmaAllocation destinationAllocation) {
	for (auto i = 0; i < (data.size_bytes() + maxStagingBufferSize - 1) / maxStagingBufferSize; ++i) {
		// Get the subspan for this iteration.
		auto subLength = min(data.size_bytes() - i * maxStagingBufferSize, maxStagingBufferSize);
		auto sub = data.subspan(i * maxStagingBufferSize, subLength);

		// Copy the memory chunk into the staging buffer
		{
			vk::ScopedMap map(allocator, stagingAllocation);
			std::memcpy(map.get(), sub.data(), sub.size_bytes());
		}

		// We perform a partial copy with vkCmdCopyBuffer on the transfer queue.
		vkResetFences(device, 1, &fence);
		vkResetCommandBuffer(commandBuffer, 0);

		const VkCommandBufferBeginInfo beginInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		const VkBufferCopy region {
			.srcOffset = 0,
			.dstOffset = i * maxStagingBufferSize,
			.size = sub.size_bytes(),
		};
		vkCmdCopyBuffer(commandBuffer, stagingBuffer, destinationBuffer, 1, &region);

		vkEndCommandBuffer(commandBuffer);

		// Submit the command buffer
		const VkPipelineStageFlags submitWaitStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		const VkSubmitInfo submitInfo {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pWaitDstStageMask = &submitWaitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffer,
		};
		auto submitResult = vkQueueSubmit(transferQueue, 1, &submitInfo, fence);
		vk::checkResult(submitResult, "Failed to submit buffer copy: {}");

		vkWaitForFences(device, 1, &fence, VK_TRUE, 9999999999);
	}

	return true;
}
