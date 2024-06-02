#include <vulkan/vk.hpp>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/buffer.hpp>

ScopedBuffer::ScopedBuffer(const Device& _device, const VkBufferCreateInfo* bufferInfo,
						   const VmaAllocationCreateInfo* allocationInfo) : device(_device), bufferSize(bufferInfo->size) {
	ZoneScoped;
	vk::checkResult(vmaCreateBuffer(device.get().allocator, bufferInfo, allocationInfo,
									&handle, &allocation, &this->allocationInfo),
					"Failed to create buffer");

	if ((allocationInfo->flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) == VMA_ALLOCATION_CREATE_MAPPED_BIT)
		persistentlyMapped = true;
}

ScopedBuffer::~ScopedBuffer() noexcept {
	ZoneScoped;
	vmaDestroyBuffer(device.get().allocator, handle, allocation);
}

VkDeviceAddress ScopedBuffer::getDeviceAddress() const noexcept {
	ZoneScoped;
	const VkBufferDeviceAddressInfo info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = getHandle(),
	};
	return vkGetBufferDeviceAddress(device.get(), &info);
}
