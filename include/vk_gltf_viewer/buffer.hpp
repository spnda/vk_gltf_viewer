#pragma once

#include <vulkan/vk.hpp>
#include <vulkan/vma.hpp>

#include <vk_gltf_viewer/device.hpp>

class ScopedBuffer {
	template <typename T> friend class ScopedMap;

	std::reference_wrapper<const Device> device;

	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkDeviceSize bufferSize = 0;

	VmaAllocationInfo allocationInfo;
	bool persistentlyMapped = false;

public:
	ScopedBuffer(const Device& _device, const VkBufferCreateInfo* bufferInfo, const VmaAllocationCreateInfo* allocationInfo);
	~ScopedBuffer() noexcept;

	ScopedBuffer(const ScopedBuffer& other) = delete;
	ScopedBuffer& operator=(const ScopedBuffer& other) = delete;

	[[nodiscard]] auto getHandle() const noexcept {
		return handle;
	}
	[[nodiscard]] auto getAllocation() const noexcept {
		return allocation;
	}
	[[nodiscard]] auto getAllocationInfo() const noexcept {
		return allocationInfo;
	}
	[[nodiscard]] VkDeviceAddress getDeviceAddress() const noexcept;
	[[nodiscard]] VkDeviceSize getBufferSize() const noexcept {
		return bufferSize;
	}

	operator VkBuffer() const noexcept {
		return getHandle();
	}
};

template<typename T = void>
class ScopedMap {
	const ScopedBuffer& buffer;
	void* data;

public:
	explicit ScopedMap(const ScopedBuffer& buffer) : buffer(buffer) {
		ZoneScoped;
		if (buffer.persistentlyMapped) {
			data = buffer.allocationInfo.pMappedData;
		} else {
			vmaMapMemory(buffer.device.get().allocator, buffer.allocation, &data);
		}
	}

	T *get() {
		return static_cast<T*>(data);
	}

	~ScopedMap() {
		ZoneScoped;
		if (!buffer.persistentlyMapped) {
			vmaUnmapMemory(buffer.device.get().allocator, buffer.allocation);
		}
	}
};
