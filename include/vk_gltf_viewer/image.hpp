#pragma once

#include <utility>

#include <vulkan/vk.hpp>
#include <vulkan/vma.hpp>

#include <vk_gltf_viewer/device.hpp>

class ScopedImage {
	std::reference_wrapper<const Device> device;
	VkImage handle = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkExtent3D extent;

	VkImageView defaultView = VK_NULL_HANDLE;

public:
	explicit ScopedImage(const Device& _device, const VkImageCreateInfo* imageInfo, const VmaAllocationCreateInfo* allocationInfo);
	~ScopedImage() noexcept;

	[[nodiscard]] auto getHandle() const noexcept {
		return handle;
	}
	[[nodiscard]] auto getAllocation() const noexcept {
		return allocation;
	}
	[[nodiscard]] auto getDefaultView() const noexcept {
		return defaultView;
	}
	[[nodiscard]] auto getExtent() const noexcept {
		return extent;
	}

	operator VkImage() const noexcept {
		return getHandle();
	}
};

class ScopedImageView {
	std::reference_wrapper<const Device> device;
	VkImageView handle = VK_NULL_HANDLE;

public:
	explicit ScopedImageView(const Device& _device, const VkImageViewCreateInfo* imageViewInfo);
	~ScopedImageView() noexcept;

	[[nodiscard]] auto getHandle() const noexcept {
		return handle;
	}

	operator VkImageView() const noexcept {
		return getHandle();
	}
};
