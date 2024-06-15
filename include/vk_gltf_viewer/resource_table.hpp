#pragma once

#include <mutex>
#include <utility>
#include <vector>

#include <vulkan/vk.hpp>

#include <resource_table.h.glsl>

struct Device;

class ResourceTable {
	std::reference_wrapper<Device> device;

	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkDescriptorSetLayout layout = VK_NULL_HANDLE;
	VkDescriptorSet set = VK_NULL_HANDLE;

	/** Each bit of each integer represents a boolean whether that array element is free or not */
	std::vector<std::uint64_t> sampledImageBitmap;
	std::vector<std::uint64_t> storageImageBitmap;
	std::mutex bitmapMutex;

	glsl::ResourceTableHandle findFirstFreeHandle(std::vector<std::uint64_t>& bitmap);
	void freeHandle(std::vector<std::uint64_t>& bitmap, glsl::ResourceTableHandle handle);

public:
	explicit ResourceTable(Device& _device);
	~ResourceTable() noexcept;

	[[nodiscard]] auto getLayout() const noexcept -> const VkDescriptorSetLayout& {
		return layout;
	}

	[[nodiscard]] auto getSet() const noexcept -> const VkDescriptorSet& {
		return set;
	}

	void removeStorageImageHandle(glsl::ResourceTableHandle handle) noexcept;
	void removeSampledImageHandle(glsl::ResourceTableHandle handle) noexcept;
	[[nodiscard]] glsl::ResourceTableHandle allocateStorageImage(VkImageView view, VkImageLayout imageLayout) noexcept;
	[[nodiscard]] glsl::ResourceTableHandle allocateSampledImage(VkImageView view, VkImageLayout imageLayout, VkSampler sampler) noexcept;
};
