#include <vk_gltf_viewer/resource_table.hpp>
#include <vk_gltf_viewer/device.hpp>

#include <fastgltf/util.hpp>

ResourceTable::ResourceTable(Device& _device) : device(_device) {
	ZoneScoped;
	auto& properties = device.get().vulkan12Properties;
	const std::array<VkDescriptorPoolSize, 2> sizes {{
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = properties.maxDescriptorSetUpdateAfterBindSampledImages,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = properties.maxDescriptorSetUpdateAfterBindStorageImages,
		},
	}};
	const VkDescriptorPoolCreateInfo poolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
		.maxSets = 1,
		.poolSizeCount = static_cast<std::uint32_t>(sizes.size()),
		.pPoolSizes = sizes.data(),
	};
	vk::checkResult(vkCreateDescriptorPool(device.get(), &poolCreateInfo, vk::allocationCallbacks.get(), &pool),
					"Failed to create descriptor pool");

	// This needs to match what resource_table.h.glsl has.
	const std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings {{
		{
			.binding = glsl::sampledImageBinding,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = properties.maxDescriptorSetUpdateAfterBindSampledImages,
			.stageFlags = VK_SHADER_STAGE_ALL,
		},
		{
			.binding = glsl::storageImageBinding,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = properties.maxDescriptorSetUpdateAfterBindStorageImages,
			.stageFlags = VK_SHADER_STAGE_ALL,
		},
	}};
	constexpr std::array<VkDescriptorBindingFlags, layoutBindings.size()> layoutBindingFlags {{
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
	}};
	const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		.bindingCount = static_cast<std::uint32_t>(layoutBindingFlags.size()),
		.pBindingFlags = layoutBindingFlags.data(),
	};
	const VkDescriptorSetLayoutCreateInfo layoutCreateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = &bindingFlagsInfo,
		.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		.bindingCount = static_cast<std::uint32_t>(layoutBindings.size()),
		.pBindings = layoutBindings.data(),
	};
	vk::checkResult(vkCreateDescriptorSetLayout(device.get(), &layoutCreateInfo, vk::allocationCallbacks.get(), &layout),
					"Failed to create descriptor set layout");

	const VkDescriptorSetAllocateInfo allocateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &layout,
	};
	vk::checkResult(vkAllocateDescriptorSets(device.get(), &allocateInfo, &set), "Failed to allocate descriptor set");

	sampledImageBitmap.resize(fastgltf::alignUp(properties.maxDescriptorSetUpdateAfterBindSampledImages, 64) / 64);
	storageImageBitmap.resize(fastgltf::alignUp(properties.maxDescriptorSetUpdateAfterBindStorageImages, 64) / 64);
}

ResourceTable::~ResourceTable() noexcept {
	ZoneScoped;
	vkDestroyDescriptorSetLayout(device.get(), layout, vk::allocationCallbacks.get());
	vkDestroyDescriptorPool(device.get(), pool, vk::allocationCallbacks.get());
}

glsl::ResourceTableHandle ResourceTable::findFirstFreeHandle(std::vector<std::uint64_t>& bitmap) {
	ZoneScoped;
	std::lock_guard lock(bitmapMutex);
	for (std::size_t i = 0; i < bitmap.size(); ++i) {
		auto& value = bitmap[i];
		if (value == ~std::uint64_t(0U))
			continue;

		if (value == 0U) {
			value = 1U;
			return i * 64U;
		}

		auto n = std::countr_one(value);
		value |= (std::uint64_t(1U) << n);
		return i * 64U + n;
	}

	throw std::runtime_error("Failed to find free handle");
}

void ResourceTable::freeHandle(std::vector<std::uint64_t>& bitmap, glsl::ResourceTableHandle handle) {
	std::lock_guard lock(bitmapMutex);
	auto i = handle / 64;
	bitmap[i] &= ~(std::uint64_t(1U) << (handle % 64));
}

void ResourceTable::removeStorageImageHandle(glsl::ResourceTableHandle handle) noexcept {
	ZoneScoped;
	if (handle == glsl::invalidHandle)
		return;
	freeHandle(storageImageBitmap, handle);
}

void ResourceTable::removeSampledImageHandle(glsl::ResourceTableHandle handle) noexcept {
	ZoneScoped;
	if (handle == glsl::invalidHandle)
		return;
	freeHandle(sampledImageBitmap, handle);
}

glsl::ResourceTableHandle ResourceTable::allocateStorageImage(VkImageView view, VkImageLayout imageLayout) noexcept {
	ZoneScoped;
	auto handle = findFirstFreeHandle(storageImageBitmap);

	const VkDescriptorImageInfo imageInfo {
		.imageView = view,
		.imageLayout = imageLayout,
	};
	const VkWriteDescriptorSet write {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = glsl::storageImageBinding,
		.dstArrayElement = handle,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &imageInfo,
	};
	vkUpdateDescriptorSets(device.get(), 1, &write, 0, nullptr);
	return handle;
}

glsl::ResourceTableHandle ResourceTable::allocateSampledImage(VkImageView view, VkImageLayout imageLayout, VkSampler sampler) noexcept {
	ZoneScoped;
	auto handle = findFirstFreeHandle(sampledImageBitmap);

	const VkDescriptorImageInfo imageInfo {
		.sampler = sampler,
		.imageView = view,
		.imageLayout = imageLayout,
	};
	const VkWriteDescriptorSet write {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = glsl::sampledImageBinding,
		.dstArrayElement = handle,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &imageInfo,
	};
	vkUpdateDescriptorSets(device.get(), 1, &write, 0, nullptr);
	return handle;
}
