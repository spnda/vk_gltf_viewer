#include <Metal/MTLSampler.hpp>

#include <graphics/resource_table.hpp>
#include <vk_gltf_viewer/device.hpp>

#include <fastgltf/util.hpp>

namespace gvk = graphics::vulkan;
namespace gmtl = graphics::metal;

shaders::ResourceTableHandle graphics::ResourceTable::findFirstFreeHandle(std::vector<std::uint64_t>& bitmap) {
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

void graphics::ResourceTable::freeHandle(std::vector<std::uint64_t>& bitmap, shaders::ResourceTableHandle handle) {
	std::lock_guard lock(bitmapMutex);
	auto i = handle / 64;
	bitmap[i] &= ~(std::uint64_t(1U) << (handle % 64));
}

void graphics::ResourceTable::removeStorageImageHandle(shaders::ResourceTableHandle handle) noexcept {
	ZoneScoped;
	if (handle == shaders::invalidHandle)
		return;
	freeHandle(storageImageBitmap, handle);
}

void graphics::ResourceTable::removeSampledImageHandle(shaders::ResourceTableHandle handle) noexcept {
	ZoneScoped;
	if (handle == shaders::invalidHandle)
		return;
	freeHandle(sampledImageBitmap, handle);
}

gvk::VkResourceTable::VkResourceTable(Device& _device) : device(_device) {
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
			.binding = shaders::sampledImageBinding,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = properties.maxDescriptorSetUpdateAfterBindSampledImages,
			.stageFlags = VK_SHADER_STAGE_ALL,
		},
		{
			.binding = shaders::storageImageBinding,
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

gvk::VkResourceTable::~VkResourceTable() noexcept {
	ZoneScoped;
	vkDestroyDescriptorSetLayout(device.get(), layout, vk::allocationCallbacks.get());
	vkDestroyDescriptorPool(device.get(), pool, vk::allocationCallbacks.get());
}

shaders::ResourceTableHandle gvk::VkResourceTable::allocateStorageImage(VkImageView view, VkImageLayout imageLayout) noexcept {
	ZoneScoped;
	auto handle = findFirstFreeHandle(storageImageBitmap);

	const VkDescriptorImageInfo imageInfo {
			.imageView = view,
			.imageLayout = imageLayout,
	};
	const VkWriteDescriptorSet write {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = set,
			.dstBinding = shaders::storageImageBinding,
			.dstArrayElement = handle,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.pImageInfo = &imageInfo,
	};
	vkUpdateDescriptorSets(device.get(), 1, &write, 0, nullptr);
	return handle;
}

shaders::ResourceTableHandle gvk::VkResourceTable::allocateSampledImage(VkImageView view, VkImageLayout imageLayout, VkSampler sampler) noexcept {
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
			.dstBinding = shaders::sampledImageBinding,
			.dstArrayElement = handle,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &imageInfo,
	};
	vkUpdateDescriptorSets(device.get(), 1, &write, 0, nullptr);
	return handle;
}

#if defined(VKV_METAL)
gmtl::MtlResourceTable::MtlResourceTable(NS::SharedPtr<MTL::Device> pDevice) : device(std::move(pDevice)) {
	ZoneScoped;
	/** The feature set tables say there's a limit of 1M textures that can be used
	 * That number is arbitrary, and there is no actual limit beyond memory capacity.
	 * For simplicity, we'll also just use 1M which should be enough in all cases. */
	static constexpr std::size_t count = fastgltf::alignUp(1'000'000, 64);
	sampledImageBuffer = device->newBuffer(count * sizeof(SampledTextureEntry), MTL::ResourceStorageModeShared);

	storageImageBuffer = device->newBuffer(count * sizeof(MTL::ResourceID), MTL::ResourceStorageModeShared);

	sampledImageBitmap.resize(count);
	storageImageBitmap.resize(count);
}

gmtl::MtlResourceTable::~MtlResourceTable() noexcept {
	ZoneScoped;
	storageImageBuffer->release();
	sampledImageBuffer->release();
}

shaders::ResourceTableHandle gmtl::MtlResourceTable::allocateStorageImage(MTL::Texture* texture) noexcept {
	ZoneScoped;
	auto handle = findFirstFreeHandle(storageImageBitmap);

	static_cast<MTL::ResourceID*>(storageImageBuffer->contents())[handle] = texture->gpuResourceID();
	return handle;
}

shaders::ResourceTableHandle gmtl::MtlResourceTable::allocateSampledImage(MTL::Texture* texture, MTL::SamplerState* sampler) noexcept {
	ZoneScoped;
	auto handle = findFirstFreeHandle(storageImageBitmap);

	auto& data = static_cast<SampledTextureEntry*>(sampledImageBuffer->contents())[handle];
	data.tex = texture->gpuResourceID();
	data.sampler = sampler->gpuResourceID();
	return handle;
}
#endif
