#include <Metal/MTLSampler.hpp>

#include <graphics/resource_table.hpp>
#include <vk_gltf_viewer/device.hpp>

#include <fastgltf/util.hpp>

namespace gvk = graphics::vulkan;
namespace gmtl = graphics::metal;

shaders::resource_table_handle_t graphics::resource_table::find_first_free_handle(std::vector<std::uint64_t>& bitmap) {
	ZoneScoped;
	std::lock_guard lock(bitmap_mutex);
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

void graphics::resource_table::free_handle(std::vector<std::uint64_t>& bitmap, shaders::resource_table_handle_t handle) {
	std::lock_guard lock(bitmap_mutex);
	auto i = handle / 64;
	bitmap[i] &= ~(std::uint64_t(1U) << (handle % 64));
}

void graphics::resource_table::remove_sampled_image_handle(shaders::resource_table_handle_t handle) noexcept {
	ZoneScoped;
	if (handle == shaders::invalid_handle)
		return;
	free_handle(sampled_image_bitmap, handle);
}

gvk::vk_resource_table::vk_resource_table(Device& _device) : device(_device) {
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

	sampled_image_bitmap.resize(fastgltf::alignUp(properties.maxDescriptorSetUpdateAfterBindSampledImages, 64) / 64);
	storage_image_bitmap.resize(fastgltf::alignUp(properties.maxDescriptorSetUpdateAfterBindStorageImages, 64) / 64);
}

gvk::vk_resource_table::~vk_resource_table() noexcept {
	ZoneScoped;
	vkDestroyDescriptorSetLayout(device.get(), layout, vk::allocationCallbacks.get());
	vkDestroyDescriptorPool(device.get(), pool, vk::allocationCallbacks.get());
}

shaders::resource_table_handle_t gvk::vk_resource_table::allocate_storage_image(VkImageView view, VkImageLayout imageLayout) noexcept {
	ZoneScoped;
	auto handle = find_first_free_handle(storage_image_bitmap);

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

shaders::resource_table_handle_t gvk::vk_resource_table::allocate_sampled_image(VkImageView view, VkImageLayout imageLayout, VkSampler sampler) noexcept {
	ZoneScoped;
	auto handle = find_first_free_handle(sampled_image_bitmap);

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

void gvk::vk_resource_table::remove_storage_image_handle(shaders::resource_table_handle_t handle) noexcept {
	ZoneScoped;
	if (handle == shaders::invalid_handle)
		return;
	free_handle(storage_image_bitmap, handle);
}

#if defined(VKV_METAL)
gmtl::mtl_resource_table::mtl_resource_table(NS::SharedPtr<MTL::Device> pDevice) : device(std::move(pDevice)) {
	ZoneScoped;
	auto* descriptor = MTL::ResidencySetDescriptor::alloc()->init();
	descriptor->setLabel(NS::String::string("Global resource table", NS::UTF8StringEncoding));
	// descriptor->initialCapacity();

	NS::Error* error = nullptr;
	residency_set = NS::TransferPtr(device->newResidencySet(descriptor, &error));
	if (error != nullptr) {
		throw std::runtime_error(fmt::format(
			"{}", error->localizedDescription()->utf8String()));
	}

	/** The feature set tables say there's a limit of 1M textures that can be used
	 * That number is arbitrary, and there is no actual limit beyond memory capacity.
	 * For simplicity, we'll also just use 1M which should be enough in all cases. */
	static constexpr std::size_t count = fastgltf::alignUp(1'000'000, 64);
	sampled_image_buffer = device->newBuffer(count * sizeof(sampled_texture_entry), MTL::ResourceStorageModeShared);
	sampled_image_buffer->setLabel(NS::String::string("Sampled image table", NS::UTF8StringEncoding));

	sampled_image_bitmap.resize(count);
}

gmtl::mtl_resource_table::~mtl_resource_table() noexcept {
	ZoneScoped;
	sampled_image_buffer->release();
}

shaders::resource_table_handle_t gmtl::mtl_resource_table::allocate_sampled_image(NS::SharedPtr<MTL::Texture> texture, NS::SharedPtr<MTL::SamplerState> sampler) noexcept {
	ZoneScoped;
	assert(texture && sampler);
	const auto handle = find_first_free_handle(sampled_image_bitmap);

	auto& data = static_cast<sampled_texture_entry*>(sampled_image_buffer->contents())[handle];
	data.tex = texture->gpuResourceID();
	data.sampler = sampler->gpuResourceID();

	std::lock_guard lock(resource_mutex);
	residency_set->addAllocation(texture.get());
	resources[handle] = std::move(texture);
	residency_dirty = true;
	return handle;
}

void gmtl::mtl_resource_table::remove_sampled_image_handle(shaders::resource_table_handle_t handle) noexcept {
	ZoneScoped;
	std::lock_guard lock(resource_mutex);
	assert(resources.contains(handle));
	const auto texture = resources[handle];
	resources.erase(handle);
	residency_set->removeAllocation(texture.get());
	residency_dirty = true;
	resource_table::remove_sampled_image_handle(handle);
}

void graphics::metal::mtl_resource_table::commit_residency() {
	ZoneScoped;
	std::lock_guard lock(resource_mutex);
	if (std::exchange(residency_dirty, false))
		residency_set->commit();
}

#endif
