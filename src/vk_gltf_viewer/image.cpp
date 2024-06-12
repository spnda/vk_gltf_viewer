#include <vk_gltf_viewer/image.hpp>

ScopedImage::ScopedImage(const Device& _device, const VkImageCreateInfo* imageInfo, const VmaAllocationCreateInfo* allocationInfo) : device(_device), extent(imageInfo->extent) {
	// Create the image itself
	vk::checkResult(vmaCreateImage(device.get().allocator, imageInfo, allocationInfo,
								   &handle, &allocation, nullptr),
					"Failed to create image: {}");

	// Create the default image view
	const VkImageViewCreateInfo imageViewInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = handle,
		.viewType = [&]() -> VkImageViewType {
			switch (imageInfo->imageType) {
				case VK_IMAGE_TYPE_1D:
					return imageInfo->arrayLayers == 1 ? VK_IMAGE_VIEW_TYPE_1D : VK_IMAGE_VIEW_TYPE_1D_ARRAY;
				case VK_IMAGE_TYPE_2D:
					return imageInfo->arrayLayers == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
				case VK_IMAGE_TYPE_3D:
					return VK_IMAGE_VIEW_TYPE_3D;
				default:
					std::unreachable();
			}
		}(),
		.format = imageInfo->format,
		.subresourceRange = {
			.aspectMask = [&]() -> VkImageAspectFlags {
				switch (imageInfo->format) {
					case VK_FORMAT_D16_UNORM:
					case VK_FORMAT_D16_UNORM_S8_UINT:
					case VK_FORMAT_D24_UNORM_S8_UINT:
					case VK_FORMAT_X8_D24_UNORM_PACK32:
					case VK_FORMAT_D32_SFLOAT:
						return VK_IMAGE_ASPECT_DEPTH_BIT;
					default:
						return VK_IMAGE_ASPECT_COLOR_BIT;
				}
			}(),
			.levelCount = imageInfo->mipLevels,
			.layerCount = imageInfo->arrayLayers,
		},
	};
	vk::checkResult(vkCreateImageView(device.get(), &imageViewInfo, vk::allocationCallbacks.get(), &defaultView),
					"Failed to create default image view: {}");
}

ScopedImage::~ScopedImage() noexcept {
	if (defaultView != VK_NULL_HANDLE)
		vkDestroyImageView(device.get(), defaultView, vk::allocationCallbacks.get());
	vmaDestroyImage(device.get().allocator, handle, allocation);
}

ScopedImageView::ScopedImageView(const Device& _device, const VkImageViewCreateInfo* imageViewInfo) : device(_device) {
	vk::checkResult(vkCreateImageView(device.get(), imageViewInfo, vk::allocationCallbacks.get(), &handle),
					"Failed to create image view");
}

ScopedImageView::~ScopedImageView() noexcept {
	vkDestroyImageView(device.get(), handle, vk::allocationCallbacks.get());
}
