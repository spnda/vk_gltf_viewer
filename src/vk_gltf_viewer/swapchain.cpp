#include <vulkan/vk.hpp>

#include <vk_gltf_viewer/swapchain.hpp>
#include <vk_gltf_viewer/application.hpp>

Swapchain::Swapchain(const Device& _device, VkSurfaceKHR _surface, VkSwapchainKHR _swapchain) : device(_device), surface(_surface) {
	ZoneScoped;
	vkb::SwapchainBuilder builder(device.get().device);
	auto result = builder
		.set_old_swapchain(_swapchain)
		.set_desired_min_image_count(Application::frameOverlap)
		.add_image_usage_flags(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
		.set_desired_format(VkSurfaceFormatKHR {.format = VK_FORMAT_R8G8B8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
		.build();
	if (!result)
		throw vulkan_error(result.error().message(), result.vk_result());

	swapchain = std::move(result).value();
	images = vk::enumerateVector<VkImage, decltype(images)>(
		vkGetSwapchainImagesKHR, VkDevice(device.get()), swapchain);

	auto imageViewResult = swapchain.get_image_views();
	if (!imageViewResult)
		throw vulkan_error(imageViewResult.error().message(), imageViewResult.vk_result());

	imageViews = std::move(imageViewResult).value();

	imageViewHandles.reserve(imageViews.size());
	for (auto& view : imageViews)
		imageViewHandles.emplace_back(
			device.get().resourceTable->allocateStorageImage(view, VK_IMAGE_LAYOUT_GENERAL));
}

Swapchain::~Swapchain() noexcept {
	ZoneScoped;
	for (auto& handle : imageViewHandles)
		device.get().resourceTable->removeStorageImageHandle(handle);
	for (auto& view : imageViews)
		vkDestroyImageView(device.get(), view, vk::allocationCallbacks.get());
	vkb::destroy_swapchain(swapchain);
}

std::unique_ptr<Swapchain> Swapchain::recreate(std::unique_ptr<Swapchain>&& oldSwapchain) {
	ZoneScoped;
	auto newSwapchain = std::make_unique<Swapchain>(
		oldSwapchain->device, oldSwapchain->surface, oldSwapchain->swapchain);

	// Push the old swapchain into the timeline deletion queue, which will delete it when it's not used anymore.
	// We move the swapchain into the lambda so that when the function gets destroyed it'll also destroy the swapchain, since it now owns it.
	newSwapchain->device.get().timelineDeletionQueue->push([oldSwapchain = std::move(oldSwapchain)]() mutable {});

	return newSwapchain;
}
