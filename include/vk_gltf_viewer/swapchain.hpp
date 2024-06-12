#pragma once

#include <vulkan/vk.hpp>
#include <VkBootstrap.h>

#include <glm/vec2.hpp>

#include <vk_gltf_viewer/device.hpp>

#include <resource_table.glsl.h>

constexpr glm::u32vec2 toVector(VkExtent2D extent) noexcept {
	return {extent.width, extent.height};
}

struct Swapchain {
	vkb::Swapchain swapchain;

	std::vector<VkImage> images;
	std::vector<VkImageView> imageViews;
	std::vector<glsl::ResourceTableHandle> imageViewHandles;

	std::reference_wrapper<const Device> device;
	VkSurfaceKHR surface = VK_NULL_HANDLE;

	explicit Swapchain(const Device& device, VkSurfaceKHR surface, VkSwapchainKHR swapchain = VK_NULL_HANDLE);
	~Swapchain() noexcept;

	static std::unique_ptr<Swapchain> recreate(std::unique_ptr<Swapchain>&& oldSwapchain);

	operator VkSwapchainKHR() const noexcept {
		return VkSwapchainKHR(swapchain);
	}
};
