#include <graphics/renderer.hpp>
#include <graphics/vk_renderer.hpp>

std::shared_ptr<graphics::Renderer> graphics::Renderer::createRenderer() {
	ZoneScoped;
	// TODO: Make this actually check for Vulkan support and platform support.
#if defined(_WIN32) || defined(__linux__)
	return std::make_shared<graphics::vulkan::VkRenderer>();
#endif
	return nullptr;
}
