#include <graphics/renderer.hpp>

#include <graphics/vk_renderer.hpp>
#include <graphics/mtl_renderer.hpp>

std::shared_ptr<graphics::renderer> graphics::renderer::create_renderer(GLFWwindow* window) {
	ZoneScoped;
	// TODO: Make this actually check for Vulkan support and platform support.
#if defined(_WIN32) || defined(__linux__)
	return std::make_shared<graphics::vulkan::VkRenderer>(window);
#elif defined(VKV_METAL)
	return std::make_shared<graphics::metal::meshlet_renderer>(window);
#endif
	return nullptr;
}
