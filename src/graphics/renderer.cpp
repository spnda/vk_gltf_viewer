#include <graphics/renderer.hpp>

#include <graphics/vk_renderer.hpp>
#include <graphics/mtl_renderer.hpp>

std::shared_ptr<graphics::Renderer> graphics::Renderer::createRenderer(GLFWwindow* window) {
	ZoneScoped;
	// TODO: Make this actually check for Vulkan support and platform support.
#if defined(_WIN32) || defined(__linux__)
	return std::make_shared<graphics::vulkan::VkRenderer>();
#elif defined(VKV_METAL)
	return std::make_shared<graphics::metal::MtlRenderer>(window);
#endif
	return nullptr;
}
