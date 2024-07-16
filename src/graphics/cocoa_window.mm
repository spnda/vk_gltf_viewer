#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3.h>
#import <GLFW/glfw3native.h>

#include <tracy/Tracy.hpp>

#import <graphics/mtl_renderer.hpp>

CA::MetalLayer* graphics::metal::createMetalLayer(GLFWwindow* window) {
	ZoneScoped;
	auto* layer = CA::MetalLayer::layer();

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	layer->setDrawableSize(CGSizeMake(width, height));

	NSWindow* nswindow = glfwGetCocoaWindow(window);
	nswindow.contentView.layer = (__bridge CALayer*)layer; // NOLINT
	nswindow.contentView.wantsLayer = TRUE;

	return layer;
	//return NS::RetainPtr(layer);
}
