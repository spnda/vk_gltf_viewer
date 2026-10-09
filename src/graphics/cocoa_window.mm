#import <QuartzCore/CAMetalLayer.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#import <GLFW/glfw3.h>
#import <GLFW/glfw3native.h>

#include <tracy/Tracy.hpp>

#import <graphics/mtl_renderer.hpp>

@interface CAMetalLayer (MetalRenderer)
@property(nonatomic, readwrite) CFStringRef colorspaceName;
@end

@implementation CAMetalLayer (MetalRenderer)
-(CFStringRef) colorspaceName {
	return CGColorSpaceGetName(self.colorspace);
}

-(void) setColorspaceName: (CFStringRef) name {
	CGColorSpaceRef cs_ref = CGColorSpaceCreateWithName(name);
	self.colorspace = cs_ref;
	CGColorSpaceRelease(cs_ref);
}
@end

CA::MetalLayer* graphics::metal::create_metal_layer(GLFWwindow* window) {
	ZoneScoped;
	auto layer = [CAMetalLayer layer];

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	layer.drawableSize = CGSizeMake(width, height);
	layer.displaySyncEnabled = YES;
	layer.framebufferOnly = NO;

	layer.wantsExtendedDynamicRangeContent = YES;
	layer.colorspaceName = kCGColorSpaceDisplayP3_PQ;

	NSWindow* nswindow = glfwGetCocoaWindow(window);
	nswindow.contentView.layer = layer; // NOLINT
	nswindow.contentView.wantsLayer = TRUE;

	return (__bridge CA::MetalLayer*)layer;
}

float graphics::metal::meshlet_renderer::get_max_edr_value() const {
	ZoneScoped;
	NSWindow* nswindow = glfwGetCocoaWindow(window);
	NSScreen* nsscreen = nswindow ? nswindow.screen : NSScreen.mainScreen;

	return nsscreen.maximumExtendedDynamicRangeColorComponentValue;
}

void graphics::metal::meshlet_renderer::set_output_color_space(const shaders::color_space_e space) {
	current_output_color_space = space;

	auto* ca_layer = ((__bridge CAMetalLayer*)layer);
	switch (space) {
		using enum shaders::color_space_e;
		case srgb_nonlinear:
			ca_layer.colorspaceName = kCGColorSpaceSRGB;
			ca_layer.wantsExtendedDynamicRangeContent = NO;
			break;
		case bt709_nonlinear:
			ca_layer.colorspaceName = kCGColorSpaceITUR_709;
			ca_layer.wantsExtendedDynamicRangeContent = NO;
			break;
		case bt2020_linear:
			ca_layer.colorspaceName = kCGColorSpaceExtendedLinearITUR_2020;
			ca_layer.wantsExtendedDynamicRangeContent = YES;
			break;
		case bt2020_nonlinear:
			ca_layer.colorspaceName = kCGColorSpaceITUR_2020;
			ca_layer.wantsExtendedDynamicRangeContent = YES;
			break;
		case bt2100_pq:
			ca_layer.colorspaceName = kCGColorSpaceITUR_2100_PQ;
			ca_layer.wantsExtendedDynamicRangeContent = YES;
			break;
		case bt2100_hlg:
			ca_layer.colorspaceName = kCGColorSpaceITUR_2100_HLG;
			ca_layer.wantsExtendedDynamicRangeContent = YES;
			break;
		case display_p3_linear:
			ca_layer.colorspaceName = kCGColorSpaceExtendedLinearDisplayP3;
			ca_layer.wantsExtendedDynamicRangeContent = YES;
			break;
		case display_p3_nonlinear:
			ca_layer.colorspaceName = kCGColorSpaceDisplayP3;
			ca_layer.wantsExtendedDynamicRangeContent = YES;
			break;
		case display_p3_pq:
			ca_layer.colorspaceName = kCGColorSpaceDisplayP3_PQ;
			ca_layer.wantsExtendedDynamicRangeContent = YES;
			break;
		case display_p3_hlg:
			ca_layer.colorspaceName = kCGColorSpaceDisplayP3_HLG;
			ca_layer.wantsExtendedDynamicRangeContent = YES;
			break;
		default:
			ca_layer.colorspaceName = nil;
			break;
	}
}
