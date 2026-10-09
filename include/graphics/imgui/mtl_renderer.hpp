#pragma once

#include <type_traits>
#include <Foundation/NSSharedPtr.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLTexture.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <QuartzCore/CAMetalLayer.hpp>

#include <graphics/resource_table.hpp>

namespace graphics::metal {
	class meshlet_renderer;
}

namespace graphics::metal::imgui {
	struct geometry_buffers {
		NS::SharedPtr<MTL::Buffer> vertexBuffer;
		NS::SharedPtr<MTL::Buffer> indexBuffer;
	};

	class imgui_renderer final {
		NS::SharedPtr<MTL::Device> device;
		std::shared_ptr<mtl_resource_table> resource_table;

		MTL::RenderPipelineState* pipeline_state = nullptr;

		NS::SharedPtr<MTL::Texture> font_atlas;
		shaders::resource_table_handle_t font_atlas_handle = shaders::invalid_handle;
		NS::SharedPtr<MTL::SamplerState> font_atlas_sampler;

		std::vector<geometry_buffers> buffers;

	public:
		explicit imgui_renderer(NS::SharedPtr<MTL::Device> device,
						  NS::SharedPtr<MTL::Library> library,
						  std::shared_ptr<mtl_resource_table> resource_table,
						  MTL::PixelFormat image_format);
		~imgui_renderer() noexcept;

		void draw(MTL::CommandBuffer* commandBuffer,
			CA::MetalDrawable* drawable, shaders::color_space_e output_color_space,
			glm::u32vec2 framebuffer_size, std::size_t frame_index, bool clear_drawable);
	};
}
