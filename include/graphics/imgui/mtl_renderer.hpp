#pragma once

#include <type_traits>
#include <Foundation/NSSharedPtr.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLTexture.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <QuartzCore/CAMetalLayer.hpp>

#include <graphics/resource_table.hpp>

namespace graphics::metal {
	class MtlRenderer;
}

namespace graphics::metal::imgui {
	struct GeometryBuffers {
		NS::SharedPtr<MTL::Buffer> vertexBuffer;
		NS::SharedPtr<MTL::Buffer> indexBuffer;
	};

	class Renderer final {
		NS::SharedPtr<MTL::Device> device;
		std::shared_ptr<MtlResourceTable> resourceTable;

		MTL::RenderPipelineState* pipelineState = nullptr;

		MTL::Texture* fontAtlas = nullptr;
		glsl::ResourceTableHandle fontAtlasHandle = glsl::invalidHandle;
		MTL::SamplerState* fontAtlasSampler = nullptr;

		std::vector<GeometryBuffers> buffers;

	public:
		explicit Renderer(NS::SharedPtr<MTL::Device> device, NS::SharedPtr<MTL::Library> library, std::shared_ptr<MtlResourceTable> resourceTable);
		~Renderer() noexcept;

		void draw(MTL::CommandBuffer* commandBuffer, CA::MetalDrawable* drawable,
				  glm::u32vec2 framebufferSize, std::size_t frameIndex);
	};
}
