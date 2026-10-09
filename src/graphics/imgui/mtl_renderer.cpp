#include <span>

#include <imgui.h>

#include <fastgltf/util.hpp>

#include <Metal/MTLSampler.hpp>
#include <Metal/MTLRenderCommandEncoder.hpp>
#include <Metal/MTLRenderPipeline.hpp>
#include <Metal/MTLLibrary.hpp>
#include <Metal/MTLDrawable.hpp>

#include <ui/ui.h>

#include <graphics/imgui/mtl_renderer.hpp>
#include <graphics/mtl_renderer.hpp>

namespace gmtl = graphics::metal;

gmtl::imgui::imgui_renderer::imgui_renderer(NS::SharedPtr<MTL::Device> nDevice, NS::SharedPtr<MTL::Library> globalLibrary, std::shared_ptr<mtl_resource_table> nResourceTable, MTL::PixelFormat imageFormat)
		: device(std::move(nDevice)), resource_table(std::move(nResourceTable)) {
	ZoneScoped;
	auto* vertexFunction = globalLibrary->newFunction(NS::String::string("ui_vert", NS::UTF8StringEncoding))->autorelease();
	auto* fragmentFunction = globalLibrary->newFunction(NS::String::string("ui_frag", NS::UTF8StringEncoding))->autorelease();

	auto* pipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init()->autorelease();
	pipelineDescriptor->setVertexFunction(vertexFunction);
	pipelineDescriptor->setFragmentFunction(fragmentFunction);
	pipelineDescriptor->setRasterSampleCount(1);

	auto* colorAttachment = pipelineDescriptor->colorAttachments()->object(0);
	colorAttachment->setPixelFormat(imageFormat);
	colorAttachment->setBlendingEnabled(true);
	colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
	colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
	colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
	colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
	colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
	colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);

	NS::Error* error = nullptr;
	pipeline_state = device->newRenderPipelineState(pipelineDescriptor, &error);
	if (!pipeline_state) {
		throw std::runtime_error(fmt::format(
			"{}", error->localizedDescription()->utf8String()));
	}

	auto& io = ImGui::GetIO();
	io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
	io.BackendRendererName = "metal::imgui::Renderer";
	io.BackendPlatformName = "Metal";

	/** Build font texture and upload GPU texture */
	io.Fonts->Build();

	unsigned char* pixels = nullptr;
	int width = 0, height = 0;
	io.Fonts->GetTexDataAsAlpha8(&pixels, &width, &height);

	/** Create the alpha Metal texture, with a one swizzle for all other channels */
	auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
			MTL::PixelFormatA8Unorm, width, height, false);
	desc->setUsage(MTL::TextureUsageShaderRead);
	desc->setStorageMode(MTL::StorageModeShared);
	desc->setSwizzle(MTL::TextureSwizzleChannels {
		MTL::TextureSwizzleOne,
		MTL::TextureSwizzleOne,
		MTL::TextureSwizzleOne,
		MTL::TextureSwizzleAlpha,
	});

	font_atlas = NS::TransferPtr(device->newTexture(desc));

	font_atlas->replaceRegion(MTL::Region::Make2D(0, 0, width, height), 0, pixels, width);

	auto* sampler_desc = MTL::SamplerDescriptor::alloc()->init()->autorelease();
	sampler_desc->setMinFilter(MTL::SamplerMinMagFilterLinear);
	sampler_desc->setMagFilter(MTL::SamplerMinMagFilterLinear);
	sampler_desc->setSupportArgumentBuffers(true);
	font_atlas_sampler = NS::TransferPtr(device->newSamplerState(sampler_desc));

	font_atlas_handle = resource_table->allocate_sampled_image(font_atlas, font_atlas_sampler);
	io.Fonts->SetTexID(font_atlas_handle);

	buffers.resize(frame_overlap);
}

gmtl::imgui::imgui_renderer::~imgui_renderer() noexcept {
	ZoneScoped;
	pipeline_state->release();
	resource_table->remove_sampled_image_handle(font_atlas_handle);
}

void gmtl::imgui::imgui_renderer::draw(MTL::CommandBuffer* commandBuffer,
		CA::MetalDrawable* drawable, shaders::color_space_e output_color_space,
		glm::u32vec2 framebuffer_size, std::size_t frame_index, bool clear_drawable) {
	ZoneScoped;
	auto* drawData = ImGui::GetDrawData();

	if (drawData->TotalVtxCount <= 0)
		return;

	auto& frameBuffers = buffers[frame_index];
	auto commandLists = std::span(drawData->CmdLists.Data, drawData->CmdLists.Size);

	const std::size_t vertexBufferSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
	const std::size_t indexBufferSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

	if (!frameBuffers.vertexBuffer || vertexBufferSize > frameBuffers.vertexBuffer->length()) {
		frameBuffers.vertexBuffer = NS::TransferPtr(
				device->newBuffer(vertexBufferSize, MTL::ResourceStorageModeShared));
	}
	if (!frameBuffers.indexBuffer || indexBufferSize > frameBuffers.indexBuffer->length()) {
		frameBuffers.indexBuffer = NS::TransferPtr(
				device->newBuffer(indexBufferSize, MTL::ResourceStorageModeShared));
	}

	// Copy the vertex and index buffers
	{
		auto* vertexDestination = static_cast<ImDrawVert*>(frameBuffers.vertexBuffer->contents());
		auto* indexDestination = static_cast<ImDrawIdx*>(frameBuffers.indexBuffer->contents());
		for (const auto& list : commandLists) {
			std::memcpy(vertexDestination, list->VtxBuffer.Data, list->VtxBuffer.Size * sizeof(ImDrawVert));
			std::memcpy(indexDestination, list->IdxBuffer.Data, list->IdxBuffer.Size * sizeof(ImDrawIdx));

			// Because the destination pointers have a type of ImDrawXYZ*, it already
			// properly takes the byte size into account.
			vertexDestination += list->VtxBuffer.Size;
			indexDestination += list->IdxBuffer.Size;
		}
		//frameBuffers.vertexBuffer->didModifyRange(NS::Range::Make(0, frameBuffers.vertexBuffer->length()));
		//frameBuffers.indexBuffer->didModifyRange(NS::Range::Make(0, frameBuffers.indexBuffer->length()));
	}

	const auto displaySize = glm::fvec2(drawData->DisplaySize);
	const auto displayPos = glm::fvec2(drawData->DisplayPos);
	const auto clipOffset = glm::fvec2(drawData->DisplayPos);      // (0,0) unless using multi-viewports
	const auto clipScale = glm::fvec2(drawData->FramebufferScale); // (1,1) unless using retina display which are often (2,2)

	auto outputSize = glm::u32vec2(displaySize * clipScale);
	shaders::UiPushConstants constants {
		.scale = 2.f / displaySize,
		.translate = -1.0F - displayPos * constants.scale,
		.output_color_space = output_color_space,
	};

	auto* renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init()->autorelease();
	auto* colorAttachment = renderPassDescriptor->colorAttachments()->object(0);
	colorAttachment->init();
	colorAttachment->setLoadAction(clear_drawable ? MTL::LoadActionClear : MTL::LoadActionLoad);
	colorAttachment->setStoreAction(MTL::StoreActionStore);
	colorAttachment->setClearColor(MTL::ClearColor::Make(0.f, 0.f, 0.f, 1.f));
	colorAttachment->setTexture(drawable->texture());

	auto* pass = commandBuffer->renderCommandEncoder(renderPassDescriptor);
	pass->setLabel(MTLSTR("ImGui"));
	pass->setRenderPipelineState(pipeline_state);

	pass->setFragmentBuffer(resource_table->sampled_image_buffer, 0, 1);

	pass->useResource(frameBuffers.vertexBuffer.get(), MTL::ResourceUsageRead);
	pass->useResource(frameBuffers.indexBuffer.get(), MTL::ResourceUsageRead);

	MTL::Viewport viewport {
		.originX = 0.0,
		.originY = 0.0,
		.width = (double)(drawData->DisplaySize.x * drawData->FramebufferScale.x),
		.height = (double)(drawData->DisplaySize.y * drawData->FramebufferScale.y),
		.znear = 0.0,
		.zfar = 1.0
	};
	pass->setViewport(viewport);

	std::uint32_t vertexOffset = 0;
	std::uint32_t indexOffset = 0;
	for (auto& list : commandLists) {
		auto cmdBuffer = std::span(list->CmdBuffer.Data, list->CmdBuffer.Size);
		for (const auto& cmd : cmdBuffer) {
			if (cmd.ElemCount == 0)
				continue;

			const glm::u32vec2 clipMin {
				fastgltf::max(0U, static_cast<std::uint32_t>((cmd.ClipRect.x - clipOffset.x) * clipScale.x)),
				fastgltf::max(0U, static_cast<std::uint32_t>((cmd.ClipRect.y - clipOffset.y) * clipScale.y))
			};
			const glm::u32vec2 clipMax {
				fastgltf::min(outputSize.x, static_cast<std::uint32_t>((cmd.ClipRect.z - clipOffset.x) * clipScale.x)),
				fastgltf::min(outputSize.y, static_cast<std::uint32_t>((cmd.ClipRect.w - clipOffset.y) * clipScale.y))
			};

			if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y) {
				continue;
			}

			MTL::ScissorRect scissorRect {
				.x = clipMin.x,
				.y = clipMin.y,
				.width = clipMax.x - clipMin.x,
				.height = clipMax.y - clipMin.y,
			};
			pass->setScissorRect(scissorRect);

			pass->useResource(font_atlas.get(), MTL::ResourceUsageSample);

			if (auto texId = cmd.GetTexID(); texId == shaders::invalid_handle) {
				constants.imageIndex = font_atlas_handle;
			} else {
				// TODO: Figure out how to make other textures resident.
				constants.imageIndex = texId;
			}
			constants.vertices = frameBuffers.vertexBuffer->gpuAddress() + (vertexOffset + cmd.VtxOffset) * sizeof(ImDrawVert);

			pass->setVertexBytes(&constants, sizeof(shaders::UiPushConstants), 0);
			pass->setFragmentBytes(&constants, sizeof(shaders::UiPushConstants), 0);

			pass->drawIndexedPrimitives(
				MTL::PrimitiveTypeTriangle,
				cmd.ElemCount,
				sizeof(ImDrawIdx) == 2 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32,
				frameBuffers.indexBuffer.get(),
				(cmd.IdxOffset + indexOffset) * sizeof(ImDrawIdx),
				1, 0, 0);
		}

		vertexOffset += list->VtxBuffer.Size;
		indexOffset += list->IdxBuffer.Size;
	}

	pass->endEncoding();
}
