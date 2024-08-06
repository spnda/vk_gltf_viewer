#include <tracy/Tracy.hpp>

#include <meshoptimizer.h>

#include <Metal/MTLLibrary.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLRenderPipeline.hpp>
#include <Metal/MTLDepthStencil.hpp>

#include <graphics/mtl_renderer.hpp>

#include <fastgltf/util.hpp>
#include <Metal/MTLComputeCommandEncoder.hpp>
#include <Metal/MTLComputePipeline.hpp>

#include "visbuffer/visbuffer.h"

namespace gmtl = graphics::metal;

graphics::InstanceIndex gmtl::MeshletScene::addMeshInstance(std::shared_ptr<Mesh> mesh) {
	ZoneScoped;
	auto meshletMesh = std::static_pointer_cast<MeshletMesh>(mesh);
	//auto meshletMesh = std::dynamic_pointer_cast<MeshletMesh>(mesh);

	// Add to mesh list if it's not there already
	std::uint32_t primitiveIndex;
	{
		auto it = meshes.begin();
		while (it != meshes.end()) {
			if (it->mesh == meshletMesh) {
				primitiveIndex = std::distance(meshes.begin(), it);
				break;
			}
			++it;
		}
		if (it == meshes.end()) {
			primitiveIndex = meshes.size();
			meshes.emplace_back(meshletMesh, shaders::Primitive {
				.vertexIndexBuffer = meshletMesh->vertexIndexBuffer->gpuAddress(),
				.primitiveIndexBuffer = meshletMesh->primitiveIndexBuffer->gpuAddress(),
				.vertexBuffer = meshletMesh->vertexBuffer->gpuAddress(),
				.meshletBuffer = meshletMesh->meshletBuffer->gpuAddress(),
				.aabbExtents = meshletMesh->aabbExtents,
				.aabbCenter = meshletMesh->aabbExtents,
				.meshletCount = meshletMesh->meshletCount,
				.materialIndex = 0,
			});
		}
	}

	auto instanceIndex = static_cast<std::uint32_t>(transforms.size());

	for (std::uint32_t i = 0; i < meshletMesh->meshletCount; ++i) {
		meshletDraws.emplace_back(shaders::MeshletDraw {
			.primitiveIndex = primitiveIndex,
			.meshletIndex = i,
			.transformIndex = instanceIndex,
		});
	}

	transforms.emplace_back();
	return instanceIndex;
}

void gmtl::MeshletScene::updateTransform(graphics::InstanceIndex instance, glm::fmat4x4 transform) {
	ZoneScoped;
	transforms[instance] = transform;
}

void gmtl::MeshletScene::updateDrawBuffers(std::size_t frameIndex) {
	ZoneScoped;
	auto& drawBuffer = drawBuffers[frameIndex];

	{
		auto length = drawBuffer.meshletDrawBuffer ? drawBuffer.meshletDrawBuffer->length() : 0;
		auto requiredLength = meshletDraws.size() * sizeof(decltype(meshletDraws)::value_type);
		if (requiredLength > length) {
			drawBuffer.meshletDrawBuffer = NS::TransferPtr(
					device->newBuffer(requiredLength, MTL::ResourceStorageModeShared));
			drawBuffer.meshletDrawBuffer->setLabel(NS::String::string("Meshlet draw buffer", NS::UTF8StringEncoding));
		}
		std::memcpy(drawBuffer.meshletDrawBuffer->contents(), meshletDraws.data(), requiredLength);
	}

	{
		auto length = drawBuffer.transformBuffer ? drawBuffer.transformBuffer->length() : 0;
		auto requiredLength = transforms.size() * sizeof(glm::fmat4x4);
		if (requiredLength > length) {
			drawBuffer.transformBuffer = NS::TransferPtr(
					device->newBuffer(requiredLength, MTL::ResourceStorageModeShared));
			drawBuffer.transformBuffer->setLabel(NS::String::string("Transform buffer", NS::UTF8StringEncoding));
		}
		std::memcpy(drawBuffer.transformBuffer->contents(), transforms.data(), requiredLength);
	}

	{
		auto length = drawBuffer.primitiveBuffer ? drawBuffer.primitiveBuffer->length() : 0;
		auto requiredLength = meshes.size() * sizeof(shaders::Primitive);
		if (requiredLength > length) {
			drawBuffer.primitiveBuffer = NS::TransferPtr(
					device->newBuffer(requiredLength, MTL::ResourceStorageModeShared));
			drawBuffer.primitiveBuffer->setLabel(NS::String::string("Primitive buffer", NS::UTF8StringEncoding));
		}
		for (std::size_t i = 0; auto& mesh : meshes)
			static_cast<shaders::Primitive*>(drawBuffer.primitiveBuffer->contents())[i++]
				= mesh.primitive;
	}
}

gmtl::MtlRenderer::MtlRenderer(GLFWwindow* window) {
	ZoneScoped;
	// pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

	device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());

	layer = createMetalLayer(window);
	layer->setDevice(device.get());
	layer->setPixelFormat(MTL::PixelFormatRGBA8Unorm);

	commandQueue = NS::TransferPtr(device->newCommandQueue());

	drawSemaphore = dispatch_semaphore_create(frameOverlap);

	resourceTable = std::make_shared<MtlResourceTable>(device);

	auto* urlString = NS::String::string("shaders.metallib", NS::UTF8StringEncoding);
	auto* libraryUrl = NS::URL::alloc()->initFileURLWithPath(urlString)->autorelease();

	NS::Error* error = nullptr;
	globalLibrary = NS::TransferPtr(device->newLibrary(libraryUrl, &error));
	if (error) {
		fmt::print("{}", error->localizedDescription()->utf8String());
	}

	imguiRenderer = std::make_unique<imgui::Renderer>(
			device, globalLibrary, resourceTable, layer->pixelFormat());

	cameraBuffers.resize(frameOverlap);
	for (std::size_t i = 0; auto& camera : cameraBuffers) {
		camera = NS::TransferPtr(device->newBuffer(sizeof(shaders::Camera), MTL::StorageModeShared));
		auto str = fmt::format("Camera buffer {}", i++);
		camera->setLabel(NS::String::string(str.c_str(), NS::UTF8StringEncoding));
	}

	initVisbufferPass();
	initVisbufferResolvePass();
}

gmtl::MtlRenderer::~MtlRenderer() noexcept = default;

void gmtl::MtlRenderer::initVisbufferPass() {
	ZoneScoped;
	auto* objectFunction = globalLibrary->newFunction(NS::String::string("visbuffer_object", NS::UTF8StringEncoding))->autorelease();
	auto* meshFunction = globalLibrary->newFunction(NS::String::string("visbuffer_mesh", NS::UTF8StringEncoding))->autorelease();
	auto* fragFunction = globalLibrary->newFunction(NS::String::string("visbuffer_frag", NS::UTF8StringEncoding))->autorelease();

	auto* pipelineDescriptor = MTL::MeshRenderPipelineDescriptor::alloc()->init()->autorelease();
	pipelineDescriptor->setObjectFunction(objectFunction);
	pipelineDescriptor->setMeshFunction(meshFunction);
	pipelineDescriptor->setFragmentFunction(fragFunction);
	pipelineDescriptor->setRasterSampleCount(1);

	auto* visbufferAttachment = pipelineDescriptor->colorAttachments()->object(0);
	visbufferAttachment->setPixelFormat(MTL::PixelFormatR32Uint);

	pipelineDescriptor->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

	NS::Error* error = nullptr;
	visbufferPass.pipelineState = NS::TransferPtr(
			device->newRenderPipelineState(pipelineDescriptor, MTL::PipelineOptionNone, nullptr, &error));
	if (!visbufferPass.pipelineState) {
		fmt::print("{}", error->localizedDescription()->utf8String());
	}

	// Create the depth state with reverse depth
	auto* depthStateDesc = MTL::DepthStencilDescriptor::alloc()->init();
	depthStateDesc->setDepthWriteEnabled(true);
	depthStateDesc->setDepthCompareFunction(MTL::CompareFunctionGreaterEqual);

	visbufferPass.depthState = NS::TransferPtr(device->newDepthStencilState(depthStateDesc));

	// Create the visbuffer & depth texture
	auto size = layer->drawableSize();
	auto* visbufferDesc = MTL::TextureDescriptor::texture2DDescriptor(
			MTL::PixelFormatR32Uint, size.width, size.height, false);
	visbufferDesc->setUsage(MTL::TextureUsageShaderRead & MTL::TextureUsageRenderTarget);
	visbufferDesc->setStorageMode(MTL::StorageModePrivate);

	visbufferPass.visbuffer = NS::TransferPtr(device->newTexture(visbufferDesc));

	auto* depthDesc = MTL::TextureDescriptor::texture2DDescriptor(
			MTL::PixelFormatDepth32Float, size.width, size.height, false);
	depthDesc->setUsage(MTL::TextureUsageShaderRead & MTL::TextureUsageRenderTarget);
	depthDesc->setStorageMode(MTL::StorageModePrivate);

	visbufferPass.depthTexture = NS::TransferPtr(device->newTexture(depthDesc));
}

void gmtl::MtlRenderer::initVisbufferResolvePass() {
	ZoneScoped;
	auto* resolveFunction = globalLibrary->newFunction(NS::String::string("visbuffer_resolve", NS::UTF8StringEncoding))->autorelease();

	auto* pipelineDescriptor = MTL::ComputePipelineDescriptor::alloc()->init()->autorelease();
	pipelineDescriptor->setComputeFunction(resolveFunction);

	NS::Error* error = nullptr;
	visbufferResolvePass.pipelineState = NS::TransferPtr(
			device->newComputePipelineState(pipelineDescriptor, MTL::PipelineOptionNone, nullptr, &error));
	if (!visbufferResolvePass.pipelineState) {
		fmt::print("{}", error->localizedDescription()->utf8String());
	}
}

std::unique_ptr<graphics::Buffer> gmtl::MtlRenderer::createUniqueBuffer() {
	ZoneScoped;
	return std::make_unique<MtlBuffer>();
}

std::shared_ptr<graphics::Buffer> gmtl::MtlRenderer::createSharedBuffer() {
	ZoneScoped;
	return std::make_shared<MtlBuffer>();
}

std::shared_ptr<graphics::Mesh> gmtl::MtlRenderer::createSharedMesh(std::span<shaders::Vertex> vertexBuffer, std::span<index_t> indexBuffer, glm::fvec3 aabbCenter, glm::fvec3 aabbExtents) {
	ZoneScoped;
	static constexpr auto coneWeight = 0.f; // We leave this as 0 because we're not using cluster cone culling.
	static constexpr auto maxPrimitives = fastgltf::alignDown(shaders::maxPrimitives, 4U); // meshopt requires the primitive count to be aligned to 4.
	std::size_t maxMeshlets = meshopt_buildMeshletsBound(indexBuffer.size(), shaders::maxVertices, maxPrimitives);

	std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
	std::vector<std::uint32_t> meshletVertices(maxMeshlets * shaders::maxVertices);
	std::vector<std::uint8_t> meshletTriangles(maxMeshlets * maxPrimitives * 3);

	auto mesh = std::make_shared<MeshletMesh>();
	mesh->aabbCenter = aabbCenter;
	mesh->aabbExtents = aabbExtents;

	// Create vertex buffer
	mesh->vertexBuffer = NS::TransferPtr(device->newBuffer(vertexBuffer.size_bytes(), MTL::StorageModeShared));
	std::memcpy(mesh->vertexBuffer->contents(), vertexBuffer.data(), vertexBuffer.size_bytes());

	// Build meshlets & trim buffers accordingly
	{
		mesh->meshletCount = meshopt_buildMeshlets(
			meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
			indexBuffer.data(), indexBuffer.size(),
			&vertexBuffer[0].position.x, vertexBuffer.size(), sizeof(decltype(vertexBuffer)::value_type),
			shaders::maxVertices, maxPrimitives, coneWeight);

		const auto& lastMeshlet = meshlets[mesh->meshletCount - 1];
		meshletVertices.resize(lastMeshlet.vertex_count + lastMeshlet.vertex_offset);
		meshletTriangles.resize(((lastMeshlet.triangle_count * 3 + 3) & ~3) + lastMeshlet.triangle_offset);
		meshlets.resize(mesh->meshletCount);
	}

	// Create meshlet buffer & transform meshlets
	mesh->meshletBuffer = NS::TransferPtr(device->newBuffer(meshlets.size() * sizeof(shaders::Meshlet), MTL::StorageModeShared));
	auto* glslMeshlets = static_cast<shaders::Meshlet*>(mesh->meshletBuffer->contents());
	for (std::size_t i = 0; auto& meshlet : meshlets) {
		meshopt_optimizeMeshlet(&meshletVertices[meshlet.vertex_offset],
								&meshletTriangles[meshlet.triangle_offset],
								meshlet.triangle_count, meshlet.vertex_count);

		// Compute meshlet bounds
		auto& initialVertex = vertexBuffer[meshletVertices[meshlet.vertex_offset]];
		auto min = glm::vec3(initialVertex.position), max = glm::vec3(initialVertex.position);

		for (std::size_t j = 1; j < meshlet.vertex_count; ++j) {
			std::uint32_t vertexIndex = meshletVertices[meshlet.vertex_offset + j];
			auto& vertex = vertexBuffer[vertexIndex];

			// The glm::min and glm::max functions are all component-wise.
			min = glm::min(min, vertex.position);
			max = glm::max(max, vertex.position);
		}

		// We can convert the count variables to a uint8_t since shaders::maxVertices and shaders::maxPrimitives both fit in 8-bits.
		assert(meshlet.vertex_count <= std::numeric_limits<std::uint8_t>::max());
		assert(meshlet.triangle_count <= std::numeric_limits<std::uint8_t>::max());
		auto center = (min + max) * 0.5f;
		glslMeshlets[i++] = shaders::Meshlet {
			.vertexOffset = meshlet.vertex_offset,
			.triangleOffset = meshlet.triangle_offset,
			.vertexCount = static_cast<std::uint8_t>(meshlet.vertex_count),
			.triangleCount = static_cast<std::uint8_t>(meshlet.triangle_count),
			.aabbExtents = max - center,
			.aabbCenter = center,
		};
	}

	// Finally, copy the index buffers
	mesh->vertexIndexBuffer = NS::TransferPtr(device->newBuffer(meshletVertices.size() * sizeof(decltype(meshletVertices)::value_type), MTL::StorageModeShared));
	std::memcpy(mesh->vertexIndexBuffer->contents(), meshletVertices.data(), mesh->vertexIndexBuffer->length());

	mesh->primitiveIndexBuffer = NS::TransferPtr(device->newBuffer(meshletTriangles.size() * sizeof(decltype(meshletTriangles)::value_type), MTL::StorageModeShared));
	std::memcpy(mesh->primitiveIndexBuffer->contents(), meshletTriangles.data(), mesh->primitiveIndexBuffer->length());

	return mesh;
}

std::shared_ptr<graphics::Scene> gmtl::MtlRenderer::createSharedScene() {
	ZoneScoped;
	return std::make_shared<MeshletScene>(device, graphics::frameOverlap);
}

shaders::ResourceTableHandle gmtl::MtlRenderer::createSampledTextureHandle() {
	ZoneScoped;
	return resourceTable->allocateSampledImage(nullptr, nullptr);
}

shaders::ResourceTableHandle gmtl::MtlRenderer::createStorageTextureHandle() {
	ZoneScoped;
	return resourceTable->allocateStorageImage(nullptr);
}

void gmtl::MtlRenderer::updateResolution(glm::u32vec2 resolution) {
	ZoneScoped;
	resolution *= 2; // TODO: Get CA::Layer::contentScale here.
	layer->setDrawableSize(CGSizeMake(resolution.x, resolution.y));
}

auto gmtl::MtlRenderer::getRenderResolution() const noexcept -> glm::u32vec2 {
	auto drawableSize = layer->drawableSize();
	return { drawableSize.width, drawableSize.height };
}

void gmtl::MtlRenderer::prepareFrame(std::size_t frameIndex) {
	ZoneScoped;
	dispatch_semaphore_wait(drawSemaphore, DISPATCH_TIME_FOREVER);
	if (commandBufferException) {
		std::rethrow_exception(commandBufferException);
	}
}

bool gmtl::MtlRenderer::draw(std::size_t frameIndex, graphics::Scene& gworld,
							 const shaders::Camera& camera, float dt) {
	ZoneScoped;
	auto* pool = NS::AutoreleasePool::alloc()->init();

	auto& scene = dynamic_cast<MeshletScene&>(gworld);

	auto* drawable = layer->nextDrawable();

	scene.updateDrawBuffers(frameIndex);
	*static_cast<shaders::Camera*>(cameraBuffers[frameIndex]->contents()) = camera;

	auto* bufferDesc = MTL::CommandBufferDescriptor::alloc()->init()->autorelease();
	bufferDesc->setErrorOptions(MTL::CommandBufferErrorOptionEncoderExecutionStatus);
	auto* buffer = commandQueue->commandBuffer(bufferDesc);

	auto drawCount = scene.meshletDraws.size();
	if (drawCount > 0) {
		auto* visbufferPassDescriptor = MTL::RenderPassDescriptor::alloc()->init()->autorelease();
		auto* visbufferAttachment = visbufferPassDescriptor->colorAttachments()->object(0);
		visbufferAttachment->setLoadAction(MTL::LoadActionClear);
		visbufferAttachment->setStoreAction(MTL::StoreActionStore);
		visbufferAttachment->setTexture(visbufferPass.visbuffer.get());

		// We need to clear using the visbuffer clear value. Metal only uses doubles in its ClearColor struct,
		// but 2^32 (which is visbufferClearValue currently) is perfectly representable by a double precision float,
		// so this will work for now.
		visbufferAttachment->setClearColor(MTL::ClearColor::Make(static_cast<double>(shaders::visbufferClearValue), 0., 0., 0.));

		auto* depthAttachment = MTL::RenderPassDepthAttachmentDescriptor::alloc()->init()->autorelease();
		depthAttachment->setLoadAction(MTL::LoadActionClear);
		depthAttachment->setClearDepth(0.);
		depthAttachment->setTexture(visbufferPass.depthTexture.get());
		visbufferPassDescriptor->setDepthAttachment(depthAttachment);

		auto* visbufferEncoder = buffer->renderCommandEncoder(visbufferPassDescriptor);
		visbufferEncoder->setRenderPipelineState(visbufferPass.pipelineState.get());
		visbufferEncoder->setDepthStencilState(visbufferPass.depthState.get());

		auto& sceneDrawBuffers = scene.drawBuffers[frameIndex];

		visbufferEncoder->setObjectBytes(&drawCount, sizeof drawCount, 0);
		visbufferEncoder->setObjectBuffer(cameraBuffers[frameIndex].get(), 0, 1);
		visbufferEncoder->setObjectBuffer(sceneDrawBuffers.meshletDrawBuffer.get(), 0, 2);
		visbufferEncoder->setObjectBuffer(sceneDrawBuffers.transformBuffer.get(), 0, 3);
		visbufferEncoder->setObjectBuffer(sceneDrawBuffers.primitiveBuffer.get(), 0, 4);

		visbufferEncoder->setMeshBuffer(sceneDrawBuffers.meshletDrawBuffer.get(), 0, 0);
		visbufferEncoder->setMeshBuffer(sceneDrawBuffers.transformBuffer.get(), 0, 1);
		visbufferEncoder->setMeshBuffer(sceneDrawBuffers.primitiveBuffer.get(), 0, 2);
		visbufferEncoder->setMeshBuffer(cameraBuffers[frameIndex].get(), 0, 3);

		for (auto& meshes : scene.meshes) {
			visbufferEncoder->useResource(meshes.mesh->vertexIndexBuffer.get(), MTL::ResourceUsageRead);
			visbufferEncoder->useResource(meshes.mesh->primitiveIndexBuffer.get(), MTL::ResourceUsageRead);
			visbufferEncoder->useResource(meshes.mesh->vertexBuffer.get(), MTL::ResourceUsageRead);
			visbufferEncoder->useResource(meshes.mesh->meshletBuffer.get(), MTL::ResourceUsageRead);
		}

		// We cull manually per primitive in the mesh shader
		visbufferEncoder->setCullMode(MTL::CullModeNone);

		visbufferEncoder->drawMeshThreadgroups(
				MTL::Size((drawCount + shaders::maxMeshlets - 1) / shaders::maxMeshlets, 1, 1),
				MTL::Size(visbufferPass.pipelineState->objectThreadExecutionWidth(), 1, 1),
				MTL::Size(visbufferPass.pipelineState->meshThreadExecutionWidth(), 1, 1));

		visbufferEncoder->endEncoding();
	}

	if (drawCount > 0) {
		auto* resolveEncoder = buffer->computeCommandEncoder();
		resolveEncoder->setComputePipelineState(visbufferResolvePass.pipelineState.get());

		auto& sceneDrawBuffers = scene.drawBuffers[frameIndex];

		resolveEncoder->setBuffer(sceneDrawBuffers.meshletDrawBuffer.get(), 0, 0);
		resolveEncoder->setBuffer(sceneDrawBuffers.primitiveBuffer.get(), 0, 1);

		resolveEncoder->setTexture(visbufferPass.visbuffer.get(), 0);
		resolveEncoder->setTexture(drawable->texture(), 1);

		auto threadgroupSize = MTL::Size::Make(16, 16, 1);
		auto threadgroupCount = MTL::Size::Make(
			(visbufferPass.visbuffer->width() + threadgroupSize.width - 1) / threadgroupSize.width,
			(visbufferPass.visbuffer->height() + threadgroupSize.height - 1) / threadgroupSize.height,
			1);
		resolveEncoder->dispatchThreadgroups(threadgroupCount, threadgroupSize);

		resolveEncoder->endEncoding();
	} else {
		// TODO: Clear image to black or something?
	}

	imguiRenderer->draw(buffer, drawable, getRenderResolution(), frameIndex);

	buffer->presentDrawable(drawable);

	buffer->addCompletedHandler([&](MTL::CommandBuffer* buffer) {
		ZoneScoped;
		dispatch_semaphore_signal(drawSemaphore);

		if (buffer->status() == MTL::CommandBufferStatusError) {
			auto* error = buffer->error();
			if (error) {
				fmt::print(stderr, "Command buffer error: {}\n", error->localizedDescription()->utf8String());

				auto* encoderInfos = static_cast<NS::Array*>(
						error->userInfo()->object(MTL::CommandBufferEncoderInfoErrorKey));
				for (std::size_t i = 0; i < encoderInfos->count(); ++i) {
					auto* info = static_cast<MTL::CommandBufferEncoderInfo*>(encoderInfos->object(i));
					for (std::size_t j = 0; j < info->debugSignposts()->count(); ++j) {
						auto* signpost = static_cast<NS::String*>(info->debugSignposts()->object(j));
						fmt::print(stderr, "Signpost {}: {}", j, signpost->utf8String());
					}
				}
			}

			commandBufferException = std::make_exception_ptr(
					std::runtime_error("Failed to execute command buffer"));
		}
	});

	buffer->commit();

	pool->release();

	return true;
}
