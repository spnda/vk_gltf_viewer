#include <tracy/Tracy.hpp>

#include <Metal/MTLLibrary.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandQueue.hpp>

#include <graphics/mtl_renderer.hpp>

namespace gmtl = graphics::metal;

graphics::InstanceIndex gmtl::MtlScene::addMesh(std::shared_ptr<Mesh> mesh) {

}

void gmtl::MtlScene::updateTransform(graphics::InstanceIndex instance, glm::fmat4x4 transform) {

}

gmtl::MtlRenderer::MtlRenderer(GLFWwindow* window) {
	ZoneScoped;
	// pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

	device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());

	layer = createMetalLayer(window);
	layer->setDevice(device.get());
	layer->setPixelFormat(MTL::PixelFormatRGBA8Unorm_sRGB);

	commandQueue = NS::TransferPtr(device->newCommandQueue());

	resourceTable = std::make_shared<MtlResourceTable>(device);

	auto* urlString = NS::String::string("shaders.metallib", NS::UTF8StringEncoding);
	auto* libraryUrl = NS::URL::alloc()->initFileURLWithPath(urlString)->autorelease();

	NS::Error* error = nullptr;
	globalLibrary = NS::TransferPtr(device->newLibrary(libraryUrl, &error));
	if (error) {
		fmt::print("{}", error->localizedDescription()->utf8String());
	}

	imguiRenderer = std::make_unique<imgui::Renderer>(device, globalLibrary, resourceTable);
}

gmtl::MtlRenderer::~MtlRenderer() noexcept = default;

std::unique_ptr<graphics::Buffer> gmtl::MtlRenderer::createUniqueBuffer() {

}

std::shared_ptr<graphics::Buffer> gmtl::MtlRenderer::createSharedBuffer() {

}

std::shared_ptr<graphics::Mesh> gmtl::MtlRenderer::createSharedMesh(std::span<glsl::Vertex> vertexBuffer, std::span<index_t> indexBuffer) {

}

std::shared_ptr<graphics::Scene> gmtl::MtlRenderer::createSharedScene() {
	ZoneScoped;
	return std::make_shared<MtlScene>();
}

glsl::ResourceTableHandle gmtl::MtlRenderer::createSampledTextureHandle() {
	ZoneScoped;
	return resourceTable->allocateSampledImage(nullptr, nullptr);
}

glsl::ResourceTableHandle gmtl::MtlRenderer::createStorageTextureHandle() {
	ZoneScoped;
	return resourceTable->allocateStorageImage(nullptr);
}

void gmtl::MtlRenderer::updateResolution(glm::u32vec2 resolution) {
	ZoneScoped;
}

void gmtl::MtlRenderer::prepareFrame(std::size_t frameIndex) {
	ZoneScoped;
}

bool gmtl::MtlRenderer::draw(std::size_t frameIndex, graphics::Scene& gworld, float dt) {
	ZoneScoped;
	auto* pool = NS::AutoreleasePool::alloc()->init();

	auto* drawable = layer->nextDrawable();

	auto* buffer = commandQueue->commandBuffer();

	auto drawableSize = layer->drawableSize();
	imguiRenderer->draw(buffer, drawable, {drawableSize.width, drawableSize.height }, frameIndex);

	buffer->presentDrawable(drawable);
	buffer->commit();

	pool->release();

	return true;
}
