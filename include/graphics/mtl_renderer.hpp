#pragma once

#include <graphics/renderer.hpp>

#include <Foundation/NSSharedPtr.hpp>
#include <Metal/MTLDevice.hpp>
#include <QuartzCore/CAMetalLayer.hpp>

#include <graphics/resource_table.hpp>
#include <graphics/imgui/mtl_renderer.hpp>

namespace graphics::metal {
class MtlScene : public graphics::Scene {
public:
	explicit MtlScene() = default;
	~MtlScene() noexcept override = default;

	InstanceIndex addMesh(std::shared_ptr<Mesh> mesh) override;
	void updateTransform(InstanceIndex instance, glm::fmat4x4 transform) override;
};

CA::MetalLayer* createMetalLayer(GLFWwindow* window);

class MtlRenderer : public graphics::Renderer {
	friend std::shared_ptr<Renderer> graphics::Renderer::createRenderer(GLFWwindow* window);

	/** This comes first so that it wraps around the entire lifetime of the renderer object */
	// NS::SharedPtr<NS::AutoreleasePool> pool;

	NS::SharedPtr<MTL::Device> device;

	CA::MetalLayer* layer;
	NS::SharedPtr<MTL::CommandQueue> commandQueue;

	std::shared_ptr<MtlResourceTable> resourceTable;

	NS::SharedPtr<MTL::Library> globalLibrary;

	std::unique_ptr<imgui::Renderer> imguiRenderer;

public:
	explicit MtlRenderer(GLFWwindow* window);
	~MtlRenderer() noexcept override;

	std::unique_ptr<Buffer> createUniqueBuffer() override;
	std::shared_ptr<Buffer> createSharedBuffer() override;

	std::shared_ptr<Mesh> createSharedMesh(std::span<glsl::Vertex> vertexBuffer, std::span<index_t> indexBuffer) override;

	std::shared_ptr<Scene> createSharedScene() override;

	glsl::ResourceTableHandle createSampledTextureHandle() override;
	glsl::ResourceTableHandle createStorageTextureHandle() override;

	bool canRender() override {
		return true; // TODO: Detect window being minimized or sth
	}

	void updateResolution(glm::u32vec2 resolution) override;

	void prepareFrame(std::size_t frameIndex) override;
	bool draw(std::size_t frameIndex, Scene& world, float dt) override;
};
}
