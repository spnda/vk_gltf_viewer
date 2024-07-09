#pragma once

#include <memory>

#include <glm/vec2.hpp>

#include <vulkan/vk.hpp>
#include <vulkan/command_pool.hpp>
#include <vulkan/sync_pools.hpp>

#include <nvsdk_ngx_defs.h>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/swapchain.hpp>

#include <imgui/renderer.hpp>
#include <graphics/renderer.hpp>

namespace graphics::vulkan {
struct VkMesh : graphics::Mesh {
	/** The buffer handles corresponding to the buffers in each glsl::Primitive. */
	std::unique_ptr<ScopedBuffer> vertexIndexBuffer;
	std::unique_ptr<ScopedBuffer> primitiveIndexBuffer;
	std::unique_ptr<ScopedBuffer> vertexBuffer;
	std::unique_ptr<ScopedBuffer> meshletBuffer;

	glm::fvec3 aabbExtents;
	glm::fvec3 aabbCenter;

	std::uint32_t meshletCount;
	std::uint32_t materialIndex;
};

struct DrawBuffers {
	bool isMeshletBufferBuilt = false;
	std::unique_ptr<ScopedBuffer> meshletDrawBuffer;
	std::unique_ptr<ScopedBuffer> transformBuffer;
};

class VkScene : graphics::Scene {
	std::vector<std::shared_ptr<Mesh>> meshes;
	std::unique_ptr<ScopedBuffer> primitiveBuffer;
	std::unique_ptr<ScopedBuffer> materialBuffer;

	std::vector<DrawBuffers> drawBuffers;

	void rebuildDrawBuffer(std::size_t frameIndex);
	void updateTransformBuffer(std::size_t frameIndex);

public:
	void addMesh(std::shared_ptr<Mesh> mesh) override;

	void updateDrawBuffers(std::size_t frameIndex, float dt);
};

/** Sync primitives required for frame synchronization around presenting and work submission */
struct FrameSyncData {
	std::unique_ptr<vk::Semaphore> imageAvailable;
	std::unique_ptr<vk::Semaphore> renderingFinished;
	std::unique_ptr<vk::Fence> presentFinished;
};

/** A CommandPool for each frame, together with pre-allocated command buffers */
struct FrameCommandPool {
	vk::CommandPool commandPool;
	VkCommandBuffer commandBuffer;
};

enum class ResolutionScalingModes {
	None,
#if defined(VKV_NV_DLSS)
	DLSS,
#endif
};

class VkRenderer : public graphics::Renderer {
	friend std::shared_ptr<Renderer> graphics::Renderer::createRenderer();

	std::unique_ptr<Instance> instance;
	std::unique_ptr<Device> device;
	std::unique_ptr<Swapchain> swapchain;

	glm::u32vec2 renderResolution;
	ResolutionScalingModes scalingMode = ResolutionScalingModes::None;
	std::vector<std::pair<ResolutionScalingModes, std::string_view>> availableScalingModes;

#if defined(VKV_NV_DLSS)
	NVSDK_NGX_Handle* dlssHandle = nullptr;
	NVSDK_NGX_PerfQuality_Value dlssQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
#endif

	std::vector<FrameSyncData> frameSyncData;
	std::vector<FrameCommandPool> frameCommandPools;

	std::unique_ptr<imgui::Renderer> imguiRenderer;

	bool swapchainNeedsRebuild = false;

	std::unique_ptr<Buffer> createUniqueBuffer() override;
	std::shared_ptr<Buffer> createSharedBuffer() override;

	std::shared_ptr<Mesh> createSharedMesh(std::span<glsl::Vertex> vertexBuffer, std::span<index_t> indexBuffer) override;

	bool canRender() override {
		return !swapchainNeedsRebuild;
	}

	void updateResolution(glm::u32vec2 resolution) override;

	void prepareFrame(std::size_t frameIndex) override;
	bool draw(std::size_t frameIndex, Scene& world, float dt) override;
};
} // namespace graphics::vulkan
