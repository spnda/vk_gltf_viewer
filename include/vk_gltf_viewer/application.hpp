#pragma once

#include <deque>
#include <filesystem>
#include <span>
#include <vector>

#include <GLFW/glfw3.h>

#include <vulkan/vk.hpp>
#include <vulkan/command_pool.hpp>

#if defined(VKV_NV_DLSS)
#include <nvsdk_ngx_defs.h>
#endif

#include <graphics/renderer.hpp>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/swapchain.hpp>
#include <vk_gltf_viewer/deletion_queue.hpp>
#include <vk_gltf_viewer/image.hpp>
#include <vk_gltf_viewer/assets.hpp>
#include <vk_gltf_viewer/camera.hpp>

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

/**
 * Visbuffer generation pass. This rasterizes the scene to a simple R32_UINT color attachment,
 * which contains the draw index and triangle index. Just from this data we can generate all
 * required information in the resolve pass. This is effectively a cut down gbuffer.
 */
struct VisibilityBufferPass {
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;

	std::unique_ptr<ScopedImage> image;
	glsl::ResourceTableHandle imageHandle = glsl::invalidHandle;
	std::unique_ptr<ScopedImage> depthImage;
	std::unique_ptr<ScopedImage> mvImage;
};

/**
 * Takes the visbuffer and calculates barycentrics to manually interpolate all vertex attributes.
 */
struct VisibilityBufferResolvePass {
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;

	/* This color image is only used when we apply resolution scaling, otherwise we render directly to the swapchain. */
	std::unique_ptr<ScopedImage> colorImage;
	glsl::ResourceTableHandle colorImageHandle = glsl::invalidHandle;
};

/**
 * The HiZ reduction pass. This takes the depth buffer generated in the visbuffer path
 * and generates a hierarchical depth buffer. These are effectively just mipmaps, however
 * the special thing is that each pixel has the greatest depth value from the previous mip.
 * This way we can perform occlusion culling in the next frame with this data.
 */
struct HiZReductionPass {
	VkPipelineLayout reducePipelineLayout = VK_NULL_HANDLE;
	VkPipeline reducePipeline = VK_NULL_HANDLE;

	struct HiZImageView {
		std::unique_ptr<ScopedImageView> view;
		glsl::ResourceTableHandle storageHandle = glsl::invalidHandle;
		glsl::ResourceTableHandle sampledHandle = glsl::invalidHandle;
	};

	std::unique_ptr<ScopedImage> depthPyramid;
	std::vector<HiZImageView> depthPyramidViews;
	glsl::ResourceTableHandle depthPyramidHandle = glsl::invalidHandle;
	VkSampler reduceSampler = VK_NULL_HANDLE;
};

enum class ResolutionScalingModes {
	None,
#if defined(VKV_NV_DLSS)
	DLSS,
#endif
};

/** The main Application class */
class Application {
	friend void glfwResizeCallback(GLFWwindow* window, int width, int height);

	std::vector<std::shared_ptr<AssetLoadTask>> assetLoadTasks;
	std::unique_ptr<World> world;

	/** The global deletion queue for all sorts of objects */
	DeletionQueue deletionQueue;

	std::shared_ptr<graphics::Renderer> renderer;

	std::shared_ptr<graphics::Scene> scene;

	GLFWwindow* window;
	VkSurfaceKHR surface = VK_NULL_HANDLE;

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

	bool swapchainNeedsRebuild = false;
	bool firstFrame = true;
	double deltaTime = 0., lastFrame = 0.;

	std::vector<FrameSyncData> frameSyncData;
	std::vector<FrameCommandPool> frameCommandPools;

	std::unique_ptr<Camera> camera;

	VisibilityBufferPass visbufferPass;
	VisibilityBufferResolvePass visbufferResolvePass;
	HiZReductionPass hizReductionPass;

	void initVisbufferPass();
	void initVisbufferResolvePass();
	void initHiZReductionPass();

	void updateRenderResolution();

	void renderUi();

public:
	explicit Application(std::span<std::filesystem::path> gltfs);
	~Application() noexcept;

	/** The amount of frames we render ahead, regardless of what the swapchain supports */
	static constexpr std::uint32_t frameOverlap = 3;

	void run();
};
