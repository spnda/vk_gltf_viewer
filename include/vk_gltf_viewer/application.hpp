#pragma once

#include <deque>
#include <filesystem>
#include <span>
#include <vector>

#include <GLFW/glfw3.h>

#include <vulkan/vk.hpp>
#include <vulkan/command_pool.hpp>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/swapchain.hpp>
#include <vk_gltf_viewer/deletion_queue.hpp>
#include <vk_gltf_viewer/image.hpp>
#include <vk_gltf_viewer/imgui_renderer.hpp>
#include <vk_gltf_viewer/assets.hpp>
#include <vk_gltf_viewer/camera.hpp>

/** Sync primitives required for frame synchronization around presenting and work submission */
struct FrameSyncData {
    VkSemaphore imageAvailable;
    VkSemaphore renderingFinished;
    VkFence presentFinished;
};

/** A CommandPool for each frame, together with pre-allocated command buffers */
struct FrameCommandPool {
	vk::CommandPool commandPool;
	VkCommandBuffer commandBuffer;
};

struct DrawBuffers {
	std::unique_ptr<ScopedBuffer> meshletDrawBuffer;
	std::unique_ptr<ScopedBuffer> transformBuffer;
};

struct VisibilityBufferPass {
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;

	std::unique_ptr<ScopedImage> image;
	glsl::ResourceTableHandle imageHandle = glsl::invalidHandle;
	std::unique_ptr<ScopedImage> depthImage;
};

struct VisibilityBufferResolvePass {
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
};

/** The main Application class */
class Application {
	friend void glfwResizeCallback(GLFWwindow* window, int width, int height);

	std::vector<std::shared_ptr<AssetLoadTask>> assetLoadTasks;
	std::vector<std::unique_ptr<Asset>> loadedAssets;

	/** The global deletion queue for all sorts of objects */
	DeletionQueue deletionQueue;

	GLFWwindow* window;
	VkSurfaceKHR surface = VK_NULL_HANDLE;

	std::unique_ptr<Instance> instance;
	std::unique_ptr<Device> device;
	std::unique_ptr<Swapchain> swapchain;

	double deltaTime = 0., lastFrame = 0.;

	std::vector<FrameSyncData> frameSyncData;
	std::vector<FrameCommandPool> frameCommandPools;

	std::unique_ptr<imgui::Renderer> imguiRenderer;

	std::unique_ptr<Camera> camera;
	std::vector<DrawBuffers> drawBuffers;

	VisibilityBufferPass visbufferPass;
	VisibilityBufferResolvePass visbufferResolvePass;

	void initVisbufferPass();
	void initVisbufferResolvePass();

	void renderUi();

	void updateDrawBuffer(std::size_t currentFrame);

public:
	explicit Application(std::span<std::filesystem::path> gltfs);
	~Application() noexcept;

	/** The amount of frames we render ahead, regardless of what the swapchain supports */
	static constexpr std::uint32_t frameOverlap = 3;

	void run();
};
