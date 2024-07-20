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

	/** The global deletion queue for all sorts of objects */
	DeletionQueue deletionQueue;

	GLFWwindow* window;
	std::shared_ptr<graphics::Renderer> renderer;
	std::shared_ptr<graphics::Scene> scene;
	std::unique_ptr<Camera> camera;

	glm::u32vec2 renderResolution;
	ResolutionScalingModes scalingMode = ResolutionScalingModes::None;
	std::vector<std::pair<ResolutionScalingModes, std::string_view>> availableScalingModes;

	bool firstFrame = true;
	double deltaTime = 0., lastFrame = 0.;

	void updateRenderResolution();
	void addAssetToScene(AssetLoadTask& assetLoadTask);
	void renderUi();

public:
	explicit Application(std::span<std::filesystem::path> gltfs);
	~Application() noexcept;

	void run();
};
