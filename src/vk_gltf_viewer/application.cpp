#include <cassert>
#include <numbers>
#include <ranges>

#include <tracy/Tracy.hpp>

#include <fmt/xchar.h>

#include <vulkan/vk.hpp>
#include <vulkan/pipeline_builder.hpp>
#include <GLFW/glfw3.h> // After Vulkan includes so that it detects it
#include <imgui_impl_glfw.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nvidia/dlss.hpp>
#if defined(VKV_NV_DLSS)
#include <nvsdk_ngx_helpers_vk.h>
#endif
#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/application.hpp>
#include <vk_gltf_viewer/assets.hpp>
#include <spirv_manifest.hpp>

#include <visbuffer/visbuffer.h.glsl>

#include <fastgltf/tools.hpp>

namespace fg = fastgltf;

void glfwErrorCallback(int errorCode, const char* description) {
	if (errorCode != GLFW_NO_ERROR) {
		fmt::print(stderr, "GLFW error: 0x{:x} {}\n", errorCode, description);
	}
}

void glfwResizeCallback(GLFWwindow* window, int width, int height) {
	ZoneScoped;
	if (width > 0 && height > 0) {
		glm::u32vec2 res(width, height);
		decltype(auto) app = *static_cast<Application*>(glfwGetWindowUserPointer(window));
		app.renderResolution = res;
		app.renderer->updateResolution(res);
	}
}

Application::Application(std::span<std::filesystem::path> gltfs) {
	ZoneScoped;

	// Initialize GLFW
	glfwSetErrorCallback(glfwErrorCallback);
	if (glfwInit() != GLFW_TRUE) {
		throw std::runtime_error("Failed to initialize GLFW");
	}
	deletionQueue.push([]() { glfwTerminate(); });

	// Get the main monitor's video mode
	auto* mainMonitor = glfwGetPrimaryMonitor();
	const auto* videoMode = glfwGetVideoMode(mainMonitor);

	// Create the window
	glfwDefaultWindowHints();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(
		static_cast<int>(static_cast<float>(videoMode->width) * 0.9f),
		static_cast<int>(static_cast<float>(videoMode->height) * 0.9f),
		"vk_viewer", nullptr, nullptr);
	if (window == nullptr)
		throw std::runtime_error("Failed to create window");
	deletionQueue.push([this]() { glfwDestroyWindow(window); });

	glfwSetWindowUserPointer(window, this);
	glfwSetWindowSizeCallback(window, glfwResizeCallback);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	//ImGui_ImplGlfw_InitForVulkan(window, true);
	ImGui_ImplGlfw_InitForOther(window, true);

	renderer = graphics::Renderer::createRenderer(window);
	scene = renderer->createSharedScene();
	camera = std::make_unique<Camera>();

	for (auto& gltf : gltfs) {
		auto& task = assetLoadTasks.emplace_back(std::make_shared<AssetLoadTask>(renderer, gltf));
		taskScheduler.AddTaskSetToPipe(task.get());
	}
}

Application::~Application() noexcept = default;

void Application::updateRenderResolution() {
	ZoneScoped;
	//auto swapchainExtent = toVector(swapchain->swapchain.extent);

#if defined(VKV_NV_DLSS)
	if (scalingMode == ResolutionScalingModes::DLSS) {
		auto settings = dlss::getRecommendedSettings(dlssQuality, swapchainExtent);
		renderResolution = settings.optimalRenderSize;

		device->timelineDeletionQueue->push([handle = dlssHandle]() {
			dlss::releaseFeature(handle);
		});
		dlssHandle = dlss::initFeature(*device, renderResolution, swapchainExtent);
	} else
#endif
	{
		//renderResolution = swapchainExtent;
	}

	firstFrame = true; // We need to re-transition the images since they've been recreated.
}

void Application::addAssetToScene(AssetLoadTask& assetLoadTask) {
	ZoneScoped;
	fastgltf::iterateSceneNodes(*assetLoadTask.asset, 0, fastgltf::math::fmat4x4(),
								[&](fastgltf::Node& node, const fastgltf::math::fmat4x4& mat) {
		if (!node.meshIndex.has_value())
			return;

		auto& mesh = assetLoadTask.meshes[*node.meshIndex];
		for (auto& idx : mesh.primitiveIndices) {
			auto instance = scene->addMeshInstance(assetLoadTask.primitives[idx]);
			scene->updateTransform(instance, glm::make_mat4x4(mat.data()));
		}
	});
}

void Application::run() {
	ZoneScoped;

	std::size_t currentFrame = 0;
	while (!glfwWindowShouldClose(window)) {
		if (renderer->canRender()) {
			ZoneScopedN("glfwPollEvents");
			glfwPollEvents();
		} else {
			// This will wait until we get an event, like the resize event which will recreate the swapchain.
			glfwWaitEvents();
			continue;
		}

		// Check if any asset load task has completed, and get the Asset object
		for (auto& task : assetLoadTasks) {
			if (task->GetIsComplete()) {
				if (task->exception)
					std::rethrow_exception(task->exception);
				addAssetToScene(*task);
				//vkQueueWaitIdle(device->graphicsQueue);
				//world->addAsset(task);
				task.reset();
			}
		}
		std::erase_if(assetLoadTasks, [](std::shared_ptr<AssetLoadTask>& value) {
			return !bool(value);
		});

		auto currentTime = glfwGetTime();
		deltaTime = currentTime - lastFrame;
		lastFrame = currentTime;

		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		renderUi();

		currentFrame = ++currentFrame % graphics::frameOverlap;

		renderer->prepareFrame(currentFrame);

		camera->updateCamera(window, deltaTime, renderer->getRenderResolution());

		renderer->draw(currentFrame, *scene, *camera, static_cast<float>(deltaTime));

		FrameMark;
		firstFrame = false;
	}
}

void Application::renderUi() {
	ZoneScoped;
	if (ImGui::Begin("vk_gltf_viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
		if (ImGui::BeginTabBar("#tabbar")) {
			if (ImGui::BeginTabItem("General")) {
				ImGui::SeparatorText("Performance");

				ImGui::Text("Frametime: %.2f ms", deltaTime * 1000.);
				ImGui::Text("FPS: %.2f", 1. / deltaTime);
				ImGui::Text("AFPS: %.2f rad/s", 2. * std::numbers::pi_v<double> / deltaTime); // Angular FPS

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Graphics")) {
				ImGui::SeparatorText("Resolution scaling");

				auto scalingModeName = std::find_if(availableScalingModes.begin(), availableScalingModes.end(), [&](auto& mode) {
					return mode.first == scalingMode;
				})->second;
				if (ImGui::BeginCombo("Resolution scaling", scalingModeName.data())) {
					for (const auto& [mode, name] : availableScalingModes) {
						bool isSelected = mode == scalingMode;
						if (ImGui::Selectable(name.data(), isSelected)) {
							scalingMode = mode;
							updateRenderResolution();
						}
						if (isSelected)
							ImGui::SetItemDefaultFocus();
					}
					ImGui::EndCombo();
				}

#if defined(VKV_NV_DLSS)
				if (scalingMode == ResolutionScalingModes::DLSS) {
					auto dlssQualityName = std::find_if(dlss::modes.begin(), dlss::modes.end(), [&](auto& mode) {
						return mode.first == dlssQuality;
					})->second;
					if (ImGui::BeginCombo("DLSS Mode", dlssQualityName.data())) {
						for (const auto& [quality, name]: dlss::modes) {
							bool isSelected = quality == dlssQuality;
							if (ImGui::Selectable(name.data(), isSelected)) {
								dlssQuality = quality;
								updateRenderResolution();
							}
							if (isSelected)
								ImGui::SetItemDefaultFocus();
						}

						ImGui::EndCombo();
					}
				}
#endif

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Debug")) {
				ImGui::SeparatorText("Camera");
				auto pos = camera->getPosition();
				ImGui::Text("Position: <%.2f, %.2f, %.2f>", pos.x, pos.y, pos.z);
				ImGui::DragFloat("Camera speed multiplier", &camera->speedMultiplier);

				ImGui::SeparatorText("Debug");
				ImGui::Checkbox("Freeze Camera frustum", &camera->freezeCameraFrustum);
				ImGui::Checkbox("Freeze Occlusion matrix", &camera->freezeCullingMatrix);
				//ImGui::Checkbox("Freeze animations", &world->freezeAnimations);

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}
	}
	ImGui::End();

	// ImGui::ShowDemoWindow();

	ImGui::Render();
}
