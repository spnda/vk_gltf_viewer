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
		decltype(auto) app = *static_cast<Application*>(glfwGetWindowUserPointer(window));
		app.renderer->updateResolution(glm::u32vec2(width, height));
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
}

Application::~Application() noexcept {
	ZoneScoped;
	if (volkGetLoadedDevice() != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(*device);

		world.reset();
	}

	deletionQueue.flush();
}

void Application::initVisbufferPass() {
	ZoneScoped;

	if (visbufferPass.image)
		device->timelineDeletionQueue->push(make_shared_function([this, handle = visbufferPass.imageHandle, image = std::move(visbufferPass.image)]() mutable {
			device->resourceTable->removeStorageImageHandle(handle);
			image.reset();
		}));

	const VmaAllocationCreateInfo allocationInfo {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	const VkImageCreateInfo imageInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R32_UINT,
		.extent = {
			.width = renderResolution.x,
			.height = renderResolution.y,
			.depth = 1,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	visbufferPass.image = std::make_unique<ScopedImage>(*device, &imageInfo, &allocationInfo);
	deletionQueue.push([this]() { visbufferPass.image.reset(); });
	vk::setDebugUtilsName(*device, visbufferPass.image->getHandle(), "Visbuffer");
	vk::setDebugUtilsName(*device, visbufferPass.image->getDefaultView(), "Visbuffer view");

	visbufferPass.imageHandle = device->resourceTable->allocateStorageImage(
		visbufferPass.image->getDefaultView(), VK_IMAGE_LAYOUT_GENERAL);

	if (visbufferPass.depthImage)
		device->timelineDeletionQueue->push(make_shared_function([image = std::move(visbufferPass.depthImage)]() mutable {
			image.reset();
		}));

	const VkImageCreateInfo depthImageInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_D32_SFLOAT,
		.extent = {
			.width = renderResolution.x,
			.height = renderResolution.y,
			.depth = 1,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	visbufferPass.depthImage = std::make_unique<ScopedImage>(*device, &depthImageInfo, &allocationInfo);
	deletionQueue.push([this]() { visbufferPass.depthImage.reset(); });
	vk::setDebugUtilsName(*device, visbufferPass.depthImage->getHandle(), "Depth image");
	vk::setDebugUtilsName(*device, visbufferPass.depthImage->getDefaultView(), "Depth image view");

	if (visbufferPass.mvImage)
		device->timelineDeletionQueue->push(make_shared_function([image = std::move(visbufferPass.mvImage)]() mutable {
			image.reset();
		}));

	const VkImageCreateInfo mvImageInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R16G16_SFLOAT,
		.extent = {
			.width = renderResolution.x,
			.height = renderResolution.y,
			.depth = 1,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	visbufferPass.mvImage = std::make_unique<ScopedImage>(*device, &mvImageInfo, &allocationInfo);
	deletionQueue.push([this]() { visbufferPass.mvImage.reset(); });
	vk::setDebugUtilsName(*device, visbufferPass.mvImage->getHandle(), "Motion vectors");
	vk::setDebugUtilsName(*device, visbufferPass.mvImage->getDefaultView(), "Motion vectors default view");

	if (visbufferPass.pipeline == VK_NULL_HANDLE) {
		const VkPushConstantRange pushConstantRange = {
			.stageFlags = VK_SHADER_STAGE_ALL,
			.offset = 0,
			.size = sizeof(glsl::VisbufferPushConstants),
		};
		const VkPipelineLayoutCreateInfo layoutCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &device->resourceTable->getLayout(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pushConstantRange,
		};
		vk::checkResult(vkCreatePipelineLayout(*device, &layoutCreateInfo, vk::allocationCallbacks.get(),
											   &visbufferPass.pipelineLayout),
						"Failed to create visbuffer pipeline layout");
		vk::setDebugUtilsName(*device, visbufferPass.pipelineLayout, "Visbuffer pipeline layout");

		deletionQueue.push([this]() {
			vkDestroyPipelineLayout(*device, visbufferPass.pipelineLayout, vk::allocationCallbacks.get());
		});

		VkShaderModule fragModule;
		loadShader(*device, visbuffer_frag_glsl, &fragModule);
		VkShaderModule meshModule;
		loadShader(*device, visbuffer_mesh_glsl, &meshModule);
		VkShaderModule taskModule;
		loadShader(*device, visbuffer_task_glsl, &taskModule);

		constexpr std::array<VkFormat, 2> colorAttachmentFormats {{
			VK_FORMAT_R32_UINT,
			VK_FORMAT_R16G16_SFLOAT,
		}};
		const auto depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
		const VkPipelineRenderingCreateInfo renderingCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.colorAttachmentCount = static_cast<std::uint32_t>(colorAttachmentFormats.size()),
			.pColorAttachmentFormats = colorAttachmentFormats.data(),
			.depthAttachmentFormat = depthAttachmentFormat,
		};

		std::array<VkPipelineColorBlendAttachmentState, 2> blendAttachments {{
			{
				.blendEnable = VK_FALSE,
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
								  VK_COLOR_COMPONENT_A_BIT,
			},
			{
				.blendEnable = VK_FALSE,
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
								  VK_COLOR_COMPONENT_A_BIT,
			}
		}};

		auto builder = vk::GraphicsPipelineBuilder(*device, 1)
			.setPipelineLayout(0, visbufferPass.pipelineLayout)
			.pushPNext(0, &renderingCreateInfo)
			.addDynamicState(0, VK_DYNAMIC_STATE_SCISSOR)
			.addDynamicState(0, VK_DYNAMIC_STATE_VIEWPORT)
			.setBlendAttachments(0, std::span(blendAttachments.data(), blendAttachments.size()))
			.setTopology(0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
			.setDepthState(0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL)
			.setRasterState(0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
			.setMultisampleCount(0, VK_SAMPLE_COUNT_1_BIT)
			.setScissorCount(0, 1U)
			.setViewportCount(0, 1U)
			.addShaderStage(0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule)
			.addShaderStage(0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule)
			.addShaderStage(0, VK_SHADER_STAGE_TASK_BIT_EXT, taskModule);

		vk::checkResult(builder.build(&visbufferPass.pipeline), "Failed to create visbuffer raster pipeline");
		vk::setDebugUtilsName(*device, visbufferPass.pipeline, "Visibility buffer mesh pipeline");

		vkDestroyShaderModule(*device, fragModule, vk::allocationCallbacks.get());
		vkDestroyShaderModule(*device, meshModule, vk::allocationCallbacks.get());
		vkDestroyShaderModule(*device, taskModule, vk::allocationCallbacks.get());

		deletionQueue.push([this]() {
			vkDestroyPipeline(*device, visbufferPass.pipeline, vk::allocationCallbacks.get());
		});
	}
}

void Application::initVisbufferResolvePass() {
	ZoneScoped;
	if (visbufferResolvePass.colorImage) {
		device->timelineDeletionQueue->push(make_shared_function([this, handle = visbufferResolvePass.colorImageHandle, image = std::move(visbufferResolvePass.colorImage)]() mutable {
			device->resourceTable->removeStorageImageHandle(handle);
			image.reset();
		}));
	}

	const VmaAllocationCreateInfo allocationInfo {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	const VkImageCreateInfo colorImageInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = swapchain->swapchain.image_format,
		.extent = {
			.width = renderResolution.x,
			.height = renderResolution.y,
			.depth = 1,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	visbufferResolvePass.colorImage = std::make_unique<ScopedImage>(*device, &colorImageInfo, &allocationInfo);
	deletionQueue.push([this]() { visbufferResolvePass.colorImage.reset(); });
	vk::setDebugUtilsName(*device, visbufferResolvePass.colorImage->getHandle(), "Color render image");
	vk::setDebugUtilsName(*device, visbufferResolvePass.colorImage->getDefaultView(), "Color render image view");

	visbufferResolvePass.colorImageHandle = device->resourceTable->allocateStorageImage(
		visbufferResolvePass.colorImage->getDefaultView(), VK_IMAGE_LAYOUT_GENERAL);

	if (visbufferResolvePass.pipeline == VK_NULL_HANDLE) {
		const VkPushConstantRange pushConstantRange = {
			.stageFlags = VK_SHADER_STAGE_ALL,
			.offset = 0,
			.size = sizeof(glsl::VisbufferResolvePushConstants),
		};
		const VkPipelineLayoutCreateInfo layoutCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &device->resourceTable->getLayout(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pushConstantRange,
		};
		vk::checkResult(vkCreatePipelineLayout(*device, &layoutCreateInfo, vk::allocationCallbacks.get(),
											   &visbufferResolvePass.pipelineLayout),
						"Failed to create visbuffer resolve pipeline layout");
		vk::setDebugUtilsName(*device, visbufferResolvePass.pipelineLayout,
							  "Visbuffer resolve pipeline layout");

		deletionQueue.push([this]() {
			vkDestroyPipelineLayout(*device, visbufferResolvePass.pipelineLayout,
									vk::allocationCallbacks.get());
		});

		VkShaderModule compModule;
		loadShader(*device, visbuffer_resolve_comp_glsl, &compModule);

		auto builder = vk::ComputePipelineBuilder(*device, 1)
			.setPipelineLayout(0, visbufferResolvePass.pipelineLayout)
			.setShaderStage(0, VK_SHADER_STAGE_COMPUTE_BIT, compModule);
		vk::checkResult(builder.build(&visbufferResolvePass.pipeline),
						"Failed to create visbuffer resolve pipeline");
		vk::setDebugUtilsName(*device, visbufferResolvePass.pipeline,
							  "Visibility buffer resolve pipeline");

		vkDestroyShaderModule(*device, compModule, vk::allocationCallbacks.get());

		deletionQueue.push([this]() {
			vkDestroyPipeline(*device, visbufferResolvePass.pipeline, vk::allocationCallbacks.get());
		});
	}
}

void Application::initHiZReductionPass() {
	ZoneScoped;
	if (hizReductionPass.reduceSampler == VK_NULL_HANDLE) {
		const VkSamplerReductionModeCreateInfo reductionModeInfo {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
			.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN,
		};
		const VkSamplerCreateInfo samplerInfo {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.pNext = &reductionModeInfo,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.minLod = 0.f,
			.maxLod = 16.f,
		};
		vk::checkResult(vkCreateSampler(*device, &samplerInfo, vk::allocationCallbacks.get(), &hizReductionPass.reduceSampler),
						"Failed to create HiZ reduction sampler");
		vk::setDebugUtilsName(*device, hizReductionPass.reduceSampler, "HiZ reduction sampler");
		deletionQueue.push([this]() { vkDestroySampler(*device, hizReductionPass.reduceSampler, vk::allocationCallbacks.get()); });
	}

	if (hizReductionPass.depthPyramid) {
		device->timelineDeletionQueue->push(make_shared_function([this, handle = hizReductionPass.depthPyramidHandle, image = std::move(hizReductionPass.depthPyramid), views = std::move(hizReductionPass.depthPyramidViews)]() mutable {
			for (auto& view : views) {
				device->resourceTable->removeSampledImageHandle(view.sampledHandle);
				device->resourceTable->removeStorageImageHandle(view.storageHandle);
				view.view.reset();
			}
			device->resourceTable->removeSampledImageHandle(handle);
			image.reset();
		}));
	}

	auto mipLevels = static_cast<std::uint32_t>(std::floor(std::log2(
		glm::max(renderResolution.x, renderResolution.y))));

	const VmaAllocationCreateInfo allocationInfo {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	const VkImageCreateInfo depthPyramidInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R32_SFLOAT,
		.extent = {
			.width = renderResolution.x >> 1,
			.height = renderResolution.y >> 1,
			.depth = 1,
		},
		.mipLevels = mipLevels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	hizReductionPass.depthPyramid = std::make_unique<ScopedImage>(*device, &depthPyramidInfo, &allocationInfo);
	deletionQueue.push([this]() { hizReductionPass.depthPyramid.reset(); });
	vk::setDebugUtilsName(*device, hizReductionPass.depthPyramid->getHandle(), "HiZ Depth pyramid");
	vk::setDebugUtilsName(*device, hizReductionPass.depthPyramid->getDefaultView(), "HiZ Depth pyramid view");

	hizReductionPass.depthPyramidHandle = device->resourceTable->allocateSampledImage(
		hizReductionPass.depthPyramid->getDefaultView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, hizReductionPass.reduceSampler);

	hizReductionPass.depthPyramidViews.resize(mipLevels + 1);
	for (std::uint32_t i = 0; i < (mipLevels + 1); ++i) {
		auto& view = hizReductionPass.depthPyramidViews[i];
		if (i == 0) {
			view.sampledHandle = device->resourceTable->allocateSampledImage(visbufferPass.depthImage->getDefaultView(), VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, hizReductionPass.reduceSampler);
			continue;
		}

		const VkImageViewCreateInfo imageViewInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = hizReductionPass.depthPyramid->getHandle(),
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = VK_FORMAT_R32_SFLOAT,
			.components = {
				.r = VK_COMPONENT_SWIZZLE_R,
			},
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = i - 1,
				.levelCount = 1,
				.layerCount = 1,
			},
		};
		view.view = std::make_unique<ScopedImageView>(*device, &imageViewInfo);
		view.storageHandle = device->resourceTable->allocateStorageImage(view.view->getHandle(), VK_IMAGE_LAYOUT_GENERAL);
		view.sampledHandle = device->resourceTable->allocateSampledImage(view.view->getHandle(), VK_IMAGE_LAYOUT_GENERAL, hizReductionPass.reduceSampler);
	}

	deletionQueue.push([this] {
		for (auto& view : hizReductionPass.depthPyramidViews) {
			device->resourceTable->removeSampledImageHandle(view.sampledHandle);
			device->resourceTable->removeStorageImageHandle(view.storageHandle);
			view.view.reset();
		}
	});

	if (hizReductionPass.reducePipeline == VK_NULL_HANDLE) {
		const VkPushConstantRange pushConstantRange = {
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.offset = 0,
			.size = sizeof(glsl::ResourceTableHandle) * 2 + sizeof(glm::u32vec2),
		};
		const VkPipelineLayoutCreateInfo layoutCreateInfo {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &device->resourceTable->getLayout(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pushConstantRange,
		};
		vk::checkResult(vkCreatePipelineLayout(*device, &layoutCreateInfo, vk::allocationCallbacks.get(), &hizReductionPass.reducePipelineLayout),
						"Failed to create HiZ reduction pipeline layout");
		vk::setDebugUtilsName(*device, hizReductionPass.reducePipelineLayout, "HiZ reduction pipeline layout");

		deletionQueue.push([this]() {
			vkDestroyPipelineLayout(*device, hizReductionPass.reducePipelineLayout, vk::allocationCallbacks.get());
		});

		VkShaderModule reduce;
		loadShader(*device, hiz_reduce_comp_glsl, &reduce);

		auto builder = vk::ComputePipelineBuilder(*device, 1)
			.setPipelineLayout(0, hizReductionPass.reducePipelineLayout)
			.setShaderStage(0, VK_SHADER_STAGE_COMPUTE_BIT, reduce);

		vk::checkResult(builder.build(&hizReductionPass.reducePipeline), "Failed to build HiZ reduction pipeline");
		vk::setDebugUtilsName(*device, hizReductionPass.reducePipeline,  "HiZ reduction pipeline");

		vkDestroyShaderModule(*device, reduce, vk::allocationCallbacks.get());

		deletionQueue.push([this] {
			vkDestroyPipeline(*device, hizReductionPass.reducePipeline, vk::allocationCallbacks.get());
		});
	}
}

void Application::updateRenderResolution() {
	ZoneScoped;
	auto swapchainExtent = toVector(swapchain->swapchain.extent);

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
		renderResolution = swapchainExtent;
	}

	initVisbufferPass();
	initVisbufferResolvePass();
	initHiZReductionPass();

	firstFrame = true; // We need to re-transition the images since they've been recreated.
}

void Application::run() {
	ZoneScoped;

	swapchainNeedsRebuild = false;
	std::size_t currentFrame = 0;
	while (!glfwWindowShouldClose(window)) {
		if (!swapchainNeedsRebuild) {
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
				vkQueueWaitIdle(device->graphicsQueue);
				world->addAsset(task);
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

		currentFrame = ++currentFrame % frameOverlap;
		auto& syncData = frameSyncData[currentFrame];

		renderer->prepareFrame(currentFrame);

		renderer->draw(currentFrame, *scene, static_cast<float>(deltaTime));

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
				ImGui::Checkbox("Freeze animations", &world->freezeAnimations);

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}
	}
	ImGui::End();

	// ImGui::ShowDemoWindow();

	ImGui::Render();
}
