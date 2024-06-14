#include <cassert>
#include <numbers>
#include <ranges>

#include <tracy/Tracy.hpp>

#include <fmt/xchar.h>

#include <vulkan/vk.hpp>
#include <vulkan/pipeline_builder.hpp>
#include <GLFW/glfw3.h> // After Vulkan includes so that it detects it

#include <imgui.h>

#include <glm/glm.hpp>

#include <nvidia/dlss.hpp>
#include <nvsdk_ngx_helpers_vk.h>
#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/application.hpp>
#include <vk_gltf_viewer/assets.hpp>
#include <spirv_manifest.hpp>

#include <visbuffer/visbuffer.glsl.h>

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
		app.swapchain = Swapchain::recreate(std::move(app.swapchain));

		app.updateRenderResolution();

		app.swapchainNeedsRebuild = false;
		app.firstFrame = true;
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

	// Create the instance
	instance = std::make_unique<Instance>();
	deletionQueue.push([this]() { instance.reset(); });

	// Create the window surface
	vk::checkResult(glfwCreateWindowSurface(*instance, window, nullptr, &surface), "Failed to create window surface");
	deletionQueue.push([this]() {
		vkDestroySurfaceKHR(*instance, surface, nullptr);
	});

	// Create the Vulkan device
	device = std::make_unique<Device>(*instance, surface);
	deletionQueue.push([this]() { device.reset(); });

	// With the device created, we can now launch asset loading tasks
	for (auto& gltf : gltfs) {
		assetLoadTasks.emplace_back(std::make_shared<AssetLoadTask>(*device, gltf));
		taskScheduler.AddTaskSetToPipe(assetLoadTasks.back().get());
	}
	deletionQueue.push([this]() {
		for (auto& task : assetLoadTasks)
			taskScheduler.WaitforTask(task.get());
		assetLoadTasks.clear();
	});

	// Create the swapchain
	swapchain = std::make_unique<Swapchain>(*device, surface);
	deletionQueue.push([this]() { swapchain.reset(); });

	renderResolution = toVector(swapchain->swapchain.extent);

	availableScalingModes.emplace_back(ResolutionScalingModes::None, "None\0");

#if defined(VKV_NV_DLSS)
	if (dlss::isSupported()) {
		availableScalingModes.emplace_back(ResolutionScalingModes::DLSS, "DLSS\0");

		dlssQuality = NVSDK_NGX_PerfQuality_Value_DLAA; // TODO: Change to 'Auto' mode?
		dlssHandle = dlss::initFeature(*device, renderResolution, toVector(swapchain->swapchain.extent));
		deletionQueue.push([this]() { dlss::releaseFeature(dlssHandle); });
	}
#endif

	// Initialize the ImGui renderer
	imguiRenderer = std::make_unique<imgui::Renderer>(*device, window, swapchain->swapchain.image_format);
	imguiRenderer->initFrameData(frameOverlap);
	auto& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_IsSRGB;
	io.Fonts->AddFontDefault();
	imguiRenderer->createFontAtlas();
	deletionQueue.push([this]() { imguiRenderer.reset(); });

	// Initialize the camera buffers
	camera = std::make_unique<Camera>(*device, frameOverlap);
	deletionQueue.push([this]() { camera.reset(); });

	// Create the per-frame sync primitives
	frameSyncData.resize(frameOverlap);
	for (auto& frame : frameSyncData) {
		frame.imageAvailable = std::make_unique<vk::Semaphore>(*device);
		vk::setDebugUtilsName(*device, frame.imageAvailable->handle, "Image acquire semaphore");

		frame.renderingFinished = std::make_unique<vk::Semaphore>(*device);
		vk::setDebugUtilsName(*device, frame.renderingFinished->handle, "Rendering finished semaphore");

		frame.presentFinished = std::make_unique<vk::Fence>(*device, VK_FENCE_CREATE_SIGNALED_BIT);
		vk::setDebugUtilsName(*device, frame.presentFinished->handle, "Present fence");
	}
	deletionQueue.push([this] {
		for (auto& frame : frameSyncData) {
			frame.presentFinished.reset();
			frame.renderingFinished.reset();
			frame.imageAvailable.reset();
		}
	});

	// Create the command pools
	frameCommandPools.resize(frameOverlap);
	for (auto& pool : frameCommandPools) {
		pool.commandPool.create(*device, device->graphicsQueueFamily);
		pool.commandBuffer = pool.commandPool.allocate();
	}
	deletionQueue.push([this] {
		for (auto& pool : frameCommandPools)
			pool.commandPool.destroy();
	});

	initVisbufferPass();
	initVisbufferResolvePass();
	initHiZReductionPass();

	world = std::make_unique<World>(*device, frameOverlap);
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
		device->timelineDeletionQueue->push([this, handle = visbufferPass.imageHandle, image = std::move(visbufferPass.image)]() mutable {
			device->resourceTable->removeStorageImageHandle(handle);
			image.reset();
		});

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
		device->timelineDeletionQueue->push([image = std::move(visbufferPass.depthImage)]() mutable {
			image.reset();
		});

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
		device->timelineDeletionQueue->push([image = std::move(visbufferPass.mvImage)]() mutable {
			image.reset();
		});

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
		device->timelineDeletionQueue->push([this, handle = visbufferResolvePass.colorImageHandle, image = std::move(visbufferResolvePass.colorImage)]() mutable {
			device->resourceTable->removeStorageImageHandle(handle);
			image.reset();
		});
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
		device->timelineDeletionQueue->push([this, handle = hizReductionPass.depthPyramidHandle, image = std::move(hizReductionPass.depthPyramid), views = std::move(hizReductionPass.depthPyramidViews)]() mutable {
			for (auto& view : views) {
				device->resourceTable->removeStorageImageHandle(view.storageHandle);
				view.view.reset();
			}
			device->resourceTable->removeSampledImageHandle(handle);
			image.reset();
		});
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

		imguiRenderer->newFrame();
		ImGui::NewFrame();

		renderUi();

		currentFrame = ++currentFrame % frameOverlap;
		auto& syncData = frameSyncData[currentFrame];

		// Wait for the last render for this frame index to finish, so that we can use
		// the associated resources again.
		syncData.presentFinished->wait(std::numeric_limits<std::uint64_t>::max());
		syncData.presentFinished->reset();

		// Check if anything can be deleted this frame.
		device->timelineDeletionQueue->check();

		camera->updateCamera(currentFrame, window, deltaTime, renderResolution);
		world->updateDrawBuffers(currentFrame, static_cast<float>(deltaTime));

		auto& cmdPool = frameCommandPools[currentFrame];
		vkResetCommandPool(*device, cmdPool.commandPool, 0);
		auto& cmd = cmdPool.commandBuffer;

		// Acquire the next swapchain image
		std::uint32_t swapchainImageIndex = 0;
		{
			auto result = vkAcquireNextImageKHR(*device, *swapchain, std::numeric_limits<std::uint64_t>::max(),
												syncData.imageAvailable->handle,
												VK_NULL_HANDLE, &swapchainImageIndex);
			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
				swapchainNeedsRebuild = true;
				continue;
			}
			vk::checkResult(result, "Failed to acquire swapchain image");
		}

		// Begin the command buffer
		VkCommandBufferBeginInfo beginInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, // We're only using once, then resetting.
		};
		vkBeginCommandBuffer(cmd, &beginInfo);

		// Transition the swapchain image from UNDEFINED -> GENERAL
		// Transition the visbuffer image from UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
		// Transition the depth image from UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL
		// Transition the depth pyramid from UNDEFINED/GENERAL -> SHADER_READ_ONLY_OPTIMAL
		{
			std::array<VkImageMemoryBarrier2, 4> imageBarriers {{
				{
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
					.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
					.srcAccessMask = VK_ACCESS_2_NONE,
					.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
					.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.newLayout = VK_IMAGE_LAYOUT_GENERAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = swapchain->images[swapchainImageIndex],
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.levelCount = 1,
						.layerCount = 1,
					},
				},
				{
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
					.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
					.srcAccessMask = VK_ACCESS_2_NONE,
					.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = visbufferPass.image->getHandle(),
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.levelCount = 1,
						.layerCount = 1,
					},
				},
				{
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
					.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
					.srcAccessMask = VK_ACCESS_2_NONE,
					.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
					.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = visbufferPass.depthImage->getHandle(),
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
						.levelCount = 1,
						.layerCount = 1,
					},
				},
				{
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
					.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
					.srcAccessMask = VK_ACCESS_2_NONE,
					.dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
					.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
					.oldLayout = firstFrame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
					.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = hizReductionPass.depthPyramid->getHandle(),
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.levelCount = VK_REMAINING_MIP_LEVELS,
						.layerCount = 1,
					},
				},
			}};
			const VkDependencyInfo dependencyInfo {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size()),
				.pImageMemoryBarriers = imageBarriers.data(),
			};
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);
		}

		// Visbuffer pass
		{
			TracyVkZone(device->tracyCtx, cmd, "Visbuffer");
			const VkDebugUtilsLabelEXT label {
				.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
				.pLabelName = "Visbuffer",
			};
			vkCmdBeginDebugUtilsLabelEXT(cmd, &label);

			std::array<VkRenderingAttachmentInfo, 2> attachments {{
				{
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.imageView = visbufferPass.image->getDefaultView(),
					.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = {
						.color = {
							.uint32 = { glsl::visbufferClearValue },
						}
					}
				},
				{
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.imageView = visbufferPass.mvImage->getDefaultView(),
					.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = {
						.color = {
							.float32 = { 0.f, 0.f, 0.f, 0.f },
						}
					}
				},
			}};
			const VkRenderingAttachmentInfo depthAttachment {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.imageView = visbufferPass.depthImage->getDefaultView(),
				.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
				.resolveMode = VK_RESOLVE_MODE_NONE,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = {0.0f, 0.0f},
			};
			const VkRenderingInfo renderingInfo {
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
				.renderArea = {
					.offset = {},
					.extent = {
						.width = renderResolution.x,
						.height = renderResolution.y,
					},
				},
				.layerCount = 1,
				.colorAttachmentCount = static_cast<std::uint32_t>(attachments.size()),
				.pColorAttachments = attachments.data(),
				.pDepthAttachment = &depthAttachment,
			};
			vkCmdBeginRendering(cmd, &renderingInfo);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, visbufferPass.pipeline);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, visbufferPass.pipelineLayout,
									0, 1, &device->resourceTable->getSet(), 0, nullptr);

			const VkViewport viewport = {
				.x = 0.0F,
				.y = 0.0F,
				.width = static_cast<float>(renderResolution.x),
				.height = static_cast<float>(renderResolution.y),
				.minDepth = 0.0F,
				.maxDepth = 1.0F,
			};
			vkCmdSetViewport(cmd, 0, 1, &viewport);

			const VkRect2D scissor = renderingInfo.renderArea;
			vkCmdSetScissor(cmd, 0, 1, &scissor);

			auto& drawBuffer = world->drawBuffers[currentFrame];
			auto totalMeshletCount = drawBuffer.meshletDrawBuffer
				? static_cast<std::uint32_t>(drawBuffer.meshletDrawBuffer->getBufferSize() / sizeof(glsl::MeshletDraw))
				: 0U;

			if (totalMeshletCount > 0) {
				glsl::VisbufferPushConstants pushConstants {
					.drawBuffer = drawBuffer.meshletDrawBuffer->getDeviceAddress(),
					.meshletDrawCount = totalMeshletCount,
					.transformBuffer = drawBuffer.transformBuffer->getDeviceAddress(),
					.primitiveBuffer = world->primitiveBuffer->getDeviceAddress(),
					.cameraBuffer = camera->getCameraDeviceAddress(currentFrame),
					.materialBuffer = 0,
					.depthPyramid = hizReductionPass.depthPyramidHandle,
				};
				vkCmdPushConstants(cmd, visbufferPass.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof pushConstants,
								   &pushConstants);

				vkCmdDrawMeshTasksEXT(cmd, (totalMeshletCount + glsl::maxMeshlets - 1) / glsl::maxMeshlets, 1, 1);
			}

			vkCmdEndRendering(cmd);

			vkCmdEndDebugUtilsLabelEXT(cmd);
		}

		// Image barrier for visbuffer image from visbuffer pass -> resolve pass
		// Transition hierarchical depth pyramid from SHADER_READ_ONLY_OPTIMAL -> GENERAL
		{
			std::array<VkImageMemoryBarrier2, 2> imageBarriers {{
				{
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
					.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
					.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
					.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					.newLayout = VK_IMAGE_LAYOUT_GENERAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = visbufferPass.image->getHandle(),
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.levelCount = 1,
						.layerCount = 1,
					},
				},
				{
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
					.srcStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
					.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
					.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
					.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					.newLayout = VK_IMAGE_LAYOUT_GENERAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = hizReductionPass.depthPyramid->getHandle(),
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = VK_REMAINING_MIP_LEVELS,
						.layerCount = 1,
					},
				},
			}};
			const VkDependencyInfo dependencyInfo {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size()),
				.pImageMemoryBarriers = imageBarriers.data(),
			};
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);
		}

		// Visbuffer resolve pass
		{
			TracyVkZone(device->tracyCtx, cmd, "Visbuffer resolve");
			const VkDebugUtilsLabelEXT label {
				.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
				.pLabelName = "Visbuffer resolve",
			};
			vkCmdBeginDebugUtilsLabelEXT(cmd, &label);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visbufferResolvePass.pipeline);

			glsl::VisbufferResolvePushConstants pushConstants {
				.visbufferHandle = visbufferPass.imageHandle,
				.outputImageHandle = scalingMode != ResolutionScalingModes::None ? visbufferResolvePass.colorImageHandle : swapchain->imageViewHandles[swapchainImageIndex],
			};
			vkCmdPushConstants(cmd, visbufferResolvePass.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof pushConstants, &pushConstants);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visbufferResolvePass.pipelineLayout,
			                        0, 1, &device->resourceTable->getSet(), 0, nullptr);

			vkCmdDispatch(cmd, renderResolution.x / 32, renderResolution.y, 1);

			vkCmdEndDebugUtilsLabelEXT(cmd);
		}

		if (!camera->freezeCullingMatrix) {
			TracyVkZone(device->tracyCtx, cmd, "HiZ reduction");
			const VkDebugUtilsLabelEXT label {
				.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
				.pLabelName = "HiZ reduction",
			};
			vkCmdBeginDebugUtilsLabelEXT(cmd, &label);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizReductionPass.reducePipelineLayout,
			                        0, 1, &device->resourceTable->getSet(), 0, nullptr);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizReductionPass.reducePipeline);

			for (std::uint32_t i = 1; i < hizReductionPass.depthPyramidViews.size(); ++i) {
				auto levelSize = renderResolution >> i;

				struct HiZreducePushConstants {
					glsl::ResourceTableHandle sourceImage;
					glsl::ResourceTableHandle outputImage;
					glm::u32vec2 imageSize;
				} pushConstants {
					.sourceImage = hizReductionPass.depthPyramidViews[i - 1].sampledHandle,
					.outputImage = hizReductionPass.depthPyramidViews[i].storageHandle,
					.imageSize = levelSize,
				};
				vkCmdPushConstants(cmd, hizReductionPass.reducePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(pushConstants), &pushConstants);

				vkCmdDispatch(cmd, fg::alignUp(levelSize.x, 32) / 32, fg::alignUp(levelSize.y, 32) / 32, 1);

				if (i + 1 != hizReductionPass.depthPyramidViews.size()) {
					const VkImageMemoryBarrier barrier{
						.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
						.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
						.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
						.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
						.newLayout = VK_IMAGE_LAYOUT_GENERAL,
						.image = *hizReductionPass.depthPyramid,
						.subresourceRange = {
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
							.baseMipLevel = i,
							.levelCount = 1,
							.layerCount = 1,
						}
					};
					vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr,
					                     1, &barrier);
				}
			}

			vkCmdEndDebugUtilsLabelEXT(cmd);
		}

#if defined(VKV_NV_DLSS)
		// DLSS pass
		if (scalingMode == ResolutionScalingModes::DLSS) {
			TracyVkZone(device->tracyCtx, cmd, "DLSS");
			const VkDebugUtilsLabelEXT label {
				.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
				.pLabelName = "DLSS",
			};
			vkCmdBeginDebugUtilsLabelEXT(cmd, &label);

			const VkMemoryBarrier2 memoryBarrier {
				.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				.dstAccessMask = VK_ACCESS_2_NONE,
			};
			const VkDependencyInfo dep {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.memoryBarrierCount = 1,
				.pMemoryBarriers = &memoryBarrier,
			};
			vkCmdPipelineBarrier2(cmd, &dep);

			NVSDK_NGX_Parameter* params = nullptr;
			NVSDK_NGX_VULKAN_AllocateParameters(&params);

			NVSDK_NGX_Resource_VK colorInput {
				.Resource = {
					.ImageViewInfo = {
						.ImageView = visbufferResolvePass.colorImage->getDefaultView(),
						.Image = *visbufferResolvePass.colorImage,
						.SubresourceRange = {
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
							.levelCount = 1,
							.layerCount = 1,
						},
						.Format = swapchain->swapchain.image_format,
						.Width = renderResolution.x,
						.Height = renderResolution.y,
					}
				},
				.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
				.ReadWrite = true,
			};
			NVSDK_NGX_Resource_VK colorOutput {
				.Resource = {
					.ImageViewInfo = {
						.ImageView = swapchain->imageViews[swapchainImageIndex],
						.Image = swapchain->images[swapchainImageIndex],
						.SubresourceRange = {
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
							.levelCount = 1,
							.layerCount = 1,
						},
						.Format = swapchain->swapchain.image_format,
						.Width = swapchain->swapchain.extent.width,
						.Height = swapchain->swapchain.extent.height,
					}
				},
				.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
				.ReadWrite = true,
			};
			NVSDK_NGX_Resource_VK depthAttachment {
				.Resource = {
					.ImageViewInfo = {
						.ImageView = visbufferPass.depthImage->getDefaultView(),
						.Image = *visbufferPass.depthImage,
						.SubresourceRange = {
							.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
							.levelCount = 1,
							.layerCount = 1,
						},
						.Format = VK_FORMAT_D32_SFLOAT,
						.Width = renderResolution.x,
						.Height = renderResolution.y,
					}
				},
				.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
				.ReadWrite = false,
			};
			NVSDK_NGX_Resource_VK motionVectors {
				.Resource = {
					.ImageViewInfo = {
						.ImageView = visbufferPass.mvImage->getDefaultView(),
						.Image = *visbufferPass.mvImage,
						.SubresourceRange = {
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
							.levelCount = 1,
							.layerCount = 1,
						},
						.Format = VK_FORMAT_R16G16_SFLOAT,
						.Width = renderResolution.x,
						.Height = renderResolution.y,
					}
				},
				.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
				.ReadWrite = true,
			};

			NVSDK_NGX_VK_DLSS_Eval_Params dlssEval {
				.Feature = {
					.pInColor = &colorInput,
					.pInOutput = &colorOutput,
					.InSharpness = 0.35f, // Some random value that works?
				},
				.pInDepth = &depthAttachment,
				.pInMotionVectors = &motionVectors,
				.InRenderSubrectDimensions = {
					.Width = renderResolution.x,
					.Height = renderResolution.y,
				},
				.InMVScaleX = 1.f,
				.InMVScaleY = 1.f,
				.InPreExposure = 1.f,
				.InExposureScale = 1.f,
				.InFrameTimeDeltaInMsec = static_cast<float>(deltaTime),
			};

			auto res = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, dlssHandle, params, &dlssEval);
			if (NVSDK_NGX_FAILED(res)) {
				fmt::print("Failed to NVSDK_NGX_VULKAN_EvaluateFeature for DLSS, code = {:x}, info: {}", std::to_underlying(res), res);
			}

			NVSDK_NGX_VULKAN_DestroyParameters(params);

			vkCmdEndDebugUtilsLabelEXT(cmd);
		}
#endif

		// Image barrier for swapchain image from resolve pass -> UI draw pass
		{
			const VkImageMemoryBarrier2 swapchainImageBarrier {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
				.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = swapchain->images[swapchainImageIndex],
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.levelCount = 1,
					.layerCount = 1,
				},
			};
			const VkDependencyInfo dependencyInfo{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &swapchainImageBarrier,
			};
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);
		}

		// Draw UI
		{
			TracyVkZone(device->tracyCtx, cmd, "ImGui rendering");
			const VkDebugUtilsLabelEXT label {
				.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
				.pLabelName = "ImGui rendering",
			};
			vkCmdBeginDebugUtilsLabelEXT(cmd, &label);

			// Insert a barrier to protect against any hazard reads from ImGui textures we might be using as render targets.
			const VkMemoryBarrier2 memoryBarrier {
				.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
				.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
			};
			const VkDependencyInfo dependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.memoryBarrierCount = 1,
				.pMemoryBarriers = &memoryBarrier,
			};
			vkCmdPipelineBarrier2(cmd, &dependency);

			auto extent = glm::u32vec2(renderResolution.x, renderResolution.y);
			imguiRenderer->draw(cmd, swapchain->imageViews[swapchainImageIndex], extent, currentFrame);

			vkCmdEndDebugUtilsLabelEXT(cmd);
		}

		// Transition the swapchain image from COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC
		{
			const VkImageMemoryBarrier2 swapchainImageBarrier {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_NONE,
				.dstAccessMask = VK_ACCESS_2_NONE,
				.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = swapchain->images[swapchainImageIndex],
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.levelCount = 1,
					.layerCount = 1,
				},
			};
			const VkDependencyInfo dependencyInfo{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &swapchainImageBarrier,
			};
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);
		}

		// Always collect at the end of the main command buffer.
		TracyVkCollect(device->tracyCtx, cmd);

		vkEndCommandBuffer(cmd);

		// Submit
		{
			const VkSemaphoreSubmitInfo waitSemaphoreInfo {
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
				.semaphore = syncData.imageAvailable->handle,
				.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			};
			const VkCommandBufferSubmitInfo bufferSubmitInfo {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
				.commandBuffer = cmd,
			};
			std::array<VkSemaphoreSubmitInfo, 2> signalSemaphoreInfos = {{
				{
					.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
					.semaphore = syncData.renderingFinished->handle,
					.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				},
				{
					.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
					.semaphore = device->timelineDeletionQueue->getSemaphoreHandle(),
					.value = device->timelineDeletionQueue->nextValue(),
					.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				},
			}};
			const VkSubmitInfo2 submitInfo {
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
				.waitSemaphoreInfoCount = 1,
				.pWaitSemaphoreInfos = &waitSemaphoreInfo,
				.commandBufferInfoCount = 1,
				.pCommandBufferInfos = &bufferSubmitInfo,
				.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signalSemaphoreInfos.size()),
				.pSignalSemaphoreInfos = signalSemaphoreInfos.data(),
			};
			vk::checkResult(device->graphicsQueue.submit(submitInfo, *syncData.presentFinished),
							"Failed to submit frame command buffer to queue");

			// Lastly, present the swapchain image as soon as rendering is done
			const VkPresentInfoKHR presentInfo {
				.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &syncData.renderingFinished->handle,
				.swapchainCount = 1,
				.pSwapchains = &swapchain->swapchain.swapchain,
				.pImageIndices = &swapchainImageIndex,
			};
			auto presentResult = device->graphicsQueue.present(presentInfo);
			if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
				swapchainNeedsRebuild = true;
				continue;
			}
			vk::checkResult(presentResult, "Failed to present to queue");
		}

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
