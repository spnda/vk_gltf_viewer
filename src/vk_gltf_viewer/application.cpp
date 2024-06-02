#include <cassert>
#include <numbers>
#include <ranges>

#include <tracy/Tracy.hpp>

#include <vulkan/vk.hpp>
#include <vulkan/pipeline_builder.hpp>
#include <GLFW/glfw3.h> // After Vulkan includes so that it detects it

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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

		app.initVisbufferPass();
		app.initVisbufferResolvePass();
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

	drawBuffers.resize(frameOverlap);
	deletionQueue.push([this]() { drawBuffers.clear(); });

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
}

Application::~Application() noexcept {
	ZoneScoped;
	if (volkGetLoadedDevice() != VK_NULL_HANDLE) {
		for (auto& asset : loadedAssets)
			asset.reset();

		vkDeviceWaitIdle(*device);
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
			.width = swapchain->swapchain.extent.width,
			.height = swapchain->swapchain.extent.height,
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
			.width = swapchain->swapchain.extent.width,
			.height = swapchain->swapchain.extent.height,
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

		const auto colorAttachmentFormat = VK_FORMAT_R32_UINT;
		const auto depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
		const VkPipelineRenderingCreateInfo renderingCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &colorAttachmentFormat,
			.depthAttachmentFormat = depthAttachmentFormat,
		};
		const VkPipelineColorBlendAttachmentState blendAttachment{
			.blendEnable = VK_FALSE,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
							  VK_COLOR_COMPONENT_A_BIT,
		};

		auto builder = vk::GraphicsPipelineBuilder(*device, 1)
			.setPipelineLayout(0, visbufferPass.pipelineLayout)
			.pushPNext(0, &renderingCreateInfo)
			.addDynamicState(0, VK_DYNAMIC_STATE_SCISSOR)
			.addDynamicState(0, VK_DYNAMIC_STATE_VIEWPORT)
			.setBlendAttachment(0, &blendAttachment)
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

void Application::run() {
	ZoneScoped;

	bool swapchainNeedsRebuild = false;
	std::size_t currentFrame = 0;
	while (!glfwWindowShouldClose(window)) {
		if (!swapchainNeedsRebuild) {
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
				loadedAssets.emplace_back(task->getLoadedAsset());
				task.reset();
			}
		}
		std::erase_if(assetLoadTasks, [](std::shared_ptr<AssetLoadTask>& value) {
			return !bool(value);
		});

		auto currentTime = glfwGetTime();
		deltaTime = currentTime - lastFrame;
		lastFrame = currentTime;

		if (!freezeAnimations) {
			animationTime += deltaTime;
		}

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

		camera->updateCamera(currentFrame, window, deltaTime, swapchain->swapchain.extent);
		updateDrawBuffer(currentFrame);

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
		{
			std::array<VkImageMemoryBarrier2, 3> imageBarriers = {{
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
				}
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

			auto& drawBuffer = drawBuffers[currentFrame];
			auto totalMeshletCount = drawBuffer.meshletDrawBuffer
				? static_cast<std::uint32_t>(drawBuffer.meshletDrawBuffer->getBufferSize() / sizeof(glsl::MeshletDraw))
				: 0U;

			const VkRenderingAttachmentInfo visbufferAttachment {
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
			};
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
					.extent = swapchain->swapchain.extent,
				},
				.layerCount = 1,
				.colorAttachmentCount = 1,
				.pColorAttachments = &visbufferAttachment,
				.pDepthAttachment = &depthAttachment,
			};
			vkCmdBeginRendering(cmd, &renderingInfo);

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, visbufferPass.pipeline);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, visbufferPass.pipelineLayout,
									0, 1, &device->resourceTable->getSet(), 0, nullptr);

			// TODO: Add support for rendering multiple assets
			//       This will require a merged primitive buffer, or some adjustment in the MeshletDraw
			//       structure, to identify to which asset a primitive belongs.
			assert(loadedAssets.size() <= 1);
			const VkViewport viewport = {
				.x = 0.0F,
				.y = 0.0F,
				.width = static_cast<float>(swapchain->swapchain.extent.width),
				.height = static_cast<float>(swapchain->swapchain.extent.height),
				.minDepth = 0.0F,
				.maxDepth = 1.0F,
			};
			vkCmdSetViewport(cmd, 0, 1, &viewport);

			const VkRect2D scissor = renderingInfo.renderArea;
			vkCmdSetScissor(cmd, 0, 1, &scissor);

			if (totalMeshletCount > 0) {
				glsl::VisbufferPushConstants pushConstants {
					.drawBuffer = drawBuffers[currentFrame].meshletDrawBuffer->getDeviceAddress(),
					.meshletDrawCount = totalMeshletCount,
					.transformBuffer = drawBuffers[currentFrame].transformBuffer->getDeviceAddress(),
					.primitiveBuffer = loadedAssets.front()->primitiveBuffer->getDeviceAddress(),
					.cameraBuffer = camera->getCameraDeviceAddress(currentFrame),
					.materialBuffer = 0,
				};
				vkCmdPushConstants(cmd, visbufferPass.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof pushConstants,
								   &pushConstants);

				vkCmdDrawMeshTasksEXT(cmd, (totalMeshletCount + glsl::maxMeshlets - 1) / glsl::maxMeshlets, 1, 1);
			}

			vkCmdEndRendering(cmd);

			vkCmdEndDebugUtilsLabelEXT(cmd);
		}

		// Image barrier for visbuffer image from visbuffer pass -> resolve pass
		{
			const VkImageMemoryBarrier2 imageBarrier {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
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
			};
			const VkDependencyInfo dependencyInfo{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &imageBarrier,
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
				.outputImageHandle = swapchain->imageViewHandles[swapchainImageIndex],
			};
			vkCmdPushConstants(cmd, visbufferResolvePass.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof pushConstants, &pushConstants);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visbufferResolvePass.pipelineLayout,
						0, 1, &device->resourceTable->getSet(), 0, nullptr);

			vkCmdDispatch(cmd, swapchain->swapchain.extent.width / 32, swapchain->swapchain.extent.height, 1);

			vkCmdEndDebugUtilsLabelEXT(cmd);
		}

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

			auto extent = glm::u32vec2(swapchain->swapchain.extent.width, swapchain->swapchain.extent.height);
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
	}
}

void Application::renderUi() {
	ZoneScoped;
	if (ImGui::Begin("vk_gltf_viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
		ImGui::SeparatorText("Performance");

		ImGui::Text("Frametime: %.2f ms", deltaTime * 1000.);
		ImGui::Text("FPS: %.2f", 1. / deltaTime);
		ImGui::Text("AFPS: %.2f rad/s", 2. * std::numbers::pi_v<double> / deltaTime); // Angular FPS

		ImGui::SeparatorText("Camera");

		auto pos = camera->getPosition();
		ImGui::Text("Position: <%.2f, %.2f, %.2f>", pos.x, pos.y, pos.z);
		ImGui::Checkbox("Freeze Camera frustum", &camera->freezeCameraFrustum);

		ImGui::SeparatorText("Debug");

		ImGui::Checkbox("Freeze animations", &freezeAnimations);
	}
	ImGui::End();

	// ImGui::ShowDemoWindow();

	ImGui::Render();
}

void Application::iterateNode(fastgltf::Asset& asset, std::size_t nodeIndex, fastgltf::math::fmat4x4 parent,
							  std::function<void(fastgltf::Node&, const fastgltf::math::fmat4x4&)>& callback) {
	ZoneScoped;
	auto& node = asset.nodes[nodeIndex];

	// Compute the animations
	fastgltf::TRS trs = std::get<fastgltf::TRS>(node.transform); // We specify DecomposeNodeMatrices, so it should always be this.
	for (std::size_t ai = 0; auto& animation : asset.animations) {
		for (auto& channel : animation.channels) {
			if (!channel.nodeIndex.has_value() || *channel.nodeIndex != nodeIndex)
				continue;

			// TODO: I don't want to see a reference to loadedAsstes in here anymore.
			auto& sampler = loadedAssets.front()->animations[ai].samplers[channel.samplerIndex];
			switch (channel.path) {
				case fg::AnimationPath::Translation: {
					trs.translation = sampler.sample<fg::AnimationPath::Translation>(asset, static_cast<float>(animationTime));
					break;
				}
				case fg::AnimationPath::Scale: {
					trs.scale = sampler.sample<fg::AnimationPath::Scale>(asset, static_cast<float>(animationTime));
					break;
				}
				case fastgltf::AnimationPath::Rotation: {
					trs.rotation = sampler.sample<fg::AnimationPath::Rotation>(asset, static_cast<float>(animationTime));
					break;
				}
				case fastgltf::AnimationPath::Weights:
					break;
			}
		}
		++ai;
	}

	// Compute the matrix with the animated values
	auto matrix = parent
		* translate(fg::math::fmat4x4(), trs.translation)
		* fg::math::fmat4x4(asMatrix(trs.rotation))
		* scale(fg::math::fmat4x4(), trs.scale);

	callback(node, matrix);

	for (auto& child : node.children) {
		iterateNode(asset, child, matrix, callback);
	}
}

void Application::updateDrawBuffer(std::size_t currentFrame) {
	ZoneScoped;
	if (loadedAssets.empty())
		return;

	auto& drawBuffer = drawBuffers[currentFrame];
	VkDeviceSize currentDrawBufferSize = drawBuffer.meshletDrawBuffer ? drawBuffer.meshletDrawBuffer->getBufferSize() : 0;
	VkDeviceSize currentTransformBufferSize = drawBuffer.transformBuffer ? drawBuffer.meshletDrawBuffer->getBufferSize() : 0;

	std::vector<glsl::MeshletDraw> draws;
	draws.reserve(currentDrawBufferSize / sizeof(glsl::MeshletDraw));

	std::vector<glm::mat4> transforms;
	transforms.reserve(currentTransformBufferSize / sizeof(glm::mat4));

	auto& asset = loadedAssets.front();
	auto& scene = asset->asset.scenes[0];
	for (auto& nodes : scene.nodeIndices) {
		std::function<void(fg::Node&, const fg::math::fmat4x4&)> callback = [&](fg::Node& node, const fg::math::fmat4x4& matrix) {
			ZoneScoped;
			if (!node.meshIndex.has_value()) {
				return;
			}

			auto transformIndex = static_cast<std::uint32_t>(transforms.size());
			transforms.emplace_back(glm::make_mat4x4(matrix.data()));

			for (auto& primitive : asset->meshes[*node.meshIndex].primitiveIndices) {
				auto& buffers = asset->primitiveBuffers[primitive];
				for (std::uint32_t i = 0; i < buffers.meshletCount; ++i) {
					draws.emplace_back(glsl::MeshletDraw {
						.primitiveIndex = static_cast<std::uint32_t>(primitive),
						.meshletIndex = i,
						.transformIndex = transformIndex,
					});
				}
			}
		};
		iterateNode(asset->asset, nodes, fg::math::fmat4x4(1.f), callback);
	}

	constexpr auto bufferUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	constexpr VmaAllocationCreateInfo allocationCreateInfo = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	const auto requiredDrawBufferSize = draws.size() * sizeof(glsl::MeshletDraw);
	if (currentDrawBufferSize < requiredDrawBufferSize) {
		const VkBufferCreateInfo bufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = requiredDrawBufferSize,
			.usage = bufferUsage,
		};
		drawBuffer.meshletDrawBuffer = std::make_unique<ScopedBuffer>(*device, &bufferCreateInfo, &allocationCreateInfo);
		vk::setDebugUtilsName(*device, drawBuffer.meshletDrawBuffer->getHandle(), fmt::format("Meshlet draw buffer {}", currentFrame));
	}

	const auto requiredTransformBufferSize = transforms.size() * sizeof(glm::mat4);
	if (currentTransformBufferSize < requiredTransformBufferSize) {
		const VkBufferCreateInfo bufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = requiredTransformBufferSize,
			.usage = bufferUsage,
		};
		drawBuffer.transformBuffer = std::make_unique<ScopedBuffer>(*device, &bufferCreateInfo, &allocationCreateInfo);
		vk::setDebugUtilsName(*device, drawBuffer.transformBuffer->getHandle(), fmt::format("Transform buffer {}", currentFrame));
	}

	// TODO: This currently takes 2x longer than the visbuffer raster itself.
	//       This should probably be refactored so that the meshlet draw commands are built upfront, or whenever the scene changes.
	{
		ZoneScopedN("Draw buffer copy");
		ScopedMap drawMap(*drawBuffer.meshletDrawBuffer);
		std::memcpy(drawMap.get(), draws.data(), drawBuffer.meshletDrawBuffer->getBufferSize());
	}

	{
		ZoneScopedN("Transform buffer copy");
		ScopedMap transformMap(*drawBuffer.transformBuffer);
		std::memcpy(transformMap.get(), transforms.data(), drawBuffer.transformBuffer->getBufferSize());
	}
}
