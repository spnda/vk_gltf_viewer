#include <functional>
#include <iostream>
#include <string_view>

#include <TaskScheduler.h>

#include <vulkan/vk.hpp>
#include <VkBootstrap.h>
#include <vulkan/vma.hpp>

#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#include <vulkan/pipeline_builder.hpp>
#include <vulkan/debug_utils.hpp>

#include <imgui.h>
#include <imgui_stdlib.h>
#include <vk_gltf_viewer/imgui_renderer.hpp>

#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/base64.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

#include <meshoptimizer.h>

#include <vk_gltf_viewer/util.hpp>
#include <vk_gltf_viewer/viewer.hpp>
#include <vk_gltf_viewer/scheduler.hpp>

enki::TaskScheduler taskScheduler;

struct Viewer;

/** Replacement buffer data adapter for fastgltf which supports decompressing with EXT_meshopt_compression */
struct CompressedBufferDataAdapter {
	std::vector<std::optional<fastgltf::StaticVector<std::byte>>> decompressedBuffers;

	/** Get the data pointer of a loaded (possibly compressed) buffer */
	[[nodiscard]] static auto getData(const fastgltf::Buffer& buffer, std::size_t byteOffset, std::size_t byteLength) {
		using namespace fastgltf;
		return std::visit(visitor {
			[](auto&) -> span<const std::byte> {
				assert(false && "Tried accessing a buffer with no data, likely because no buffers were loaded. Perhaps you forgot to specify the LoadExternalBuffers option?");
				return {};
			},
			[](const sources::Fallback& fallback) -> span<const std::byte> {
				assert(false && "Tried accessing data of a fallback buffer.");
				return {};
			},
			[&](const sources::Array& array) -> span<const std::byte> {
				return span(reinterpret_cast<const std::byte*>(array.bytes.data()), array.bytes.size_bytes());
			},
			[&](const sources::Vector& vec) -> span<const std::byte> {
				return span(reinterpret_cast<const std::byte*>(vec.bytes.data()), vec.bytes.size());
			},
			[&](const sources::ByteView& bv) -> span<const std::byte> {
				return bv.bytes;
			},
		}, buffer.data).subspan(byteOffset, byteLength);
	}

	/** Decompress all buffer views and store them in this adapter */
	bool decompress(const fastgltf::Asset& asset) {
		ZoneScoped;
		using namespace fastgltf;

		decompressedBuffers.reserve(asset.bufferViews.size());
		for (auto& bufferView : asset.bufferViews) {
			if (!bufferView.meshoptCompression) {
				decompressedBuffers.emplace_back(std::nullopt);
				continue;
			}

			// This is a compressed buffer view.
			// For the original implementation, see https://github.com/jkuhlmann/cgltf/pull/129#issue-739550034
			auto& mc = *bufferView.meshoptCompression;
			fastgltf::StaticVector<std::byte> result(mc.count * mc.byteStride);

			// Get the data span from the compressed buffer.
			auto data = getData(asset.buffers[mc.bufferIndex], mc.byteOffset, mc.byteLength);

			int rc = -1;
			switch (mc.mode) {
				case MeshoptCompressionMode::Attributes: {
					rc = meshopt_decodeVertexBuffer(result.data(), mc.count, mc.byteStride,
													reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
				case MeshoptCompressionMode::Triangles: {
					rc = meshopt_decodeIndexBuffer(result.data(), mc.count, mc.byteStride,
											  reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
				case MeshoptCompressionMode::Indices: {
					rc = meshopt_decodeIndexSequence(result.data(), mc.count, mc.byteStride,
												reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
			}

			if (rc != 0)
				return false;

			switch (mc.filter) {
				case MeshoptCompressionFilter::None:
					break;
				case MeshoptCompressionFilter::Octahedral: {
					meshopt_decodeFilterOct(result.data(), mc.count, mc.byteStride);
					break;
				}
				case MeshoptCompressionFilter::Quaternion: {
					meshopt_decodeFilterQuat(result.data(), mc.count, mc.byteStride);
					break;
				}
				case MeshoptCompressionFilter::Exponential: {
					meshopt_decodeFilterExp(result.data(), mc.count, mc.byteStride);
					break;
				}
			}

			decompressedBuffers.emplace_back(std::move(result));
		}

		return true;
	}

	auto operator()(const fastgltf::Asset& asset, std::size_t bufferViewIdx) const {
		using namespace fastgltf;

		auto& bufferView = asset.bufferViews[bufferViewIdx];
		if (bufferView.meshoptCompression) {
			assert(decompressedBuffers.size() == asset.bufferViews.size());

			assert(decompressedBuffers[bufferViewIdx].has_value());
			return span(decompressedBuffers[bufferViewIdx]->data(), decompressedBuffers[bufferViewIdx]->size_bytes());
		}

		return getData(asset.buffers[bufferView.bufferIndex], bufferView.byteOffset, bufferView.byteLength);
	}
};

void glfwErrorCallback(int errorCode, const char* description) {
    if (errorCode != GLFW_NO_ERROR) {
		fmt::print(stderr, "GLFW error: {} {}\n", errorCode, description);
    }
}

void glfwResizeCallback(GLFWwindow* window, int width, int height) {
    if (width > 0 && height > 0) {
        auto* viewer = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
        viewer->rebuildSwapchain(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
    }
}

void cursorCallback(GLFWwindow* window, double xpos, double ypos) {
	void* ptr = glfwGetWindowUserPointer(window);
	auto& movement = static_cast<Viewer*>(ptr)->movement;

	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
	if (state != GLFW_PRESS) {
		movement.lastCursorPosition = { xpos, ypos };
		return;
	}

	if (movement.firstMouse) {
		movement.lastCursorPosition = { xpos, ypos };
		movement.firstMouse = false;
	}

	auto offset = glm::vec2(xpos - movement.lastCursorPosition.x, movement.lastCursorPosition.y - ypos);
	movement.lastCursorPosition = { xpos, ypos };
	offset *= 0.1f;

	movement.yaw   += offset.x;
	movement.pitch += offset.y;
	movement.pitch = glm::clamp(movement.pitch, -89.0f, 89.0f);

	auto& direction = movement.direction;
	direction.x = cos(glm::radians(movement.yaw)) * cos(glm::radians(movement.pitch));
	direction.y = sin(glm::radians(movement.pitch));
	direction.z = sin(glm::radians(movement.yaw)) * cos(glm::radians(movement.pitch));
	direction = glm::normalize(direction);
}

VkBool32 vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT          messageSeverity,
                             VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
                             const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
                             void*                                            pUserData) {
	fmt::print("{}\n", pCallbackData->pMessage);
	fmt::print("\tObjects: {}\n", pCallbackData->objectCount);
	for (std::size_t i = 0; i < pCallbackData->objectCount; ++i) {
		auto& obj = pCallbackData->pObjects[i];
		fmt::print("\t\t[{}] 0x{:x}, {}, {}\n", i, obj.objectHandle, obj.objectType, obj.pObjectName == nullptr ? "nullptr" : obj.pObjectName);
	}
    return VK_FALSE; // Beware: VK_TRUE here and the layers will kill the app instantly.
}

template <typename T>
void checkResult(vkb::Result<T> result) noexcept(false) {
    if (!result) {
        throw vulkan_error(result.error().message(), result.vk_result());
    }
}

void Viewer::setupVulkanInstance() {
	ZoneScoped;
	vk::checkResult(volkInitialize(), "No compatible Vulkan loader or driver found: {}");

    auto version = volkGetInstanceVersion();
    if (version < VK_API_VERSION_1_1) {
        throw std::runtime_error("The Vulkan loader only supports version 1.0.");
    }

    vkb::InstanceBuilder builder;

    // Enable GLFW extensions
    {
        std::uint32_t glfwExtensionCount = 0;
        const auto* glfwExtensionArray = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        builder.enable_extensions(glfwExtensionCount, glfwExtensionArray);
    }

    auto instanceResult = builder
            .set_app_name("vk_viewer")
            .require_api_version(1, 3, 0)
            .request_validation_layers()
            .set_debug_callback(vulkanDebugCallback)
            .build();
    checkResult(instanceResult);

    instance = instanceResult.value();
    deletionQueue.push([this]() {
        vkb::destroy_instance(instance);
    });

    volkLoadInstanceOnly(instance);
}

void Viewer::setupVulkanDevice() {
	ZoneScoped;
	const VkPhysicalDeviceFeatures vulkan10features {
		.multiDrawIndirect = VK_TRUE,
		.samplerAnisotropy = VK_TRUE,
	};

	const VkPhysicalDeviceVulkan11Features vulkan11Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.storageBuffer16BitAccess = VK_TRUE,
		.shaderDrawParameters = VK_TRUE,
	};

	const VkPhysicalDeviceVulkan12Features vulkan12Features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.storageBuffer8BitAccess = VK_TRUE,
		.shaderFloat16 = VK_TRUE,
		.shaderInt8 = VK_TRUE,
		.shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
		.descriptorBindingPartiallyBound = VK_TRUE,
		.runtimeDescriptorArray = VK_TRUE,
		.scalarBlockLayout = VK_TRUE,
#if defined(TRACY_ENABLE)
		.hostQueryReset = VK_TRUE,
#endif
		.timelineSemaphore = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };

	const VkPhysicalDeviceVulkan13Features vulkan13Features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
		.maintenance4 = VK_TRUE,
    };

    const VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
		.taskShader = VK_TRUE,
        .meshShader = VK_TRUE,
    };

	// Select an appropriate device with the given requirements.
    vkb::PhysicalDeviceSelector selector(instance);

    auto selectionResult = selector
            .set_surface(surface)
            .set_minimum_version(1, 3) // We want Vulkan 1.3.
			.set_required_features(vulkan10features)
			.set_required_features_11(vulkan11Features)
            .set_required_features_12(vulkan12Features)
            .set_required_features_13(vulkan13Features)
            .add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME)
#if TRACY_ENABLE
			.add_required_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
#endif
            .add_required_extension_features(meshShaderFeatures)
            .require_present()
            .require_dedicated_transfer_queue()
            .select();
    checkResult(selectionResult);

	VmaAllocatorCreateFlags allocatorFlags = 0;
	auto& physicalDevice = selectionResult.value();
	if (physicalDevice.enable_extension_if_present(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
		allocatorFlags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	// Generate the queue descriptions for vkb. Use one queue for everything except
	// for dedicated transfer queues.
	std::vector<vkb::CustomQueueDescription> queues;
	auto queueFamilies = physicalDevice.get_queue_families();
	for (std::uint32_t i = 0; i < queueFamilies.size(); i++) {
		std::size_t queueCount = 1;
		// Dedicated transfer queue; does not support graphics or present
		if ((queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && (queueFamilies[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0)
			queueCount = queueFamilies[i].queueCount;
		std::vector<float> priorities(queueCount, 1.0f);
		queues.emplace_back(i, std::move(priorities));
	}

	vkb::DeviceBuilder deviceBuilder(selectionResult.value());
    auto creationResult = deviceBuilder
			.custom_queue_setup(queues)
            .build();
    checkResult(creationResult);

    device = creationResult.value();
    deletionQueue.push([this]() {
        vkb::destroy_device(device);
    });

    volkLoadDevice(device);

#if defined(TRACY_ENABLE)
	// Initialize the tracy context
	tracyCtx = TracyVkContextHostCalibrated(device.physical_device, device, vkResetQueryPool, vkGetPhysicalDeviceCalibrateableTimeDomainsEXT, vkGetCalibratedTimestampsEXT);
	deletionQueue.push([this]() {
		DestroyVkContext(tracyCtx);
	});
#endif

	// Create the VMA allocator
	// Create the VMA allocator object
	const VmaVulkanFunctions vmaFunctions {
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
	};
	const VmaAllocatorCreateInfo allocatorInfo {
		.flags = allocatorFlags | VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = device.physical_device,
		.device = device,
		.pVulkanFunctions = &vmaFunctions,
		.instance = instance,
		.vulkanApiVersion = VK_API_VERSION_1_3,
	};
	auto result = vmaCreateAllocator(&allocatorInfo, &allocator);
	vk::checkResult(result, "Failed to create VMA allocator: {}");

	deletionQueue.push([this]() {
		vmaDestroyAllocator(allocator);
	});

	{
		// Get the main graphics queue
		auto graphicsQueueIndexResult = device.get_queue_index(vkb::QueueType::graphics);
		checkResult(graphicsQueueIndexResult);
		graphicsQueueFamily = graphicsQueueIndexResult.value();

		vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue.handle);
		graphicsQueue.lock = std::make_unique<std::mutex>();

		// Get the transfer queue handles
		auto transferQueueIndexRes = device.get_dedicated_queue_index(vkb::QueueType::transfer);
		checkResult(transferQueueIndexRes);

		transferQueueFamily = transferQueueIndexRes.value();
		transferQueues.resize(queueFamilies[transferQueueFamily].queueCount);
		for (std::size_t i = 0; auto& queue: transferQueues) {
			queue.lock = std::make_unique<std::mutex>();
			vkGetDeviceQueue(device, transferQueueFamily, i++, &queue.handle);
		}
	}

	// Create the transfer command pools
	auto threadCount = std::thread::hardware_concurrency();
	auto createPools = [&](std::vector<CommandPool>& pools, std::uint32_t queueFamily) {
		pools.resize(threadCount);
		for (auto& commandPool : pools) {
			const VkCommandPoolCreateInfo commandPoolInfo{
				.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
				.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
				.queueFamilyIndex = queueFamily,
			};
			auto createResult = vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool.pool);
			vk::checkResult(createResult, "Failed to allocate buffer upload command pool: {}");

			const VkCommandBufferAllocateInfo allocateInfo {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = commandPool.pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};
			auto allocateResult = vkAllocateCommandBuffers(device, &allocateInfo, &commandPool.buffer);
			vk::checkResult(allocateResult, "Failed to allocate buffer upload command buffers: {}");
		}
	};
	createPools(uploadCommandPools, transferQueueFamily);
	createPools(graphicsCommandPools, graphicsQueueFamily);

	deletionQueue.push([this]() {
		for (auto& cmdPool : uploadCommandPools)
			vkDestroyCommandPool(device, cmdPool.pool, nullptr);
		for (auto& cmdPool : graphicsCommandPools)
			vkDestroyCommandPool(device, cmdPool.pool, nullptr);
	});

	fencePool.init(device);
	deletionQueue.push([this]() {
		fencePool.destroy();
	});
}

void Viewer::rebuildSwapchain(std::uint32_t width, std::uint32_t height) {
	ZoneScoped;
    vkb::SwapchainBuilder swapchainBuilder(device);
    auto swapchainResult = swapchainBuilder
            .set_old_swapchain(swapchain)
			.set_desired_extent(width, height)
			.set_desired_min_image_count(frameOverlap)
            .build();
    checkResult(swapchainResult);

	// We delay the destruction of the old swapchain, its views, and the depth image until all presents related to this have finished.
	// TODO: Is there no nicer way of declaring these captures?
	timelineDeletionQueue.push([this, copy = swapchain, views = swapchainImageViews]() {
		for (auto& view : views)
			vkDestroyImageView(device, view, nullptr);
		vkb::destroy_swapchain(copy);
	});
	timelineDeletionQueue.push([this, image = depthImage, allocation = depthImageAllocation, view = depthImageView] {
		if (image != VK_NULL_HANDLE) {
			vkDestroyImageView(device, view, VK_NULL_HANDLE);
			vmaDestroyImage(allocator, image, allocation);
		}
	});

	swapchainImageViews.clear();
    swapchain = swapchainResult.value();

	swapchainImages = std::move(vk::enumerateVector<VkImage, decltype(swapchainImages)>(vkGetSwapchainImagesKHR, device, swapchain));

    auto imageViewResult = swapchain.get_image_views();
    checkResult(imageViewResult);
	for (auto& view : imageViewResult.value())
		swapchainImageViews.emplace_back(view);

	const VmaAllocationCreateInfo allocationInfo {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	const VkImageCreateInfo imageInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_D32_SFLOAT,
		.extent = {
			.width = width,
			.height = height,
			.depth = 1,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	auto result = vmaCreateImage(allocator, &imageInfo, &allocationInfo, &depthImage, &depthImageAllocation, VK_NULL_HANDLE);
	vk::checkResult(result, "Failed to create depth image: {}");
	vk::setDebugUtilsName(device, depthImage, "Depth image");
	vk::setAllocationName(allocator, depthImageAllocation, "Depth image allocation");

	const VkImageViewCreateInfo imageViewInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = depthImage,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = imageInfo.format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		}
	};
	result = vkCreateImageView(device, &imageViewInfo, VK_NULL_HANDLE, &depthImageView);
	vk::checkResult(result, "Failed to create depth image view: {}");
	vk::setDebugUtilsName(device, depthImageView, "Depth image view");
}

void Viewer::createDescriptorPool() {
	ZoneScoped;
	auto& limits = device.physical_device.properties.limits;

	// TODO: Do we want to always just use the *mega* descriptor pool?
	std::array<VkDescriptorPoolSize, 3> sizes = {{
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = util::min(1048576U, limits.maxDescriptorSetUniformBuffers),
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = util::min(1048576U, limits.maxDescriptorSetStorageBuffers),
		},
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = util::min(16536U, limits.maxDescriptorSetSampledImages),
		}
	}};
	const VkDescriptorPoolCreateInfo poolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 50, // TODO ?
		.poolSizeCount = static_cast<std::uint32_t>(sizes.size()),
		.pPoolSizes = sizes.data(),
	};
	auto result = vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool);
	vk::checkResult(result, "Failed to create descriptor pool: {}");

	deletionQueue.push([&]() {
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
	});
}

void Viewer::buildCameraDescriptor() {
	ZoneScoped;
	// The camera descriptor layout
	std::array<VkDescriptorSetLayoutBinding, 1> layoutBindings = {{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT,
		},
	}};
	const VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = static_cast<std::uint32_t>(layoutBindings.size()),
		.pBindings = layoutBindings.data(),
	};
	auto result = vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfo,
											  VK_NULL_HANDLE, &cameraSetLayout);
	vk::checkResult(result, "Failed to create camera descriptor set layout: {}");
	vk::setDebugUtilsName(device, cameraSetLayout, "Camera descriptor layout");

	deletionQueue.push([&]() {
		vkDestroyDescriptorSetLayout(device, cameraSetLayout, nullptr);
	});

	// Allocate frameOverlap descriptor sets used for the camera buffer.
	// We update their contents at the end of the function
	std::vector<VkDescriptorSetLayout> setLayouts(frameOverlap, cameraSetLayout);
	const VkDescriptorSetAllocateInfo allocateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = static_cast<std::uint32_t>(setLayouts.size()),
		.pSetLayouts = setLayouts.data(),
	};

	// Allocate the sets. We copy each member of the sets vector in the loop below.
	std::vector<VkDescriptorSet> sets(frameOverlap);
	result = vkAllocateDescriptorSets(device, &allocateInfo, sets.data());
	vk::checkResult(result, "Failed to allocate camera descriptor sets: {}");

	// Generate descriptor writes to update the descriptor
	std::array<VkDescriptorBufferInfo, frameOverlap> bufferInfos {};
	std::array<VkWriteDescriptorSet, frameOverlap> descriptorWrites {};
	cameraBuffers.resize(frameOverlap);

	for (std::size_t i = 0; auto& cameraBuffer : cameraBuffers) {
		// Copy the created camera sets into the every cameraBuffer structs.
		cameraBuffer.cameraSet = sets[i];

		// Create the camera buffers
		const VmaAllocationCreateInfo allocationCreateInfo {
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO,
		};
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(glsl::Camera),
			.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		};
		result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
									  &cameraBuffer.handle, &cameraBuffer.allocation, VK_NULL_HANDLE);
		vk::checkResult(result, "Failed to allocate camera buffer: {}");
		vk::setDebugUtilsName(device, cameraBuffer.handle, fmt::format("Camera buffer {}", i));

		deletionQueue.push([&]() {
			vmaDestroyBuffer(allocator, cameraBuffer.handle, cameraBuffer.allocation);
		});

		// Initialise the camera descriptor set
		const VkDescriptorBufferInfo bufferInfo = {
			.buffer = cameraBuffer.handle,
			.offset = 0,
			.range = VK_WHOLE_SIZE,
		};
		bufferInfos[i] = bufferInfo;

		const VkWriteDescriptorSet descriptorWrite = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = cameraBuffer.cameraSet,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &bufferInfos[i],
		};
		descriptorWrites[i] = descriptorWrite;
		++i;
	}

	// Update the descriptors
	vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(descriptorWrites.size()),
						   descriptorWrites.data(), 0, nullptr);
}

void Viewer::buildMeshPipeline() {
	ZoneScoped;
    // Build the mesh pipeline layout
    std::array<VkDescriptorSetLayout, 3> layouts {{ cameraSetLayout, meshletSetLayout, materialSetLayout }};
    const VkPipelineLayoutCreateInfo layoutCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 0,
    };
    auto result = vkCreatePipelineLayout(device, &layoutCreateInfo, VK_NULL_HANDLE, &meshPipelineLayout);
	vk::checkResult(result, "Failed to create mesh pipeline layout: {}");
	vk::setDebugUtilsName(device, meshPipelineLayout, "Mesh shading pipeline layout");

    // Load the mesh pipeline shaders
	// TODO: Check return value of loadShaderModule
    VkShaderModule fragModule, meshModule, taskModule;
    vk::loadShaderModule("main.frag.glsl.spv", device, &fragModule);
    vk::loadShaderModule("main.mesh.glsl.spv", device, &meshModule);
	vk::loadShaderModule("main.task.glsl.spv", device, &taskModule);

	// Load AABB visualizer shaders
	VkShaderModule aabbFragModule, aabbVertModule;
	vk::loadShaderModule("aabb_visualizer.frag.glsl.spv", device, &aabbFragModule);
	vk::loadShaderModule("aabb_visualizer.vert.glsl.spv", device, &aabbVertModule);

	// Build the mesh pipeline
    const auto colorAttachmentFormat = swapchain.image_format;
	const auto depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    const VkPipelineRenderingCreateInfo renderingCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &colorAttachmentFormat,
			.depthAttachmentFormat = depthAttachmentFormat,
    };

    const VkPipelineColorBlendAttachmentState blendAttachment {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

	// We specialize the task shader to have a local workgroup size which is exactly the subgroup size,
	// to efficiently use subgroup intrinsics for counting the total number of passed meshlets.
	VkPhysicalDeviceVulkan11Properties vulkan11Properties {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
	};
	VkPhysicalDeviceProperties2 properties {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &vulkan11Properties,
	};
	vkGetPhysicalDeviceProperties2(device.physical_device, &properties);
	const VkSpecializationMapEntry taskSubgroupSizeSpecMapEntry {
		.constantID = 0,
		.offset = 0,
		.size = sizeof(decltype(vulkan11Properties.subgroupSize)),
	};
	const VkSpecializationInfo taskSubgroupSizeSpecialization {
		.mapEntryCount = 1,
		.pMapEntries = &taskSubgroupSizeSpecMapEntry,
		.dataSize = taskSubgroupSizeSpecMapEntry.size,
		.pData = &vulkan11Properties.subgroupSize,
	};

	// Create mesh and AABB visualizing pipeline
	// TODO: glTF does not necessarily have a CCW winding order. Specifically, the spec says this:
	//       If the determinant of the transform is a negative value, the winding order of the mesh triangle faces should be reversed.
	//       This supports negative scales for mirroring geometry.
    auto builder = vk::GraphicsPipelineBuilder(device, 2)
        .setPipelineLayout(0, meshPipelineLayout)
        .pushPNext(0, &renderingCreateInfo)
        .addDynamicState(0, VK_DYNAMIC_STATE_SCISSOR)
        .addDynamicState(0, VK_DYNAMIC_STATE_VIEWPORT)
		.setBlendAttachment(0, &blendAttachment)
        .setTopology(0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setDepthState(0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL)
        .setRasterState(0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .setMultisampleCount(0, VK_SAMPLE_COUNT_1_BIT)
        .setScissorCount(0, 1U)
        .setViewportCount(0, 1U)
        .addShaderStage(0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule)
        .addShaderStage(0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule)
		.addShaderStage(0, VK_SHADER_STAGE_TASK_BIT_EXT, taskModule, "main", &taskSubgroupSizeSpecialization);

	const VkPipelineRenderingCreateInfo aabbRenderingCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &colorAttachmentFormat,
		.depthAttachmentFormat = depthAttachmentFormat,
	};

	// We want the AABBs to appear slightly transparent, which is why we need blending.
	// This just essentially just multiplies the fragment shader's value with its alpha value.
	const VkPipelineColorBlendAttachmentState aabbBlendAttachment {
		.blendEnable = VK_TRUE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	// Create AABB pipeline
	builder
		.setPipelineLayout(1, meshPipelineLayout)
		.pushPNext(1, &aabbRenderingCreateInfo)
		.addDynamicState(1, VK_DYNAMIC_STATE_SCISSOR)
		.addDynamicState(1, VK_DYNAMIC_STATE_VIEWPORT)
		.setBlendAttachment(1, &aabbBlendAttachment)
		.setTopology(1, VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
		.setDepthState(1, VK_TRUE, VK_TRUE, VK_COMPARE_OP_GREATER_OR_EQUAL)
		.setRasterState(1, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
		.setMultisampleCount(1, VK_SAMPLE_COUNT_1_BIT)
		.setScissorCount(1, 1U)
		.setViewportCount(1, 1U)
		.addShaderStage(1, VK_SHADER_STAGE_FRAGMENT_BIT, aabbFragModule, "main")
		.addShaderStage(1, VK_SHADER_STAGE_VERTEX_BIT, aabbVertModule, "main");

	std::array<VkPipeline, 2> pipelines {};
    result = builder.build(pipelines.data());
	vk::checkResult(result, "Failed to create mesh and aabb visualizing pipeline: {}");
	vk::setDebugUtilsName(device, meshPipeline, "Mesh pipeline");
	vk::setDebugUtilsName(device, aabbVisualizingPipeline, "AABB Visualization pipeline");

	meshPipeline = pipelines[0];
	aabbVisualizingPipeline = pipelines[1];

	// We don't need the shader modules after creating the pipeline anymore.
    vkDestroyShaderModule(device, fragModule, VK_NULL_HANDLE);
    vkDestroyShaderModule(device, meshModule, VK_NULL_HANDLE);
	vkDestroyShaderModule(device, taskModule, VK_NULL_HANDLE);
	vkDestroyShaderModule(device, aabbFragModule, VK_NULL_HANDLE);
	vkDestroyShaderModule(device, aabbVertModule, VK_NULL_HANDLE);

    deletionQueue.push([this]() {
        vkDestroyPipeline(device, meshPipeline, VK_NULL_HANDLE);
        vkDestroyPipelineLayout(device, meshPipelineLayout, VK_NULL_HANDLE);
		vkDestroyPipeline(device, aabbVisualizingPipeline, VK_NULL_HANDLE);
    });
}

void Viewer::createFrameData() {
	ZoneScoped;
    frameSyncData.resize(frameOverlap);
    for (auto& frame : frameSyncData) {
        constexpr VkSemaphoreCreateInfo semaphoreCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
        auto semaphoreResult = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.imageAvailable);
		vk::checkResult(semaphoreResult, "Failed to create image semaphore: {}");
		vk::setDebugUtilsName(device, frame.imageAvailable, "Image acquire semaphore");

        semaphoreResult = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.renderingFinished);
		vk::checkResult(semaphoreResult, "Failed to create rendering semaphore: {}");
		vk::setDebugUtilsName(device, frame.renderingFinished, "Rendering finished semaphore");

		constexpr VkFenceCreateInfo fenceCreateInfo {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
        auto fenceResult = vkCreateFence(device, &fenceCreateInfo, nullptr, &frame.presentFinished);
		vk::checkResult(fenceResult, "Failed to create present fence: {}");
		vk::setDebugUtilsName(device, frame.presentFinished, "Present fence");
    }
	deletionQueue.push([this]() {
		for (auto& frame : frameSyncData) {
			vkDestroyFence(device, frame.presentFinished, nullptr);
			vkDestroySemaphore(device, frame.renderingFinished, nullptr);
			vkDestroySemaphore(device, frame.imageAvailable, nullptr);
		}
	});

    frameCommandPools.resize(frameOverlap);
    for (auto& frame : frameCommandPools) {
        const VkCommandPoolCreateInfo commandPoolInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			// .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = graphicsQueueFamily,
		};
        auto createResult = vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frame.pool);
		vk::checkResult(createResult, "Failed to create frame command pool: {}");

        const VkCommandBufferAllocateInfo allocateInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = frame.pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
        frame.commandBuffers.resize(1);
        auto allocateResult = vkAllocateCommandBuffers(device, &allocateInfo, frame.commandBuffers.data());
		vk::checkResult(allocateResult, "Failed to allocate frame command buffers: {}");
    }
	deletionQueue.push([this]() {
		for (auto& frame : frameCommandPools)
			vkDestroyCommandPool(device, frame.pool, nullptr);
	});
}

class Base64DecodeTask final : public enki::ITaskSet {
    std::string_view encodedData;
	std::uint8_t* outputData;

public:
    // Arbitrarily chosen 1MB. Lower values will cause too many tasks to spawn, slowing down the process.
    // Perhaps even larger values would be necessary, as even this gets decoded incredibly quick and the
    // overhead of launching threaded tasks gets noticeable.
    static constexpr const size_t minBase64DecodeSetSize = 1 * 1024 * 1024; // 1MB.

    explicit Base64DecodeTask(std::uint32_t dataSize, std::string_view encodedData, std::uint8_t* outputData)
            : enki::ITaskSet(dataSize, minBase64DecodeSetSize), encodedData(encodedData), outputData(outputData) {}

    void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override {
		ZoneScoped;
        fastgltf::base64::decode_inplace(encodedData.substr(static_cast<std::size_t>(range.start) * 4, static_cast<std::size_t>(range.end) * 4),
                                         &outputData[range.start * 3], 0);
    }
};

// The custom base64 callback for fastgltf to multithread base64 decoding, to divide the (possibly) large
// input buffer into smaller chunks that can be worked on by multiple threads.
void multithreadedBase64Decoding(std::string_view encodedData, std::uint8_t* outputData,
                                 std::size_t padding, std::size_t outputSize, void* userPointer) {
	ZoneScoped;
    assert(fastgltf::base64::getOutputSize(encodedData.size(), padding) <= outputSize);
    assert(userPointer != nullptr);
    assert(encodedData.size() % 4 == 0);

    // Check if the data is smaller than minBase64DecodeSetSize, and if so just decode it on the main thread.
    // TaskSetPartition start and end is currently an uint32_t, so we'll check if we exceed that for safety.
    if (encodedData.size() < Base64DecodeTask::minBase64DecodeSetSize
        || encodedData.size() > std::numeric_limits<decltype(enki::TaskSetPartition::start)>::max()) {
        fastgltf::base64::decode_inplace(encodedData, outputData, padding);
        return;
    }

    // We divide by 4 to essentially create as many sets as there are decodable base64 blocks.
    Base64DecodeTask task(encodedData.size() / 4, encodedData, outputData);
    taskScheduler.AddTaskSetToPipe(&task);

    // Finally, wait for all other tasks to finish. enkiTS will use this thread as well to process the tasks.
    taskScheduler.WaitforTask(&task);
}

void Viewer::loadGltf(const std::filesystem::path& filePath) {
	ZoneScoped;
    fastgltf::GltfDataBuffer fileBuffer;
    if (!fileBuffer.loadFromFile(filePath)) {
        throw std::runtime_error("Failed to load file");
    }

	static constexpr auto supportedExtensions = fastgltf::Extensions::KHR_mesh_quantization
		| fastgltf::Extensions::KHR_lights_punctual
		| fastgltf::Extensions::KHR_materials_variants
		| fastgltf::Extensions::KHR_texture_transform
		| fastgltf::Extensions::KHR_texture_basisu
		| fastgltf::Extensions::EXT_meshopt_compression
		| fastgltf::Extensions::MSFT_texture_dds;

    fastgltf::Parser parser(supportedExtensions);
    parser.setUserPointer(this);
    parser.setBase64DecodeCallback(multithreadedBase64Decoding);

	// TODO: Extract buffer/image loading into async functions in the future
	static constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers
		| fastgltf::Options::LoadExternalBuffers
		| fastgltf::Options::LoadExternalImages
		| fastgltf::Options::GenerateMeshIndices;

    auto expected = parser.loadGltf(&fileBuffer, filePath.parent_path(), gltfOptions);
    if (expected.error() != fastgltf::Error::None) {
        auto message = fastgltf::getErrorMessage(expected.error());
        throw std::runtime_error(std::string("Failed to load glTF: ") + std::string(message));
    }

    asset = std::move(expected.get());

    // We'll always do additional validation
    if (auto validation = fastgltf::validate(asset); validation != fastgltf::Error::None) {
        auto message = fastgltf::getErrorMessage(validation);
        throw std::runtime_error(std::string("Asset failed validation") + std::string(message));
    }
}

/** Processes the primitives of every mesh and generates meshlet data */
struct PrimitiveProcessingTask : enki::ITaskSet {
	Viewer* viewer;
	GlobalMeshData& meshData;
	CompressedBufferDataAdapter& adapter;

	explicit PrimitiveProcessingTask(Viewer* viewer, GlobalMeshData& data, CompressedBufferDataAdapter& adapter) : viewer(viewer), meshData(data), adapter(adapter) {
		m_SetSize = viewer->asset.meshes.size();
	}

	glm::vec3 getMinMax(decltype(fastgltf::Accessor::min)& values) {
		return std::visit(fastgltf::visitor {
			[](auto& arg) {
				return glm::vec3();
			},
			[&](FASTGLTF_STD_PMR_NS::vector<double>& values) {
				assert(values.size() == 3);
				return glm::fvec3(values[0], values[1], values[2]);
			},
			[&](FASTGLTF_STD_PMR_NS::vector<std::int64_t>& values) {
				assert(values.size() == 3);
				return glm::fvec3(values[0], values[1], values[2]);
			},
		}, values);
	}

	void loadPrimitive(Mesh& mesh, fastgltf::Primitive& gltfPrimitive) {
		ZoneScoped;
		// These cases are possible in code, but cannot happen in reality.
		auto* positionIt = gltfPrimitive.findAttribute("POSITION");
		assert(positionIt != gltfPrimitive.attributes.end());
		assert(gltfPrimitive.indicesAccessor.has_value());

		auto& primitive = mesh.primitives.emplace_back();
		if (gltfPrimitive.materialIndex.has_value()) {
			primitive.materialIndex = gltfPrimitive.materialIndex.value() + Viewer::numDefaultMaterials;
		} else {
			gltfPrimitive.materialIndex = 0;
		}

		// Copy the positions and indices
		auto& posAccessor = viewer->asset.accessors[positionIt->second];

		// Get AABB for the entire primitive to quantize positions
		auto primitiveMin = getMinMax(posAccessor.min);
		auto primitiveMax = getMinMax(posAccessor.max);

		// Read vertices.
		std::vector<glsl::Vertex> vertices; vertices.reserve(posAccessor.count);
		fastgltf::iterateAccessor<glm::vec3>(viewer->asset, posAccessor, [&](glm::vec3 val) {
			auto& vertex = vertices.emplace_back();
			vertex.position = val;
			vertex.color = glm::vec4(1.0f);
			vertex.uv = {
				meshopt_quantizeHalf(0.0f),
				meshopt_quantizeHalf(0.0f),
			};
		}, adapter);

		auto& indicesAccessor = viewer->asset.accessors[gltfPrimitive.indicesAccessor.value()];
		std::vector<std::uint32_t> indices(indicesAccessor.count);
		fastgltf::copyFromAccessor<std::uint32_t>(viewer->asset, indicesAccessor, indices.data(), adapter);

		if (auto* colorAttribute = gltfPrimitive.findAttribute("COLOR_0"); colorAttribute != gltfPrimitive.attributes.end()) {
			// The glTF spec allows VEC3 and VEC4 for COLOR_n, with VEC3 data having to be extended with 1.0f for the fourth component.
			auto& colorAccessor = viewer->asset.accessors[colorAttribute->second];
			if (colorAccessor.type == fastgltf::AccessorType::Vec4) {
				fastgltf::iterateAccessorWithIndex<glm::vec4>(viewer->asset, viewer->asset.accessors[colorAttribute->second], [&](glm::vec4 val, std::size_t idx) {
					vertices[idx].color = val;
				}, adapter);
			} else if (colorAccessor.type == fastgltf::AccessorType::Vec3) {
				fastgltf::iterateAccessorWithIndex<glm::vec3>(viewer->asset, viewer->asset.accessors[colorAttribute->second], [&](glm::vec3 val, std::size_t idx) {
					vertices[idx].color = glm::vec4(val, 1.0f);
				}, adapter);
			}
		}

		if (auto* uvAttribute = gltfPrimitive.findAttribute("TEXCOORD_0"); uvAttribute != gltfPrimitive.attributes.end()) {
			fastgltf::iterateAccessorWithIndex<glm::vec2>(viewer->asset, viewer->asset.accessors[uvAttribute->second], [&](glm::vec2 val, std::size_t idx) {
				vertices[idx].uv = {
					meshopt_quantizeHalf(val.x),
					meshopt_quantizeHalf(val.y),
				};
			}, adapter);
		}

		// TODO: These are the optimal values for NVIDIA. What about the others?
		const std::size_t maxVertices = 64;
		const std::size_t maxTriangles = 124; // NVIDIA wants 126 but meshopt only allows 124 for alignment reasons.
		const float coneWeight = 0.0f; // We leave this as 0 because we're not using cluster cone culling.

		std::size_t maxMeshlets = meshopt_buildMeshletsBound(indicesAccessor.count, maxVertices, maxTriangles);
		std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
		std::vector<std::uint32_t> meshlet_vertices(maxMeshlets * maxVertices);
		std::vector<std::uint8_t> meshlet_triangles(maxMeshlets * maxTriangles * 3);

		// Generate the meshlets for this primitive
		primitive.meshlet_count = meshopt_buildMeshlets(
			meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
			indices.data(), indices.size(),
			&vertices[0].position.x, vertices.size(), sizeof(decltype(vertices)::value_type),
			maxVertices, maxTriangles, coneWeight);

		// Trim the buffers
		const auto& lastMeshlet = meshlets[primitive.meshlet_count - 1];
		meshlet_vertices.resize(lastMeshlet.vertex_count + lastMeshlet.vertex_offset);
		meshlet_triangles.resize(((lastMeshlet.triangle_count * 3 + 3) & ~3) + lastMeshlet.triangle_offset);
		meshlets.resize(primitive.meshlet_count);

		std::vector<glsl::Meshlet> finalMeshlets; finalMeshlets.reserve(primitive.meshlet_count);
		for (auto& meshlet : meshlets) {
			// Compute AABB bounds
			auto& initialVertex = vertices[meshlet_vertices[meshlet.vertex_offset]];
			auto min = glm::vec3(initialVertex.position), max = glm::vec3(initialVertex.position);

			for (std::size_t i = 1; i < meshlet.vertex_count; ++i) {
				std::uint32_t vertexIndex = meshlet_vertices[meshlet.vertex_offset + i];
				auto& vertex = vertices[vertexIndex];

				if (min.x > vertex.position.x)
					min.x = vertex.position.x;
				if (min.y > vertex.position.y)
					min.y = vertex.position.y;
				if (min.z > vertex.position.z)
					min.z = vertex.position.z;

				if (max.x < vertex.position.x)
					max.x = vertex.position.x;
				if (max.y < vertex.position.y)
					max.y = vertex.position.y;
				if (max.z < vertex.position.z)
					max.z = vertex.position.z;
			}

			assert(meshlet.vertex_count <= std::numeric_limits<std::uint8_t>::max());
			assert(meshlet.triangle_count <= std::numeric_limits<std::uint8_t>::max());
			glm::vec3 center = (min + max) * 0.5f;
			finalMeshlets.emplace_back(glsl::Meshlet {
				.vertexOffset = meshlet.vertex_offset,
				.triangleOffset = meshlet.triangle_offset,
				.vertexCount = static_cast<std::uint8_t>(meshlet.vertex_count),
				.triangleCount = static_cast<std::uint8_t>(meshlet.triangle_count),
				.aabbExtents = max - center,
				.aabbCenter = center,
			});
		}

		{
			std::lock_guard lock(meshData.lock);

			primitive.descOffset = meshData.globalMeshlets.size();
			primitive.vertexIndicesOffset = meshData.globalMeshletVertices.size();
			primitive.triangleIndicesOffset = meshData.globalMeshletTriangles.size();
			primitive.verticesOffset = meshData.globalVertices.size();

			// Append the data to the end of the global buffers.
			meshData.globalVertices.insert(meshData.globalVertices.end(), vertices.begin(), vertices.end());
			meshData.globalMeshlets.insert(meshData.globalMeshlets.end(), finalMeshlets.begin(), finalMeshlets.end());
			meshData.globalMeshletVertices.insert(meshData.globalMeshletVertices.end(), meshlet_vertices.begin(), meshlet_vertices.end());
			meshData.globalMeshletTriangles.insert(meshData.globalMeshletTriangles.end(), meshlet_triangles.begin(), meshlet_triangles.end());
		}
	}

	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override {
		ZoneScoped;
		for (auto i = range.start; i < range.end; ++i) {
			auto& gltfMesh = viewer->asset.meshes[i];
			auto& mesh = viewer->meshes[i];

			mesh.primitives.reserve(gltfMesh.primitives.size());
			for (auto& gltfPrimitive : gltfMesh.primitives) {
				loadPrimitive(mesh, gltfPrimitive);
			}
		}
	}
};

void Viewer::loadGltfMeshes() {
	ZoneScoped;
	// The meshlet descriptor layout
	std::array<VkDescriptorSetLayoutBinding, 5> layoutBindings = {{
		// Meshlet descriptions
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT,
		},
		// Vertex indices
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
		},
		// Primitive indices
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
		},
		// Vertices
		{
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
		},
		// The (indirect) draw commands
		{
			.binding = 4,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT,
		}
	}};
	const VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = static_cast<std::uint32_t>(layoutBindings.size()),
		.pBindings = layoutBindings.data(),
	};
	auto result = vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfo,
											  VK_NULL_HANDLE, &meshletSetLayout);
	vk::checkResult(result, "Failed to create meshlet descriptor set layout: {}");
	vk::setDebugUtilsName(device, meshletSetLayout, "Mesh shader pipeline descriptor layout");

	deletionQueue.push([this]() {
		vkDestroyDescriptorSetLayout(device, meshletSetLayout, nullptr);
	});

	GlobalMeshData globalMeshData;

	// Create the compressed adapter, which decompresses all buffer views using EXT_meshopt_compression.
	// All other data is passed along as usual.
	CompressedBufferDataAdapter adapter;
	if (!adapter.decompress(asset))
		throw std::runtime_error("Failed to decompress all glTF buffers");

	meshes.resize(asset.meshes.size());
	PrimitiveProcessingTask task(this, globalMeshData, adapter);
	taskScheduler.AddTaskSetToPipe(&task);
	taskScheduler.WaitforTask(&task);

	uploadMeshlets(globalMeshData);
}

VkResult Viewer::createGpuTransferBuffer(std::size_t byteSize, VkBuffer *buffer, VmaAllocation *allocation) const noexcept {
	const VmaAllocationCreateInfo allocationCreateInfo {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	const VkBufferCreateInfo bufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = byteSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	return vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
						   buffer, allocation, VK_NULL_HANDLE);
}

VkResult Viewer::createHostStagingBuffer(std::size_t byteSize, VkBuffer* buffer, VmaAllocation* allocation) const noexcept {
	// Using HOST_ACCESS_SEQUENTIAL_WRITE and adding TRANSFER_SRC to usage will make this buffer
	// either land in BAR scope, or in system memory. In either case, this is sufficient for a staging buffer.
	const VmaAllocationCreateInfo allocationCreateInfo {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
	};
	const VkBufferCreateInfo bufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = byteSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	};
	return vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
						   buffer, allocation, VK_NULL_HANDLE);
}

void Viewer::uploadBufferToDevice(std::span<const std::byte> bytes, VkBuffer* bufferHandle, VmaAllocation* allocationHandle) {
	ZoneScoped;
	// Create the destination buffer
	auto result = createGpuTransferBuffer(bytes.size_bytes(), bufferHandle, allocationHandle);
	vk::checkResult(result, "Failed to create GPU transfer storage buffer: {}");

	// Create the staging buffer and map it.
	VkBuffer stagingBuffer;
	VmaAllocation stagingAllocation;
	result = createHostStagingBuffer(bytes.size_bytes(), &stagingBuffer, &stagingAllocation);
	vk::checkResult(result, "Failed to create host staging buffer: {}");

	{
		vk::ScopedMap map(allocator, stagingAllocation);
		std::memcpy(map.get(), bytes.data(), bytes.size_bytes());
	}

	// Reset fences and command buffers.
	auto cmd = uploadCommandPools[taskScheduler.GetThreadNum()].buffer;
	vkResetCommandBuffer(cmd, 0);
	auto fence = fencePool.acquire();
	vkResetFences(device, 1, &fence->handle);

	const VkCommandBufferBeginInfo beginInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &beginInfo);

	// Perform the simple copy
	const VkBufferCopy region {
		.srcOffset = 0,
		.dstOffset = 0,
		.size = bytes.size_bytes(),
	};
	vkCmdCopyBuffer(cmd, stagingBuffer, *bufferHandle, 1, &region);

	vkEndCommandBuffer(cmd);

	auto& queue = getNextTransferQueueHandle();
	{
		// We need to guard the vkQueueSubmit call
		std::lock_guard lock(*queue.lock);

		// Submit the command buffer
		const VkPipelineStageFlags submitWaitStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		const VkSubmitInfo submitInfo {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pWaitDstStageMask = &submitWaitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &cmd,
		};
		auto submitResult = vkQueueSubmit(queue.handle, 1, &submitInfo, fence->handle);
		vk::checkResult(submitResult, "Failed to submit buffer copy: {}");
	}

	// We always wait for this operation to complete here, to free up the command buffer and fence for the next iteration.
	fencePool.wait_and_free(fence);

	vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
}

void Viewer::uploadImageToDevice(std::size_t stagingBufferSize, const std::function<void(VkCommandBuffer, VkBuffer, VmaAllocation)>& commands) {
	ZoneScoped;
	// Create a host allocation to hold our image data.
	const VmaAllocationCreateInfo allocationCreateInfo {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
	};
	const VkBufferCreateInfo bufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = stagingBufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	};
	VkBuffer stagingBuffer;
	VmaAllocation stagingAllocation;
	auto result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
					&stagingBuffer, &stagingAllocation, VK_NULL_HANDLE);
	vk::checkResult(result, "Failed to create staging buffer for image upload: {}");

	// Reset fences and command buffers.
	auto cmd = uploadCommandPools[taskScheduler.GetThreadNum()].buffer;
	vkResetCommandBuffer(cmd, 0);
	auto fence = fencePool.acquire();
	vkResetFences(device, 1, &fence->handle);

	const VkCommandBufferBeginInfo beginInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &beginInfo);

	commands(cmd, stagingBuffer, stagingAllocation);

	vkEndCommandBuffer(cmd);

	auto& queue = getNextTransferQueueHandle();
	{
		std::lock_guard lock(*queue.lock);

		const VkCommandBufferSubmitInfo cmdInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
			.commandBuffer = cmd,
		};
		const VkSubmitInfo2 submitInfo {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.commandBufferInfoCount = 1,
			.pCommandBufferInfos = &cmdInfo,
		};
		result = vkQueueSubmit2(queue.handle, 1, &submitInfo, fence->handle);
		vk::checkResult(result, "Failed to submit image upload: {}");
	}

	// We always wait for this operation to complete here, to free up the command buffer and fence for the next iteration.
	fencePool.wait_and_free(fence);

	// Destroy the staging buffer
	vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
}

void Viewer::uploadMeshlets(GlobalMeshData& data) {
	ZoneScoped;
	{
		// Create the meshlet description buffer
		uploadBufferToDevice(std::as_bytes(std::span(data.globalMeshlets.begin(), data.globalMeshlets.end())),
							 &globalMeshBuffers.descHandle, &globalMeshBuffers.descAllocation);
		vk::setDebugUtilsName(device, globalMeshBuffers.descHandle, "Meshlet descriptions");
	}
	{
		// Create the vertex index buffer
		uploadBufferToDevice(std::as_bytes(std::span(data.globalMeshletVertices.begin(), data.globalMeshletVertices.end())),
							 &globalMeshBuffers.vertexIndiciesHandle, &globalMeshBuffers.vertexIndiciesAllocation);
		vk::setDebugUtilsName(device, globalMeshBuffers.vertexIndiciesHandle, "Meshlet vertex indices");
	}
	{
		// Create the meshlet description buffer
		uploadBufferToDevice(std::as_bytes(std::span(data.globalMeshletTriangles.begin(), data.globalMeshletTriangles.end())),
							 &globalMeshBuffers.triangleIndicesHandle, &globalMeshBuffers.triangleIndicesAllocation);
		vk::setDebugUtilsName(device, globalMeshBuffers.triangleIndicesHandle, "Meshlet triangle indices");
	}
	{
		// Create the vertex buffer
		uploadBufferToDevice(std::as_bytes(std::span(data.globalVertices.begin(), data.globalVertices.end())),
							 &globalMeshBuffers.verticesHandle, &globalMeshBuffers.verticesAllocation);
		vk::setDebugUtilsName(device, globalMeshBuffers.verticesHandle, "Meshlet vertices");
	}

	deletionQueue.push([this]() {
		vmaDestroyBuffer(allocator, globalMeshBuffers.verticesHandle, globalMeshBuffers.verticesAllocation);
		vmaDestroyBuffer(allocator, globalMeshBuffers.triangleIndicesHandle, globalMeshBuffers.triangleIndicesAllocation);
		vmaDestroyBuffer(allocator, globalMeshBuffers.vertexIndiciesHandle, globalMeshBuffers.vertexIndiciesAllocation);
		vmaDestroyBuffer(allocator, globalMeshBuffers.descHandle, globalMeshBuffers.descAllocation);
	});

	// Allocate the primitive descriptor set
	std::array<VkDescriptorSetLayout, frameOverlap> setLayouts {};
	std::fill(setLayouts.begin(), setLayouts.end(), meshletSetLayout);
	const VkDescriptorSetAllocateInfo allocateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = static_cast<std::uint32_t>(setLayouts.size()),
		.pSetLayouts = setLayouts.data(),
	};

	// Allocate the sets. We copy each member of the sets vector in the loop below.
	globalMeshBuffers.descriptors.resize(allocateInfo.descriptorSetCount);
	auto result = vkAllocateDescriptorSets(device, &allocateInfo, globalMeshBuffers.descriptors.data());
	vk::checkResult(result, "Failed to allocate mesh buffers descriptor set: {}");

	for (auto& descriptor : globalMeshBuffers.descriptors) {
		// Update the descriptors with the buffer handles
		std::array<VkDescriptorBufferInfo, 4> descriptorBufferInfos{{
			{
				.buffer = globalMeshBuffers.descHandle,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			},
			{
				.buffer = globalMeshBuffers.vertexIndiciesHandle,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			},
			{
				.buffer = globalMeshBuffers.triangleIndicesHandle,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			},
			{
				.buffer = globalMeshBuffers.verticesHandle,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			},
		}};
		std::array<VkWriteDescriptorSet, 4> descriptorWrites{{
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &descriptorBufferInfos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &descriptorBufferInfos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &descriptorBufferInfos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &descriptorBufferInfos[3],
			}
		}};
		vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0,
							   nullptr);
	}
}

#include <stb_image.h>

#define DDS_USE_STD_FILESYSTEM 1
#include <dds.hpp>

#include <ktx.h>

template <>
struct fmt::formatter<dds::ReadResult> : formatter<std::string_view> {
    template <typename FormatContext>
    inline auto format(dds::ReadResult const& result, FormatContext& ctx) const {
		std::string_view stringified;
		switch (result) {
			case dds::ReadResult::Success: {
				stringified = "Success";
				break;
			}
			case dds::ReadResult::Failure: {
				stringified = "Failure";
				break;
			}
			case dds::ReadResult::UnsupportedFormat: {
				stringified = "UnsupportedFormat";
				break;
			}
			case dds::ReadResult::NoDx10Header: {
				stringified = "NoDx10Header";
				break;
			}
			case dds::ReadResult::InvalidSize: {
				stringified = "InvalidSize";
				break;
			}
		}
		return formatter<string_view>::format(stringified, ctx);
    }
};

struct ImageLoadTask : public ExceptionTaskSet {
	Viewer* viewer;

	explicit ImageLoadTask(Viewer* viewer) noexcept : viewer(viewer) {
		// We load the default images elsewhere.
		m_SetSize = viewer->images.size() - Viewer::numDefaultTextures;
	}

	void loadWithStb(std::uint32_t imageIdx, std::span<std::uint8_t> encodedImageData) const {
		ZoneScoped;
		static constexpr VkFormat imageFormat = VK_FORMAT_R8G8B8A8_SRGB;
		static constexpr auto channels = 4;

		// Load and decode the image data using stbi
		int width = 0, height = 0, nrChannels = 0;
		auto* ptr = stbi_load_from_memory(encodedImageData.data(), static_cast<int>(encodedImageData.size()), &width, &height, &nrChannels, channels);
		if (ptr == nullptr) {
			if (auto* reason = stbi_failure_reason(); reason != nullptr)
				throw std::runtime_error(fmt::format("Failed to load image using stbi from memory: {}", reason));
			throw std::runtime_error("Failed to load image using stbi from memory");
		}

		auto& sampledImage = viewer->images[imageIdx];

		auto mipLevels = static_cast<std::uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
		const VkImageCreateInfo imageInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = imageFormat,
			.extent = {
				.width = static_cast<std::uint32_t>(width),
				.height = static_cast<std::uint32_t>(height),
				.depth = 1,
			},
			.mipLevels = mipLevels,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		const VmaAllocationCreateInfo allocationInfo {
			.usage = VMA_MEMORY_USAGE_AUTO,
		};
		auto result = vmaCreateImage(viewer->allocator, &imageInfo, &allocationInfo,
					   &sampledImage.image, &sampledImage.allocation, nullptr);
		vk::checkResult(result, "Failed to create Vulkan image for stbi image: {}");

		const VkImageViewCreateInfo imageViewInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = sampledImage.image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = imageInfo.format,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = mipLevels,
				.layerCount = 1,
			},
		};
		result = vkCreateImageView(viewer->device, &imageViewInfo, VK_NULL_HANDLE, &sampledImage.imageView);
		vk::checkResult(result, "Failed to create Vulkan image view for stbi image: {}");

		auto imageData = std::span(ptr, width * height * sizeof(std::uint8_t) * channels);
		viewer->uploadImageToDevice(imageData.size_bytes(), [&](VkCommandBuffer cmd, VkBuffer stagingBuffer, VmaAllocation stagingAllocation) {
			{
				vk::ScopedMap map(viewer->allocator, stagingAllocation);
				std::memcpy(map.get(), imageData.data(), imageData.size_bytes());
			}

			// Transition the entire image (with all mip levels) to TRANSFER_DST_OPTIMAL
			VkImageMemoryBarrier2 imageBarrier {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
				.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = sampledImage.image,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = mipLevels,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			const VkDependencyInfo dependencyInfo {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &imageBarrier,
			};
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);

			// Copy the image
			const VkBufferImageCopy copy {
				.bufferOffset = 0,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.layerCount = 1,
				},
				.imageOffset = {
					.x = 0,
					.y = 0,
					.z = 0,
				},
				.imageExtent = imageInfo.extent,
			};
			vkCmdCopyBufferToImage(cmd, stagingBuffer, sampledImage.image,
								   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
		});

		stbi_image_free(imageData.data());

		// Generate mipmaps
		auto cmd = viewer->graphicsCommandPools[taskScheduler.GetThreadNum()].buffer;
		vkResetCommandBuffer(cmd, 0);
		auto fence = viewer->fencePool.acquire();
		vkResetFences(viewer->device, 1, &fence->handle);

		const VkCommandBufferBeginInfo beginInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		vkBeginCommandBuffer(cmd, &beginInfo);

		// Reused image barrier for transitioning each mip layer
		VkImageMemoryBarrier2 barrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = sampledImage.image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		const VkDependencyInfo dependencyInfo {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier,
		};

		std::int32_t mipWidth = width;
		std::int32_t mipHeight = height;
		for (std::uint32_t i = 1; i < mipLevels; ++i) {
			// Transition mip i - 1 to TRANSFER_SRC_OPTIMAL
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
			vkCmdPipelineBarrier2(cmd,&dependencyInfo);

			const VkImageBlit blit {
				.srcSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = i - 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.srcOffsets = {
					0, 0, 0,
					mipWidth, mipHeight, 1,
				},
				.dstSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = i,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.dstOffsets = {
					0, 0, 0,
					mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1,
				},
			};
			vkCmdBlitImage(cmd,
						   sampledImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						   sampledImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
						   1, &blit, VK_FILTER_LINEAR);

			// Transition mip i to TRANSFER_SRC_OPTIMAL
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
			vkCmdPipelineBarrier2(cmd,&dependencyInfo);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		// Transition the last mip level to SHARED_READ_ONLY_OPTIMAL
		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
		vkCmdPipelineBarrier2(cmd,&dependencyInfo);

		vkEndCommandBuffer(cmd);

		// TODO: Use compute queues instead of the main queue?
		auto& queue = viewer->graphicsQueue;
		{
			std::lock_guard lock(*queue.lock);

			const VkCommandBufferSubmitInfo cmdInfo {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
				.commandBuffer = cmd,
			};
			const VkSubmitInfo2 submitInfo {
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
				.commandBufferInfoCount = 1,
				.pCommandBufferInfos = &cmdInfo,
			};
			result = vkQueueSubmit2(queue.handle, 1, &submitInfo, fence->handle);
			vk::checkResult(result, "Failed to submit mipmap generation: {}");
		}

		viewer->fencePool.wait_and_free(fence);
	}

	void loadDds(std::uint32_t imageIdx, std::span<std::uint8_t> encodedImageData) const {
		ZoneScoped;
		dds::Image image;
		auto ddsResult = dds::readImage(encodedImageData.data(), encodedImageData.size_bytes(), &image);
		if (ddsResult != dds::ReadResult::Success) {
			throw std::runtime_error(fmt::format("Failed to read DDS image: {}", ddsResult));
		}

		auto& sampledImage = viewer->images[imageIdx];

		// Create the Vulkan image
		auto vkFormat = dds::getVulkanFormat(image.format, image.supportsAlpha);
		auto imageInfo = dds::getVulkanImageCreateInfo(&image, vkFormat);
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		const VmaAllocationCreateInfo allocationInfo {
			.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		};
		auto result = vmaCreateImage(viewer->allocator, &imageInfo, &allocationInfo,
					   &sampledImage.image, &sampledImage.allocation, nullptr);
		vk::checkResult(result, "Failed to create Vulkan image for DDS texture: {}");

		auto imageViewInfo = dds::getVulkanImageViewCreateInfo(&image, vkFormat);
		imageViewInfo.image = sampledImage.image;
		result = vkCreateImageView(viewer->device, &imageViewInfo, nullptr, &sampledImage.imageView);
		vk::checkResult(result, "Failed to create Vulkan image view for DDS texture: {}");

		// Compute the aligned offsets for every mip level
		std::size_t totalSize = 0;
		fastgltf::StaticVector<VkDeviceSize> offsets(image.mipmaps.size());
		for (std::size_t i = 0; auto& mip : image.mipmaps) {
			offsets[i++] = fastgltf::alignUp(totalSize, static_cast<std::int32_t>(dds::getBlockSize(image.format)));
			totalSize += mip.size_bytes();
		}

		viewer->uploadImageToDevice(totalSize, [&](VkCommandBuffer cmd, VkBuffer stagingBuffer, VmaAllocation stagingAllocation) {
			// Copy the image mips with correct offsets, aligned to the format's texel size
			{
				vk::ScopedMap<std::byte> map(viewer->allocator, stagingAllocation);
				for (std::size_t i = 0; auto& mip : image.mipmaps) {
					std::memcpy(map.get() + offsets[i++], mip.data(), mip.size_bytes());
				}
			}

			// Transition the image to TRANSFER_DST_OPTIMAL
			VkImageMemoryBarrier2 imageBarrier {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_NONE,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
				.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = sampledImage.image,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = image.numMips,
					.layerCount = 1,
				},
			};
			const VkDependencyInfo dependencyInfo {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &imageBarrier,
			};
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);

			// For each mip, submit a command buffer copying the staging buffer to the device image
			glm::u32vec2 extent(image.width, image.height);
			for (std::uint32_t i = 0; i < image.numMips; ++i) {
				const VkBufferImageCopy copy {
					.bufferOffset = offsets[i],
					.bufferRowLength = 0,
					.bufferImageHeight = 0,
					.imageSubresource = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.mipLevel = i,
						.layerCount = 1,
					},
					.imageOffset = {
						.x = 0,
						.y = 0,
						.z = 0,
					},
					.imageExtent = {
						.width = extent.x,
						.height = extent.y,
						.depth = 1,
					},
				};
				vkCmdCopyBufferToImage(cmd, stagingBuffer, sampledImage.image,
									   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

				extent /= 2;
			}

			// Transition the image into SHADER_READ_ONLY_OPTIMAL
			imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
			imageBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			imageBarrier.dstAccessMask = VK_ACCESS_2_NONE;
			imageBarrier.oldLayout = imageBarrier.newLayout;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);
		});
	}

	void loadKtx(std::uint32_t imageIdx, std::span<std::uint8_t> data) const {
		ZoneScoped;
		// The KTX library also has no idea what modern programming looks like apparently, and has an absolute abomination of an API.
		// Well, we have to live with it as I'm not making my own transcoder for unsupported compressed formats.
		ktxTexture2* texture = nullptr;
		auto ktxError = ktxTexture2_CreateFromMemory(
			data.data(),
			data.size(),
			KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
			&texture);
		if (ktxError != KTX_SUCCESS) {
			throw std::runtime_error(
				fmt::format("Failed to create KTX2 texture: {}", fastgltf::to_underlying(ktxError)));
		}

		// Transcode the format if necessary. This will change the value of vkFormat.
		// TODO: We always transcode to BC7_SRGB. Try and detect other possible formats?
		auto imageFormat = static_cast<VkFormat>(texture->vkFormat);
		if (ktxTexture2_NeedsTranscoding(texture)) {
			ktxError = ktxTexture2_TranscodeBasis(texture, KTX_TTF_BC7_RGBA, KTX_TF_HIGH_QUALITY);
			if (ktxError != KTX_SUCCESS) {
				throw std::runtime_error(fmt::format("Failed to transcoe basisu from KTX2 texture: {}", fastgltf::to_underlying(ktxError)));
			}

			imageFormat = VK_FORMAT_BC7_SRGB_BLOCK;
		}

		auto& sampledImage = viewer->images[imageIdx];

		const VkImageCreateInfo imageInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = imageFormat,
			.extent = {
				.width = texture->baseWidth,
				.height = texture->baseHeight,
				.depth = texture->baseDepth,
			},
			.mipLevels = texture->numLevels,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		const VmaAllocationCreateInfo allocationInfo {
			.usage = VMA_MEMORY_USAGE_AUTO,
		};
		auto result = vmaCreateImage(viewer->allocator, &imageInfo, &allocationInfo,
									 &sampledImage.image, &sampledImage.allocation, nullptr);
		vk::checkResult(result, "Failed to create Vulkan image for KTX2 texture: {}");

		const VkImageViewCreateInfo imageViewInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = sampledImage.image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = imageFormat,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = texture->numLevels,
				.layerCount = 1,
			}
		};
		result = vkCreateImageView(viewer->device, &imageViewInfo, nullptr, &sampledImage.imageView);
		vk::checkResult(result, "Failed to create Vulkan image view for KTX2 texture: {}");

		viewer->uploadImageToDevice(texture->dataSize, [&](VkCommandBuffer cmd, VkBuffer stagingBuffer, VmaAllocation stagingAllocation) {
			{
				vk::ScopedMap map(viewer->allocator, stagingAllocation);
				std::memcpy(map.get(), texture->pData, texture->dataSize);
			}

			// Transition the image to TRANSFER_DST_OPTIMAL
			VkImageMemoryBarrier2 imageBarrier {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_NONE,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
				.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = sampledImage.image,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.levelCount = texture->numLevels,
					.layerCount = 1,
				},
			};
			const VkDependencyInfo dependencyInfo {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &imageBarrier,
			};
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);

			glm::u32vec2 extent(texture->baseWidth, texture->baseHeight);
			for (std::uint32_t i = 0; i < texture->numLevels; ++i) {
				std::uint64_t offset = 0;
				ktxTexture_GetImageOffset(ktxTexture(texture), i, 0, 0, &offset);
				const VkBufferImageCopy copy {
					.bufferOffset = offset,
					.imageSubresource = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.mipLevel = i,
						.layerCount = 1,
					},
					.imageExtent = {
						.width = extent.x,
						.height = extent.y,
						.depth = 1,
					}
				};
				vkCmdCopyBufferToImage(cmd, stagingBuffer, sampledImage.image,
									   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

				extent /= 2;
			}

			// Transition the image into SHADER_READ_ONLY_OPTIMAL
			imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
			imageBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			imageBarrier.dstAccessMask = VK_ACCESS_2_NONE;
			imageBarrier.oldLayout = imageBarrier.newLayout;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			vkCmdPipelineBarrier2(cmd, &dependencyInfo);
		});

		ktxTexture_Destroy(ktxTexture(texture));
	}

	/** Sometimes, the mimeType field is unspecified. Here, we try and detect the mimeType if it's initially None */
	fastgltf::MimeType detectMimeType(std::span<std::uint8_t> data, fastgltf::MimeType mimeType) {
		if (mimeType != fastgltf::MimeType::None) {
			return mimeType;
		}

		// The glTF did not provide a mime type. Try to detect the image format using the header magic.
		// See https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#images for the table of patterns
		if (data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
			return fastgltf::MimeType::JPEG;
		} else if (data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 && data[4] == 0x0D && data[5] == 0x0A && data[6] == 0x1A && data[7] == 0x0A) {
			return fastgltf::MimeType::PNG;
		} else if (*reinterpret_cast<std::uint32_t*>(data.data()) == dds::DdsMagicNumber::DDS) {
			return fastgltf::MimeType::DDS;
		} else if (data[0] == 0xAB && data[1] == 'K' && data[2] == 'T' && data[3] == 'X') {
			return fastgltf::MimeType::KTX2;
		} else {
			throw std::runtime_error(
				fmt::format("Failed to detect image mime type while loading. Header magic: {0:x}", *reinterpret_cast<std::uint32_t*>(data.data())));
		}
	}

	void load(std::uint32_t imageIdx, std::span<std::uint8_t> data, fastgltf::MimeType mimeType) {
		auto actualMime = detectMimeType(data, mimeType);
		switch (actualMime) {
			case fastgltf::MimeType::PNG:
			case fastgltf::MimeType::JPEG: {
				loadWithStb(imageIdx, data);
				return;
			}
			case fastgltf::MimeType::DDS: {
				loadDds(imageIdx, data);
				return;
			}
			case fastgltf::MimeType::KTX2: {
				loadKtx(imageIdx, data);
				break;
			}
			default: {
				throw std::runtime_error(fmt::format("Unsupported mime type for loading images: {}", fastgltf::to_underlying(actualMime)));
			}
		}
	}

	// We'll use the range to operate over multiple images
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override {
		ZoneScoped;
		for (auto i = range.start; i < range.end; ++i) {
			auto& image = viewer->asset.images[i];

			auto imageIdx = i + Viewer::numDefaultTextures;
			std::visit(fastgltf::visitor{
				[](auto& arg) {
					throw std::runtime_error("Got an unexpected image data source. Can't load image.");
					return;
				},
				[&](fastgltf::sources::Array& array) {
					load(imageIdx, std::span(array.bytes.data(), array.bytes.size()), array.mimeType);
				},
				[&](fastgltf::sources::BufferView& bufferView) {
					auto& view = viewer->asset.bufferViews[bufferView.bufferViewIndex];
					auto& buffer = viewer->asset.buffers[view.bufferIndex];
					std::visit(fastgltf::visitor{
						[](auto& arg) {
							throw std::runtime_error("Got an unexpected image data source. Can't load image.");
							return;
						},
						[&](fastgltf::sources::Array& array) {
							load(imageIdx, std::span(array.bytes.data() + view.byteOffset, array.bytes.size()), bufferView.mimeType);
						}
					}, buffer.data);
				}
			}, image.data);

			if (image.name.empty()) {
				image.name = fmt::format("Image {}", i);
			}

			auto& sampledImage = viewer->images[imageIdx];
			vk::setDebugUtilsName(viewer->device, sampledImage.image, image.name.c_str());
			vk::setDebugUtilsName(viewer->device, sampledImage.imageView, fmt::format("View of {}", image.name));
			vk::setAllocationName(viewer->allocator, sampledImage.allocation,
								  fmt::format("Allocation of {}", image.name));
		}
	}
};

void Viewer::createDefaultImages() {
	ZoneScoped;
	// Create a default 1x1 white image used as a fallback
	const VkImageCreateInfo imageInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_UNORM,
		.extent = {
			.width = 1,
			.height = 1,
			.depth = 1,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo allocationInfo {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	auto& defaultTexture = images[0];
	auto result = vmaCreateImage(allocator, &imageInfo, &allocationInfo,
				   &defaultTexture.image, &defaultTexture.allocation, nullptr);
	vk::checkResult(result, "Failed to create default image: {}");
	vk::setDebugUtilsName(device, defaultTexture.image, "Default image");
	vk::setAllocationName(allocator, defaultTexture.allocation, "Default image allocation");

	// We use R8G8B8A8_UNORM, so we need to use 8-bit integers for the colors here.
	std::array<std::uint8_t, 4> white {{ 255, 255, 255, 255 }};
	uploadImageToDevice(sizeof white, [&](VkCommandBuffer cmd, VkBuffer stagingBuffer, VmaAllocation stagingAllocation) {
		{
			vk::ScopedMap map(allocator, stagingAllocation);
			std::memcpy(map.get(), white.data(), sizeof white);
		}

		// Transition the image to TRANSFER_DST_OPTIMAL
		VkImageMemoryBarrier2 imageBarrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
			.srcAccessMask = VK_ACCESS_2_NONE,
			.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = defaultTexture.image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.layerCount = 1,
			},
		};
		const VkDependencyInfo dependencyInfo {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &imageBarrier,
		};
		vkCmdPipelineBarrier2(cmd, &dependencyInfo);

		// Copy the image
		const VkBufferImageCopy copy {
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.layerCount = 1,
			},
			.imageOffset = {
				.x = 0,
				.y = 0,
				.z = 0,
			},
			.imageExtent = imageInfo.extent,
		};
		vkCmdCopyBufferToImage(cmd, stagingBuffer, defaultTexture.image,
							   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

		// Transition the image into the destinationLayout
		imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
		imageBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
		imageBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		imageBarrier.oldLayout = imageBarrier.newLayout;
		imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		vkCmdPipelineBarrier2(cmd, &dependencyInfo);
	});

	const VkImageViewCreateInfo imageViewInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = defaultTexture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = imageInfo.format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1,
		},
	};
	result = vkCreateImageView(device, &imageViewInfo, VK_NULL_HANDLE, &defaultTexture.imageView);
	vk::checkResult(result, "Failed to create default image view: {}");
	vk::setDebugUtilsName(device, defaultTexture.imageView, "Default image view");
}

VkFilter getVulkanFilter(fastgltf::Filter filter) {
	switch (filter) {
		case fastgltf::Filter::Nearest:
		case fastgltf::Filter::NearestMipMapNearest:
		case fastgltf::Filter::NearestMipMapLinear:
		default:
			return VK_FILTER_NEAREST;
		case fastgltf::Filter::Linear:
		case fastgltf::Filter::LinearMipMapNearest:
		case fastgltf::Filter::LinearMipMapLinear:
			return VK_FILTER_LINEAR;
	}
}

VkSamplerMipmapMode getVulkanMipmapMode(fastgltf::Filter filter) {
	switch (filter) {
		case fastgltf::Filter::NearestMipMapNearest:
		case fastgltf::Filter::LinearMipMapNearest:
			return VK_SAMPLER_MIPMAP_MODE_NEAREST;

		case fastgltf::Filter::NearestMipMapLinear:
		case fastgltf::Filter::LinearMipMapLinear:
		default:
			return VK_SAMPLER_MIPMAP_MODE_LINEAR;
	}
}

VkSamplerAddressMode getVulkanAddressMode(fastgltf::Wrap wrap) {
	switch (wrap) {
		case fastgltf::Wrap::ClampToEdge:
			return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		case fastgltf::Wrap::MirroredRepeat:
			return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
		case fastgltf::Wrap::Repeat:
		default:
			return VK_SAMPLER_ADDRESS_MODE_REPEAT;
	}
}

void Viewer::loadGltfImages() {
	ZoneScoped;
	// Schedule image loading first
	// TODO: When anything throws while these tasks are running, the task will be destroyed and the ~enki::ICompletable
	//       destructor will assert, as the task has not completed yet.
	images.resize(numDefaultTextures + asset.images.size());
	ImageLoadTask loadTask(this);
	taskScheduler.AddTaskSetToPipe(&loadTask);

	createDefaultImages();

	// Create the material descriptor layout
	// TODO: We currently use a fixed size for the descriptorCount of the image samplers.
	//       Using VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT we could change the descriptor size.
	// TODO: We currently dont use UPDATE_AFTER_BIND, making us use either frameOverlap count of sets, or restricting
	//       us to a fixed set of textures for rendering.
	std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings = {{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = static_cast<std::uint32_t>(asset.textures.size() + numDefaultTextures),
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		}
	}};
	std::array<VkDescriptorBindingFlags, layoutBindings.size()> layoutBindingFlags = {{
		0,
		0, // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
		0,
	}};
	const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		.bindingCount = static_cast<std::uint32_t>(layoutBindingFlags.size()),
		.pBindingFlags = layoutBindingFlags.data(),
	};
	const VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = &bindingFlagsInfo,
		.bindingCount = static_cast<std::uint32_t>(layoutBindings.size()),
		.pBindings = layoutBindings.data(),
	};
	auto result = vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfo,
											  VK_NULL_HANDLE, &materialSetLayout);
	vk::checkResult(result, "Failed to create material descriptor set layout: {}");
	vk::setDebugUtilsName(device, materialSetLayout, "Material descriptor layout");
	deletionQueue.push([this]() {
		vkDestroyDescriptorSetLayout(device, materialSetLayout, nullptr);
	});

	// Allocate the material descriptor
	const VkDescriptorSetAllocateInfo allocateInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &materialSetLayout,
	};
	result = vkAllocateDescriptorSets(device, &allocateInfo, &materialSet);
	vk::checkResult(result, "Failed to allocate material descriptor set: {}");
	vk::setDebugUtilsName(device, materialSet, "Material descriptor set");

	// While we're here, also load the materials
	loadGltfMaterials();

	samplers.resize(asset.samplers.size() + numDefaultSamplers);
	// Create the default sampler
	VkSamplerCreateInfo samplerInfo {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.mipLodBias = -1.0f,
		.anisotropyEnable = VK_TRUE,
		.maxAnisotropy = 16.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
	};
	result = vkCreateSampler(device, &samplerInfo, nullptr, &samplers[0]);
	vk::checkResult(result, "Failed to create default sampler: {}");

	// Create the glTF samplers
	for (auto i = 0; i < asset.samplers.size(); ++i) {
		auto& sampler = asset.samplers[i];
		samplerInfo.magFilter = getVulkanFilter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
		samplerInfo.minFilter = getVulkanFilter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
		samplerInfo.mipmapMode = getVulkanMipmapMode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
		samplerInfo.addressModeU = getVulkanAddressMode(sampler.wrapS);
		samplerInfo.addressModeV = getVulkanAddressMode(sampler.wrapT);
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
		result = vkCreateSampler(device, &samplerInfo, nullptr, &samplers[numDefaultSamplers + i]);
		vk::checkResult(result, "Failed to create sampler: {}");
	}

	// Finish all texture decode and upload tasks
	taskScheduler.WaitforTask(&loadTask);
	if (loadTask.exception)
		std::rethrow_exception(loadTask.exception);

	// Update the texture descriptor
	std::vector<VkWriteDescriptorSet> writes; writes.reserve(asset.textures.size() + numDefaultTextures);
	std::vector<VkDescriptorImageInfo> infos; infos.reserve(writes.capacity());

	// Write the default texture
	infos.emplace_back(VkDescriptorImageInfo {
		.sampler = samplers[0],
		.imageView = images[0].imageView,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	});
	writes.emplace_back(VkWriteDescriptorSet {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = materialSet,
		.dstBinding = 2,
		.dstArrayElement = 0U,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &infos.back(),
	});

	// Write the glTF textures
	for (std::size_t i = 0; i < asset.textures.size(); ++i) {
		auto& texture = asset.textures[i];

		// Get the image index for the "best" image. If we have compressed images available, we prefer those.
		std::size_t imageViewIndex = 0;
		if (texture.basisuImageIndex.has_value()) {
			imageViewIndex = texture.basisuImageIndex.value() + numDefaultTextures;
		} else if (texture.ddsImageIndex.has_value()) {
			imageViewIndex = texture.ddsImageIndex.value() + numDefaultTextures;
		} else if (texture.imageIndex.has_value()) {
			imageViewIndex = texture.imageIndex.value() + numDefaultTextures;
		}

		// Well map a glTF texture to a single combined image sampler
		infos.emplace_back(VkDescriptorImageInfo {
			.sampler = samplers[texture.samplerIndex.has_value() ? *texture.samplerIndex + numDefaultSamplers : 0],
			.imageView = images[imageViewIndex].imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		});

		writes.emplace_back(VkWriteDescriptorSet {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = materialSet,
			.dstBinding = 2,
			.dstArrayElement = static_cast<std::uint32_t>(i),
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &infos.back(),
		});
	}
	vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
}

void Viewer::createShadowMapAndPipeline() {
	ZoneScoped;
	// We don't check if D32_SFLOAT supports SAMPLED_BIT as it's supported on effectively 100% of devices.
	const VkImageCreateInfo shadowMapInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_D32_SFLOAT,
		.extent = {
			.width = shadowResolution.x,
			.height = shadowResolution.y,
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
	const VmaAllocationCreateInfo allocationInfo {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	auto result = vmaCreateImage(allocator, &shadowMapInfo, &allocationInfo, &shadowMapImage, &shadowMapAllocation, nullptr);
	vk::checkResult(result, "Failed to create shadow map image: {}");
	vk::setDebugUtilsName(device, shadowMapImage, "Shadow map image");
	vk::setAllocationName(allocator, shadowMapAllocation, "Shadow map image allocation");

	const VkImageViewCreateInfo imageViewInfo {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = shadowMapImage,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = shadowMapInfo.format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		}
	};
	result = vkCreateImageView(device, &imageViewInfo, VK_NULL_HANDLE, &shadowMapImageView);
	vk::checkResult(result, "Failed to create shadow map image view: {}");
	vk::setDebugUtilsName(device, depthImageView, "Shadow map image view");

	// Create the sampler for the shadow map
	const VkSamplerCreateInfo samplerInfo {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.maxAnisotropy = 1.0f,
		.maxLod = 1.0f,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE, // I think this should be 1.0f in all components?
	};
	result = vkCreateSampler(device, &samplerInfo, nullptr, &shadowMapSampler);
	vk::checkResult(result, "Failed to create shadow map sampler: {}");

	deletionQueue.push([this]() {
		vkDestroySampler(device, shadowMapSampler, nullptr);
		vkDestroyImageView(device, shadowMapImageView, nullptr);
		vmaDestroyImage(allocator, shadowMapImage, shadowMapAllocation);
	});

	// Update the material descriptor with the shadowMap
	const VkDescriptorImageInfo imageInfo {
		.sampler = shadowMapSampler,
		.imageView = shadowMapImageView,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	const VkWriteDescriptorSet write {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = materialSet,
		.dstBinding = 1,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &imageInfo,
	};
	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

	// Build the shadow map pipeline layout
	std::array<VkDescriptorSetLayout, 2> layouts {{ cameraSetLayout, meshletSetLayout }};
	const VkPipelineLayoutCreateInfo layoutCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 0,
	};
	result = vkCreatePipelineLayout(device, &layoutCreateInfo, VK_NULL_HANDLE, &shadowMapPipelineLayout);
	vk::checkResult(result, "Failed to create shadow map pipeline layout: {}");
	vk::setDebugUtilsName(device, meshPipelineLayout, "Shadow map pipeline layout");

	// Load shaders
	VkShaderModule meshModule, fragModule;
	vk::loadShaderModule("shadow_map.mesh.glsl.spv", device, &meshModule);
	vk::loadShaderModule("shadow_map.frag.glsl.spv", device, &fragModule);

	const auto depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
	const VkPipelineRenderingCreateInfo renderingCreateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 0,
		.depthAttachmentFormat = depthAttachmentFormat,
	};

	auto builder = vk::GraphicsPipelineBuilder(device, 1)
		.setPipelineLayout(0, shadowMapPipelineLayout)
		.pushPNext(0, &renderingCreateInfo)
		.addDynamicState(0, VK_DYNAMIC_STATE_SCISSOR)
		.addDynamicState(0, VK_DYNAMIC_STATE_VIEWPORT)
		.setTopology(0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		.setDepthState(0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL)
		// We're using CULL_MODE_FRONT to avoid peter panning
		.setRasterState(0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
		.setMultisampleCount(0, VK_SAMPLE_COUNT_1_BIT)
		.setScissorCount(0, 1U)
		.setViewportCount(0, 1U)
		.addShaderStage(0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule)
		.addShaderStage(0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule);

	result = builder.build(&shadowMapPipeline);
	vk::checkResult(result, "Failed to create shadow map pipeline: {}");
	vk::setDebugUtilsName(device, shadowMapPipeline, "Shadow map pipeline");

	vkDestroyShaderModule(device, fragModule, nullptr);
	vkDestroyShaderModule(device, meshModule, nullptr);

	deletionQueue.push([this]() {
		vkDestroyPipeline(device, shadowMapPipeline, nullptr);
		vkDestroyPipelineLayout(device, shadowMapPipelineLayout, nullptr);
	});
}

void Viewer::loadGltfMaterials() {
	ZoneScoped;
	// Create the material buffer data
	std::vector<glsl::Material> materials; materials.reserve(asset.materials.size());

	// Add the default material
	materials.emplace_back(glsl::Material {
		.albedoFactor = glm::vec4(1.0f),
		.albedoIndex = 0,
		.alphaCutoff = 0.5f,
	});

	for (auto& gltfMaterial : asset.materials) {
		auto& mat = materials.emplace_back();
		mat.albedoFactor = glm::make_vec4(gltfMaterial.pbrData.baseColorFactor.data());
		mat.uvOffset = glm::vec2(0);
		mat.uvScale = glm::vec2(1.0f);
		mat.uvRotation = 0.0f;

		if (gltfMaterial.pbrData.baseColorTexture.has_value()) {
			auto& albedoTex = gltfMaterial.pbrData.baseColorTexture.value();
			mat.albedoIndex = albedoTex.textureIndex;
			if (albedoTex.transform) {
				mat.uvOffset = glm::make_vec2(albedoTex.transform->uvOffset.data());
				mat.uvScale = glm::make_vec2(albedoTex.transform->uvScale.data());
				mat.uvRotation = albedoTex.transform->rotation;
			}
		} else {
			mat.albedoIndex = 0;
		}
		mat.alphaCutoff = gltfMaterial.alphaCutoff;
	}

	// Create the material buffer
	const VmaAllocationCreateInfo allocationCreateInfo {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	const VkBufferCreateInfo bufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = materials.size() * sizeof(decltype(materials)::value_type),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	auto result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
								  &materialBuffer, &materialAllocation, VK_NULL_HANDLE);
	vk::checkResult(result, "Failed to allocate material buffer: {}");
	vk::setDebugUtilsName(device, materialBuffer, "Material buffer");

	deletionQueue.push([this]() {
		vmaDestroyBuffer(allocator, materialBuffer, materialAllocation);
	});

	// Copy the material data to the buffer
	{
		vk::ScopedMap<glsl::Material> map(allocator, materialAllocation);
		std::memcpy(map.get(), materials.data(), bufferCreateInfo.size);
	}

	// Update the material descriptor
	const VkDescriptorBufferInfo bufferInfo {
		.buffer = materialBuffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};
	const VkWriteDescriptorSet write {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = materialSet,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &bufferInfo,
	};
	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

glm::mat4 getTransformMatrix(const fastgltf::Node& node, glm::mat4x4& base) {
	/** Both a matrix and TRS values are not allowed
	 * to exist at the same time according to the spec */
	if (const auto* pMatrix = std::get_if<fastgltf::Node::TransformMatrix>(&node.transform)) {
		return base * glm::mat4x4(glm::make_mat4x4(pMatrix->data()));
	}

	if (const auto* pTransform = std::get_if<fastgltf::TRS>(&node.transform)) {
		return base
			   * glm::translate(glm::mat4(1.0f), glm::make_vec3(pTransform->translation.data()))
			   * glm::toMat4(glm::quat::wxyz(pTransform->rotation[3], pTransform->rotation[0], pTransform->rotation[1], pTransform->rotation[2]))
			   * glm::scale(glm::mat4(1.0f), glm::make_vec3(pTransform->scale.data()));
	}

	return base;
}

void Viewer::drawNode(std::vector<glsl::PrimitiveDraw>& cmd, std::vector<VkDrawIndirectCommand>& aabbCmd, std::size_t nodeIndex, glm::mat4 matrix) {
	assert(asset.nodes.size() > nodeIndex);
	ZoneScoped;

	auto& node = asset.nodes[nodeIndex];
	matrix = getTransformMatrix(node, matrix);

	if (node.meshIndex.has_value()) {
		drawMesh(cmd, aabbCmd, node.meshIndex.value(), matrix);
	}

	for (auto& child : node.children) {
		drawNode(cmd, aabbCmd, child, matrix);
	}
}

void Viewer::drawMesh(std::vector<glsl::PrimitiveDraw>& cmd, std::vector<VkDrawIndirectCommand>& aabbCmd, std::size_t meshIndex, glm::mat4 matrix) {
	assert(meshes.size() > meshIndex);
	ZoneScoped;

	auto& mesh = meshes[meshIndex];

	for (std::size_t i = 0; auto& primitive : mesh.primitives) {
		auto& draw = cmd.emplace_back();

		// Get the appropriate material variant, if any.
		std::uint32_t materialIndex;
		auto& mappings = asset.meshes[meshIndex].primitives[i++].mappings;
		if (!mappings.empty() && mappings[materialVariant].has_value()) {
			materialIndex = mappings[materialVariant].value() + numDefaultMaterials; // Adjust for default material
		} else {
			materialIndex = primitive.materialIndex;
		}

		// Dispatch so many groups that we only have to use up to maxMeshlets 16-bit indices in the shared payload.
		const VkDrawMeshTasksIndirectCommandEXT indirectCommand {
			.groupCountX = static_cast<std::uint32_t>((primitive.meshlet_count + glsl::maxMeshlets - 1) / glsl::maxMeshlets),
			.groupCountY = 1,
			.groupCountZ = 1,
		};
		draw.command = indirectCommand;
		draw.modelMatrix = matrix;
		draw.descOffset = primitive.descOffset;
		draw.vertexIndicesOffset = primitive.vertexIndicesOffset;
		draw.triangleIndicesOffset = primitive.triangleIndicesOffset;
		draw.verticesOffset = primitive.verticesOffset;
		draw.meshletCount = static_cast<std::uint32_t>(primitive.meshlet_count);
		draw.materialIndex = materialIndex;

		// Create the AABB draw command
		auto& aabb = aabbCmd.emplace_back();
		aabb.vertexCount = 12 * 2; // 12 edges with each 2 vertices
		aabb.instanceCount = draw.meshletCount;
		aabb.firstVertex = 0;
		aabb.firstInstance = 0;
	}
}

void Viewer::updateDrawBuffer(std::size_t currentFrame) {
	ZoneScoped;
	assert(drawBuffers.size() > currentFrame);

	auto& currentDrawBuffer = drawBuffers[currentFrame];

	std::vector<glsl::PrimitiveDraw> draws;
	std::vector<VkDrawIndirectCommand> aabbDraws;

	if (asset.scenes.empty() || sceneIndex >= asset.scenes.size())
		return;

	auto& scene = asset.scenes[sceneIndex];
	for (auto& nodeIdx : scene.nodeIndices) {
		drawNode(draws, aabbDraws, nodeIdx, glm::mat4(1.0f));
	}

	// TODO: This limits our primitive count to 4.2 billion. Can we set this limit somewhere else,
	//		 or could we dispatch multiple indirect draws to remove the uint32_t limit?
	currentDrawBuffer.drawCount = static_cast<std::uint32_t>(draws.size());

	auto byteSize = currentDrawBuffer.drawCount * sizeof(decltype(draws)::value_type);
	if (currentDrawBuffer.primitiveDrawBufferSize < byteSize) {
		if (currentDrawBuffer.primitiveDrawHandle != VK_NULL_HANDLE) {
			vmaDestroyBuffer(allocator, currentDrawBuffer.primitiveDrawHandle,
							 currentDrawBuffer.primitiveDrawAllocation);
		}

		const VmaAllocationCreateInfo allocationCreateInfo {
			.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		};
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = byteSize,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
		};
		auto result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
							   &currentDrawBuffer.primitiveDrawHandle, &currentDrawBuffer.primitiveDrawAllocation, VK_NULL_HANDLE);
		vk::checkResult(result, "Failed to allocate indirect draw buffer: {}");
		vk::setDebugUtilsName(device, currentDrawBuffer.primitiveDrawHandle, fmt::format("Indirect draw buffer {}", currentFrame));
		currentDrawBuffer.primitiveDrawBufferSize = byteSize;

		// Update the descriptor
		const VkDescriptorBufferInfo bufferInfo {
			.buffer = currentDrawBuffer.primitiveDrawHandle,
			.offset = 0,
			.range = VK_WHOLE_SIZE,
		};
		const VkWriteDescriptorSet writeDescriptor {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = globalMeshBuffers.descriptors[currentFrame],
			.dstBinding = 4,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &bufferInfo,
		};
		vkUpdateDescriptorSets(device, 1, &writeDescriptor, 0, nullptr);
	}

	{
		vk::ScopedMap<glsl::PrimitiveDraw> map(allocator, currentDrawBuffer.primitiveDrawAllocation);
		auto* data = map.get();
		std::copy(draws.begin(), draws.end(), data);
	}

	// Resize the AABB visualizing draw buffer
	auto aabbByteSize = currentDrawBuffer.drawCount * sizeof(decltype(aabbDraws)::value_type);
	if (currentDrawBuffer.aabbDrawBufferSize < aabbByteSize) {
		if (currentDrawBuffer.aabbDrawHandle != VK_NULL_HANDLE) {
			vmaDestroyBuffer(allocator, currentDrawBuffer.aabbDrawHandle, currentDrawBuffer.aabbDrawAllocation);
		}

		const VmaAllocationCreateInfo allocationCreateInfo {
			.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		};
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = aabbByteSize,
			.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
		};
		auto result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
									  &currentDrawBuffer.aabbDrawHandle, &currentDrawBuffer.aabbDrawAllocation, VK_NULL_HANDLE);
		vk::checkResult(result, "Failed to allocate indirect AABB draw buffer: {}");
		vk::setDebugUtilsName(device, currentDrawBuffer.aabbDrawHandle, fmt::format("Indirect AABB draw buffer {}", currentFrame));
		currentDrawBuffer.aabbDrawBufferSize = aabbByteSize;
	}

	{
		vk::ScopedMap<VkDrawIndirectCommand> map(allocator, currentDrawBuffer.aabbDrawAllocation);
		auto* data = map.get();
		std::copy(aabbDraws.begin(), aabbDraws.end(), data);
	}
}

glm::mat4 Viewer::getCameraProjectionMatrix(fastgltf::Camera& camera) const {
	ZoneScoped;
	// The following projection matrices do not use the math defined by the glTF spec here:
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#projection-matrices
	// The reason is that Vulkan uses a different depth range to OpenGL, which has to be accounted.
	// Therefore, we always use the appropriate _ZO glm functions.
	return std::visit(fastgltf::visitor {
		[&](fastgltf::Camera::Perspective& perspective) {
			assert(swapchain.extent.width != 0 && swapchain.extent.height != 0);
			auto aspectRatio = perspective.aspectRatio.value_or(
				static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height));

			if (perspective.zfar.has_value()) {
				return glm::perspectiveRH_ZO(perspective.yfov, aspectRatio, perspective.znear, *perspective.zfar);
			} else {
				return glm::infinitePerspectiveRH_ZO(perspective.yfov, aspectRatio, perspective.znear);
			}
		},
		[&](fastgltf::Camera::Orthographic& orthographic) {
			return glm::orthoRH_ZO(-orthographic.xmag, orthographic.xmag,
								   -orthographic.ymag, orthographic.ymag,
								   orthographic.znear, orthographic.zfar);
		},
	}, camera.camera);
}

glm::mat4 reverseDepth(glm::mat4 projection) {
	// We use reversed Z, see https://iolite-engine.com/blog_posts/reverse_z_cheatsheet
	constexpr glm::mat4 reverseZ {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, -1.f, 0.0f,
		0.0f, 0.0f, 1.0f, 1.0f
	};
	return reverseZ * projection;
}

void Viewer::updateCameraBuffer(std::size_t currentFrame) {
	assert(cameraBuffers.size() > currentFrame);
	ZoneScoped;
	static constexpr auto cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
	static constexpr auto cameraRight = glm::vec3(0.0f, 0.0f, 1.0f);

	// Calculate new camera matrices, upload to GPU
	auto& cameraBuffer = cameraBuffers[currentFrame];
	vk::ScopedMap<glsl::Camera> map(allocator, cameraBuffer.allocation);
	auto& camera = *map.get();

	glm::mat4 viewMatrix(1.0f), projectionMatrix;
	if (cameraIndex.has_value()) {
		auto& scene = asset.scenes[sceneIndex];

		// Get the view matrix by traversing the node tree and finding the selected cameraNode
		auto getCameraViewMatrix = [this, &viewMatrix](std::size_t nodeIndex, glm::mat4 matrix, auto& self) -> void {
			auto& node = asset.nodes[nodeIndex];
			matrix = getTransformMatrix(node, matrix);

			if (node.cameraIndex.has_value() && &node == cameraNodes[*cameraIndex]) {
				viewMatrix = glm::affineInverse(matrix);
			}

			for (auto& child : node.children) {
				self(child, matrix, self);
			}
		};
		for (auto& sceneNode : scene.nodeIndices) {
			getCameraViewMatrix(sceneNode, glm::mat4(1.0f), getCameraViewMatrix);
		}

		projectionMatrix = getCameraProjectionMatrix(asset.cameras[*cameraIndex]);
	} else {
		// Update the accelerationVector depending on states returned by glfwGetKey.
		auto& acc = movement.accelerationVector;
		acc = glm::vec3(0.0f);
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			acc += movement.direction;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			acc -= movement.direction;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			acc += glm::normalize(glm::cross(movement.direction, cameraUp));
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			acc -= glm::normalize(glm::cross(movement.direction, cameraUp));
		}
		if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
			acc += cameraUp;
		}
		if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
			acc -= cameraUp;
		}

		float speedMultiplier = movement.speedMultiplier;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
			speedMultiplier *= 10.0f;
		}

		movement.velocity += (movement.accelerationVector * speedMultiplier);
		// Lerp the velocity to 0, adding deceleration.
		movement.velocity = movement.velocity + (5.0f * deltaTime) * (-movement.velocity);
		// Add the velocity into the position
		movement.position += movement.velocity * deltaTime;
		viewMatrix = glm::lookAtRH(movement.position, movement.position + movement.direction, cameraUp);

		// glm::perspectiveRH_ZO is correct, see https://johannesugb.github.io/gpu-programming/setting-up-a-proper-vulkan-projection-matrix/
		static constexpr auto zNear = 0.01f;
		static constexpr auto zFar = 10000.0f;
		static constexpr auto fov = glm::radians(75.0f);
		const auto aspectRatio = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);
		projectionMatrix = glm::perspectiveRH_ZO(fov, aspectRatio, zNear, zFar);
	}

	projectionMatrix[1][1] *= -1;
	camera.viewProjection = reverseDepth(projectionMatrix) * viewMatrix;

	if (!freezeCameraFrustum) {
		// This plane extraction code is from https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
		const auto& vp = camera.viewProjection;
		auto& p = camera.frustum;
		for (glm::length_t i = 0; i < 4; ++i) { p[0][i] = vp[i][3] + vp[i][0]; }
		for (glm::length_t i = 0; i < 4; ++i) { p[1][i] = vp[i][3] - vp[i][0]; }
		for (glm::length_t i = 0; i < 4; ++i) { p[2][i] = vp[i][3] + vp[i][1]; }
		for (glm::length_t i = 0; i < 4; ++i) { p[3][i] = vp[i][3] - vp[i][1]; }
		for (glm::length_t i = 0; i < 4; ++i) { p[4][i] = vp[i][3] + vp[i][2]; }
		for (glm::length_t i = 0; i < 4; ++i) { p[5][i] = vp[i][3] - vp[i][2]; }
		for (auto& plane: p) {
			plane /= glm::length(glm::vec3(plane));
			plane.w = -plane.w;
		}
	}

	// Set the sun light matrix
	glm::mat4 lightProjection = glm::orthoRH_ZO(-10.0f, 10.0f,
												-10.0f, 10.0f,
												0.01f, 100.0f);
	glm::mat4 lightView = glm::lookAtRH(lightPosition,
									  glm::vec3(0.0f, 0, 0),
									  cameraUp);
	lightProjection[1][1] *= -1;
	camera.lightSpaceMatrix = lightProjection * lightView;
}

void Viewer::updateCameraNodes(std::size_t nodeIndex) {
	ZoneScoped;
	// This function recursively traverses the node hierarchy starting with the node at nodeIndex
	// to find any nodes holding cameras.
	auto& node = asset.nodes[nodeIndex];

	if (node.cameraIndex.has_value()) {
		if (node.name.empty()) {
			// Always have a non-empty string for the ImGui UI
			node.name = std::string("Camera ") + std::to_string(cameraNodes.size());
		}
		cameraNodes.emplace_back(&node);
	}

	for (auto& child : node.children) {
		updateCameraNodes(child);
	}
}

void Viewer::renderUi() {
	ZoneScoped;
	if (ImGui::Begin("vk_gltf_viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
		ImGui::BeginDisabled(asset.scenes.size() <= 1);
		auto& sceneName = asset.scenes[sceneIndex].name;
		if (ImGui::BeginCombo("Scene", sceneName.c_str(), ImGuiComboFlags_None)) {
			for (std::size_t i = 0; i < asset.scenes.size(); ++i) {
				const bool isSelected = i == sceneIndex;
				if (ImGui::Selectable(asset.scenes[i].name.c_str(), isSelected)) {
					sceneIndex = i;

					cameraNodes.clear();
					auto& scene = asset.scenes[sceneIndex];
					for (auto& node: scene.nodeIndices) {
						updateCameraNodes(node);
					}
				}
				if (isSelected)
					ImGui::SetItemDefaultFocus();
			}

			ImGui::EndCombo();
		}
		ImGui::EndDisabled();

		ImGui::BeginDisabled(cameraNodes.empty());
		auto cameraName = cameraIndex.has_value() ? cameraNodes[*cameraIndex]->name.c_str() : "Default";
		if (ImGui::BeginCombo("Camera", cameraName, ImGuiComboFlags_None)) {
			// Default camera
			{
				const bool isSelected = !cameraIndex.has_value();
				if (ImGui::Selectable("Default", isSelected)) {
					cameraIndex.reset();
				}
				if (isSelected)
					ImGui::SetItemDefaultFocus();
			}

			for (std::size_t i = 0; i < cameraNodes.size(); ++i) {
				const bool isSelected = cameraIndex.has_value() && i == cameraIndex.value();
				if (ImGui::Selectable(cameraNodes[i]->name.c_str(), isSelected)) {
					cameraIndex = i;
				}
				if (isSelected)
					ImGui::SetItemDefaultFocus();
			}

			ImGui::EndCombo();
		}
		ImGui::EndDisabled();

		ImGui::BeginDisabled(asset.materialVariants.empty());
		const auto currentVariantName = asset.materialVariants.empty()
			? "N/A"
			: asset.materialVariants[materialVariant].c_str();
		if (ImGui::BeginCombo("Variant", currentVariantName, ImGuiComboFlags_None)) {
			for (std::size_t i = 0; i < asset.materialVariants.size(); ++i) {
				const bool isSelected = i == materialVariant;
				if (ImGui::Selectable(asset.materialVariants[i].c_str(), isSelected))
					materialVariant = i;
				if (isSelected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		ImGui::EndDisabled();

		ImGui::DragFloat("Camera speed", &movement.speedMultiplier, 0.01f, 0.05f, 10.0f, "%.2f");

		ImGui::Text("Camera position: %.2f %.2f %.2f", movement.position.x, movement.position.y, movement.position.z);
		ImGui::DragFloat3("Light position", glm::value_ptr(lightPosition));

		ImGui::Separator();

		ImGui::Text("Frametime: %.2f ms (%.2f FPS)", deltaTime * 1000, 1.f / deltaTime);

		ImGui::Separator();

		ImGui::Checkbox("Enable AABB visualization", &enableAabbVisualization);
		ImGui::Checkbox("Freeze Camera frustum", &freezeCameraFrustum);
	}
	ImGui::End();

	ImGui::Render();
}

#ifdef _MSC_VER
int wmain(int argc, wchar_t* argv[]) {
	if (argc < 2) {
		fmt::print("No glTF file specified\n");
		return -1;
	}
	std::filesystem::path gltfFile { argv[1] };
#else
int main(int argc, char* argv[]) {
    if (argc < 2) {
		fmt::print("No glTF file specified\n");
        return -1;
    }
	std::filesystem::path gltfFile { argv[1] };
#endif

	if (!std::filesystem::is_regular_file(gltfFile)) {
		return -1;
	}

	taskScheduler.Initialize();

    Viewer viewer {};

    glfwSetErrorCallback(glfwErrorCallback);

    try {
		// Load the glTF asset
		viewer.loadGltf(gltfFile);

		// Initialize GLFW
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("Failed to initialize glfw");
        }

        // Setup the Vulkan instance
        viewer.setupVulkanInstance();

        // Create the window
        auto* mainMonitor = glfwGetPrimaryMonitor();
        const auto* videoMode = glfwGetVideoMode(mainMonitor);

        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        viewer.window = glfwCreateWindow(
                static_cast<int>(static_cast<float>(videoMode->width) * 0.9f),
                static_cast<int>(static_cast<float>(videoMode->height) * 0.9f),
                "vk_viewer", nullptr, nullptr);

        if (viewer.window == nullptr) {
            throw std::runtime_error("Failed to create window");
        }

        glfwSetWindowUserPointer(viewer.window, &viewer);
        glfwSetWindowSizeCallback(viewer.window, glfwResizeCallback);

		glfwSetCursorPosCallback(viewer.window, cursorCallback);
		// glfwSetInputMode(viewer.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();

        // Create the Vulkan surface
        auto surfaceResult = glfwCreateWindowSurface(viewer.instance, viewer.window, nullptr, &viewer.surface);
		vk::checkResult(surfaceResult, "Failed to create window surface: {}");
        viewer.deletionQueue.push([&]() {
            vkDestroySurfaceKHR(viewer.instance, viewer.surface, nullptr);
        });

        // Create the Vulkan device
        viewer.setupVulkanDevice();
		viewer.timelineDeletionQueue.create(viewer.device);

		// Create the MEGA descriptor pool
		viewer.createDescriptorPool();

		// Build the camera descriptors and buffers
		viewer.buildCameraDescriptor();

		// This also creates the descriptor layout required for the pipeline creation later.
		viewer.loadGltfMeshes();

		viewer.loadGltfImages();

        // Create the swapchain
        viewer.rebuildSwapchain(videoMode->width, videoMode->height);

		// Build the mesh pipeline
        viewer.buildMeshPipeline();

		// Build the shadow map pipeline
		viewer.createShadowMapAndPipeline();

		// Resize the drawBuffers vector
		viewer.drawBuffers.resize(frameOverlap);

		// Setup ImGui. This requires the swapchain to already exist to know the format
		auto imguiResult = viewer.imgui.init(&viewer);
		vk::checkResult(imguiResult, "Failed to create ImGui rendering context: {}");
		auto& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_IsSRGB;
		io.Fonts->AddFontDefault();
		viewer.imgui.createFontAtlas();
		viewer.deletionQueue.push([&]() {
			viewer.imgui.destroy();
		});

		// Init ImGui frame data
		viewer.imgui.initFrameData(frameOverlap);

        // Creates the required fences and semaphores for frame sync
        viewer.createFrameData();

		// Set scene defaults and give every object a readable name, if required and empty.
		viewer.sceneIndex = viewer.asset.defaultScene.value_or(0);
		for (std::size_t i = 0; auto& scene : viewer.asset.scenes) {
			if (!scene.name.empty())
				continue;
			scene.name = std::string("Scene ") + std::to_string(i++);
		}

		// Initialize the glTF cameras array
		auto& scene = viewer.asset.scenes[viewer.sceneIndex];
		for (auto& node : scene.nodeIndices) {
			viewer.updateCameraNodes(node);
		}

		// The render loop
        std::size_t currentFrame = 0;
        while (glfwWindowShouldClose(viewer.window) != GLFW_TRUE) {
            if (!viewer.swapchainNeedsRebuild) {
                glfwPollEvents();
            } else {
                // This will wait until we get an event, like the resize event which will recreate the swapchain.
                glfwWaitEvents();
                continue;
            }

			FrameMarkStart("frame");

			auto currentTime = static_cast<float>(glfwGetTime());
			viewer.deltaTime = currentTime - viewer.lastFrame;
			viewer.lastFrame = currentTime;

			// New ImGui frame
			viewer.imgui.newFrame();
			ImGui::NewFrame();

			viewer.renderUi();

            currentFrame = ++currentFrame % frameOverlap;
            auto& frameSyncData = viewer.frameSyncData[currentFrame];

            // Wait for the last frame with the current index to have finished presenting, so that we can start
            // using the semaphores and command buffers.
            vkWaitForFences(viewer.device, 1, &frameSyncData.presentFinished, VK_TRUE, UINT64_MAX);
            vkResetFences(viewer.device, 1, &frameSyncData.presentFinished);

			viewer.timelineDeletionQueue.check();

			// Update the camera matrices
			viewer.updateCameraBuffer(currentFrame);

			// Update the draw-list
			viewer.updateDrawBuffer(currentFrame);

            // Reset the command pool
            auto& commandPool = viewer.frameCommandPools[currentFrame];
            vkResetCommandPool(viewer.device, commandPool.pool, 0);
            auto& cmd = commandPool.commandBuffers.front();

            // Acquire the next swapchain image
            std::uint32_t swapchainImageIndex = 0;
            auto acquireResult = vkAcquireNextImageKHR(viewer.device, viewer.swapchain, UINT64_MAX,
                                                       frameSyncData.imageAvailable,
                                                       VK_NULL_HANDLE, &swapchainImageIndex);
            if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR || acquireResult == VK_SUBOPTIMAL_KHR) {
                viewer.swapchainNeedsRebuild = true;
                continue;
            }
			vk::checkResult(acquireResult, "Failed to acquire swapchain image: {}");

            // Begin the command buffer
            VkCommandBufferBeginInfo beginInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, // We're only using once, then resetting.
            };
            vkBeginCommandBuffer(cmd, &beginInfo);

			{
				TracyVkZone(viewer.tracyCtx, cmd, "Shadow map generation");

				// Transition the shadow map image from UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL
				const VkImageMemoryBarrier2 imageBarrier {
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
					.srcStageMask = VK_PIPELINE_STAGE_2_NONE,
					.srcAccessMask = VK_ACCESS_2_NONE,
					.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
					.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = viewer.shadowMapImage,
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
						.levelCount = 1,
						.layerCount = 1,
					},
				};
				const VkDependencyInfo dependencyInfo {
					.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
					.imageMemoryBarrierCount = 1,
					.pImageMemoryBarriers = &imageBarrier,
				};
				vkCmdPipelineBarrier2(cmd, &dependencyInfo);

				const VkRenderingAttachmentInfo depthAttachment {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.imageView = viewer.shadowMapImageView,
					.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = {1.0f, 0.0f},
				};
				const VkRenderingInfo renderingInfo {
					.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
					.renderArea = {
						.offset = {},
						.extent = {
							.width = Viewer::shadowResolution.x,
							.height = Viewer::shadowResolution.y,
						},
					},
					.layerCount = 1,
					.colorAttachmentCount = 0,
					.pDepthAttachment = &depthAttachment,
				};
				vkCmdBeginRendering(cmd, &renderingInfo);

				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, viewer.shadowMapPipeline);

				std::array<VkDescriptorSet, 2> descriptorBinds {{
					viewer.cameraBuffers[currentFrame].cameraSet, // Set 0
					viewer.globalMeshBuffers.descriptors[currentFrame], // Set 1
				}};
				// Bind the camera descriptor set
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, viewer.shadowMapPipelineLayout,
										0, static_cast<std::uint32_t>(descriptorBinds.size()), descriptorBinds.data(),
										0, nullptr);

				const VkViewport viewport = {
					.x = 0.0F,
					.y = 0.0F,
					.width = static_cast<float>(renderingInfo.renderArea.extent.width),
					.height = static_cast<float>(renderingInfo.renderArea.extent.height),
					.minDepth = 0.0F,
					.maxDepth = 1.0F,
				};
				vkCmdSetViewport(cmd, 0, 1, &viewport);

				const VkRect2D scissor = renderingInfo.renderArea;
				vkCmdSetScissor(cmd, 0, 1, &scissor);

				vkCmdDrawMeshTasksIndirectEXT(cmd,
							  viewer.drawBuffers[currentFrame].primitiveDrawHandle, 0,
							  viewer.drawBuffers[currentFrame].drawCount,
							  sizeof(glsl::PrimitiveDraw));

				vkCmdEndRendering(cmd);
			}

            {
				TracyVkZone(viewer.tracyCtx, cmd, "Mesh shading");

				// Transition the swapchain image from UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL for rendering
				// Transition the depth image from UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL
				// Transition the shadow map image from DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
				std::array<VkImageMemoryBarrier2, 3> imageBarriers = {{
					{
						.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
						.srcStageMask = VK_PIPELINE_STAGE_2_NONE,
						.srcAccessMask = VK_ACCESS_2_NONE,
						.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
						.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
						.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
						.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = viewer.swapchainImages[swapchainImageIndex],
						.subresourceRange = {
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
							.levelCount = 1,
							.layerCount = 1,
						},
					},
					{
						.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
						.srcStageMask = VK_PIPELINE_STAGE_2_NONE,
						.srcAccessMask = VK_ACCESS_2_NONE,
						.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
						.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
						.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
						.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = viewer.depthImage,
						.subresourceRange = {
							.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
							.levelCount = 1,
							.layerCount = 1,
						},
					},
					{
						.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
						.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
						.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
						.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
						.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
						.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
						.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = viewer.shadowMapImage,
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

				const VkRenderingAttachmentInfo swapchainAttachment {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.imageView = viewer.swapchainImageViews[swapchainImageIndex],
					.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				};
				const VkRenderingAttachmentInfo depthAttachment {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.imageView = viewer.depthImageView,
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
						.extent = viewer.swapchain.extent,
					},
					.layerCount = 1,
					.colorAttachmentCount = 1,
					.pColorAttachments = &swapchainAttachment,
					.pDepthAttachment = &depthAttachment,
				};
				vkCmdBeginRendering(cmd, &renderingInfo);

				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, viewer.meshPipeline);

				std::array<VkDescriptorSet, 3> descriptorBinds {{
					viewer.cameraBuffers[currentFrame].cameraSet, // Set 0
					viewer.globalMeshBuffers.descriptors[currentFrame], // Set 1
 					viewer.materialSet, // Set 2
				}};
				// Bind the camera descriptor set
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, viewer.meshPipelineLayout,
										0, static_cast<std::uint32_t>(descriptorBinds.size()), descriptorBinds.data(),
										0, nullptr);

				const VkViewport viewport = {
					.x = 0.0F,
					.y = 0.0F,
					.width = static_cast<float>(viewer.swapchain.extent.width),
					.height = static_cast<float>(viewer.swapchain.extent.height),
					.minDepth = 0.0F,
					.maxDepth = 1.0F,
				};
				vkCmdSetViewport(cmd, 0, 1, &viewport);

				const VkRect2D scissor = renderingInfo.renderArea;
				vkCmdSetScissor(cmd, 0, 1, &scissor);

				vkCmdDrawMeshTasksIndirectEXT(cmd,
											  viewer.drawBuffers[currentFrame].primitiveDrawHandle, 0,
											  viewer.drawBuffers[currentFrame].drawCount,
											  sizeof(glsl::PrimitiveDraw));

				if (viewer.enableAabbVisualization) {
					// Visualize the AABBs. We don't need to rebind descriptor sets as we use the same pipeline layout as the mesh pipeline
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, viewer.aabbVisualizingPipeline);

					vkCmdDrawIndirect(cmd, viewer.drawBuffers[currentFrame].aabbDrawHandle, 0,
									  viewer.drawBuffers[currentFrame].drawCount,
									  sizeof(VkDrawIndirectCommand));
				}

				vkCmdEndRendering(cmd);
            }

			// Draw UI
			{
				TracyVkZone(viewer.tracyCtx, cmd, "ImGui rendering");

				auto extent = glm::u32vec2(viewer.swapchain.extent.width, viewer.swapchain.extent.height);
				viewer.imgui.draw(cmd, viewer.swapchainImageViews[swapchainImageIndex], extent, currentFrame);
			}

            // Transition the swapchain image from COLOR_ATTACHMENT -> PRESENT_SRC_KHR
			const VkImageMemoryBarrier2 swapchainImageBarrier {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				.dstAccessMask = VK_ACCESS_2_NONE,
				.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = viewer.swapchainImages[swapchainImageIndex],
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.levelCount = 1,
					.layerCount = 1,
				},
			};

			const VkDependencyInfo dependencyInfo {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &swapchainImageBarrier,
			};
            vkCmdPipelineBarrier2(cmd, &dependencyInfo);

			// Always collect at the end of the main command buffer.
			TracyVkCollect(viewer.tracyCtx, cmd);

            vkEndCommandBuffer(cmd);

            // Submit the command buffer
			const VkSemaphoreSubmitInfo waitSemaphoreInfo {
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
				.semaphore = frameSyncData.imageAvailable,
				.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			};
			const VkCommandBufferSubmitInfo cmdSubmitInfo {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
				.commandBuffer = cmd,
			};
			std::array<VkSemaphoreSubmitInfo, 2> signalSemaphoreInfos = {{
				{
					.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
					.semaphore = frameSyncData.renderingFinished,
					.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				},
				{
					.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
					.semaphore = viewer.timelineDeletionQueue.getSemaphoreHandle(),
					.value = viewer.timelineDeletionQueue.getSemaphoreCounter(),
					.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				},
			}};
			const VkSubmitInfo2 submitInfo {
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
				.waitSemaphoreInfoCount = 1,
				.pWaitSemaphoreInfos = &waitSemaphoreInfo,
				.commandBufferInfoCount = 1,
				.pCommandBufferInfos = &cmdSubmitInfo,
				.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signalSemaphoreInfos.size()),
				.pSignalSemaphoreInfos = signalSemaphoreInfos.data(),
			};

			// Submit & present
			{
				std::lock_guard lock(*viewer.graphicsQueue.lock);
				auto submitResult = vkQueueSubmit2(viewer.graphicsQueue.handle, 1, &submitInfo,
												   frameSyncData.presentFinished);
				vk::checkResult(submitResult, "Failed to submit to queue: {}");

				// Present the rendered image
				const VkPresentInfoKHR presentInfo {
					.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
					.waitSemaphoreCount = 1,
					.pWaitSemaphores = &frameSyncData.renderingFinished,
					.swapchainCount = 1,
					.pSwapchains = &viewer.swapchain.swapchain,
					.pImageIndices = &swapchainImageIndex,
				};
				auto presentResult = vkQueuePresentKHR(viewer.graphicsQueue.handle, &presentInfo);
				if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
					viewer.swapchainNeedsRebuild = true;
					continue;
				}
				vk::checkResult(presentResult, "Failed to present to queue: {}");
			}

			FrameMarkEnd("frame");
        }
    } catch (const vulkan_error& error) {
		fmt::print("{}: {}\n", error.what(), error.what_result());
    } catch (const std::runtime_error& error) {
		fmt::print("{}\n", error.what());
    }

	if (volkGetLoadedDevice() != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(viewer.device); // Make sure everything is done

		for (auto& frame : viewer.frameSyncData) {
			vkWaitForFences(viewer.device, 1, &frame.presentFinished, VK_TRUE, UINT64_MAX);
		}

		taskScheduler.WaitforAll();

		// Destroy the samplers
		for (auto& sampler: viewer.samplers) {
			vkDestroySampler(viewer.device, sampler, VK_NULL_HANDLE);
		}

		// Destroy the images
		for (auto& image: viewer.images) {
			vkDestroyImageView(viewer.device, image.imageView, VK_NULL_HANDLE);
			vmaDestroyImage(viewer.allocator, image.image, image.allocation);
		}

		// Destroy the draw buffers
		for (auto& drawBuffer: viewer.drawBuffers) {
			vmaDestroyBuffer(viewer.allocator, drawBuffer.aabbDrawHandle, drawBuffer.aabbDrawAllocation);
			vmaDestroyBuffer(viewer.allocator, drawBuffer.primitiveDrawHandle, drawBuffer.primitiveDrawAllocation);
		}

		for (auto& view : viewer.swapchainImageViews)
			vkDestroyImageView(viewer.device, view, nullptr);
		vkb::destroy_swapchain(viewer.swapchain);
		vkDestroyImageView(viewer.device, viewer.depthImageView, VK_NULL_HANDLE);
		vmaDestroyImage(viewer.allocator, viewer.depthImage, viewer.depthImageAllocation);

		// Destroys everything. We leave this out of the try-catch block to make sure it gets executed.
		viewer.timelineDeletionQueue.destroy();
		viewer.flushObjects();
	}

    glfwDestroyWindow(viewer.window);
    glfwTerminate();

    taskScheduler.WaitforAllAndShutdown();

    return 0;
}
