#include <functional>
#include <iostream>
#include <string_view>

#include <TaskScheduler.h>

#include <tracy/Tracy.hpp>

#include <vulkan/vk.hpp>
#include <VkBootstrap.h>
#include <vulkan/vma.hpp>
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

	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);
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

static constexpr auto cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	void* ptr = glfwGetWindowUserPointer(window);
	auto& movement = static_cast<Viewer*>(ptr)->movement;

	auto& acceleration = movement.accelerationVector;
	switch (key) {
		case GLFW_KEY_W:
			acceleration += movement.direction;
			break;
		case GLFW_KEY_S:
			acceleration -= movement.direction;
			break;
		case GLFW_KEY_D:
			acceleration += glm::normalize(glm::cross(movement.direction, cameraUp));
			break;
		case GLFW_KEY_A:
			acceleration -= glm::normalize(glm::cross(movement.direction, cameraUp));
			break;
		default:
			break;
	}
}

VkBool32 vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT          messageSeverity,
                             VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
                             const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
                             void*                                            pUserData) {
	fmt::print("{}\n", pCallbackData->pMessage);
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
    if (auto result = volkInitialize(); result != VK_SUCCESS) {
        throw vulkan_error("No compatible Vulkan loader or driver found.", result);
    }

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
	};

	const VkPhysicalDeviceVulkan11Features vulkan11Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.shaderDrawParameters = VK_TRUE,
	};

	const VkPhysicalDeviceVulkan12Features vulkan12Features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.storageBuffer8BitAccess = VK_TRUE,
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

	// Get the main graphics queue
	auto graphicsQueueResult = device.get_queue(vkb::QueueType::graphics);
	checkResult(graphicsQueueResult);
	graphicsQueue = Queue {
		.handle = graphicsQueueResult.value(),
		.lock = std::make_unique<std::mutex>(),
	};

	// Get the transfer queue handles
	auto transferQueueIndexRes = device.get_dedicated_queue_index(vkb::QueueType::transfer);
	checkResult(transferQueueIndexRes);

	auto transferQueueFamilyIndex = transferQueueIndexRes.value();
	transferQueues.resize(queueFamilies[transferQueueFamilyIndex].queueCount);
	for (std::size_t i = 0; auto& queue : transferQueues) {
		queue.lock = std::make_unique<std::mutex>();
		vkGetDeviceQueue(device, transferQueueFamilyIndex, i++, &queue.handle);
	}

	auto threadCount = std::thread::hardware_concurrency();

	// Create the transfer command pools
	uploadCommandPools.resize(threadCount);
	for (auto& commandPool : uploadCommandPools) {
		const VkCommandPoolCreateInfo commandPoolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = transferQueueFamilyIndex,
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
	deletionQueue.push([this]() {
		for (auto& cmdPool : uploadCommandPools)
			vkDestroyCommandPool(device, cmdPool.pool, nullptr);
	});

	// Create the transfer fences
	uploadFences.resize(threadCount);
	for (auto& fence : uploadFences) {
		// Create the submit fence
		const VkFenceCreateInfo fenceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		auto fenceResult = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
		vk::checkResult(fenceResult, "Failed to create buffer upload fence: {}");
	}
	deletionQueue.push([this]() {
		for (auto& fence : uploadFences)
			vkDestroyFence(device, fence, nullptr);
	});
}

void Viewer::rebuildSwapchain(std::uint32_t width, std::uint32_t height) {
	ZoneScoped;
    vkb::SwapchainBuilder swapchainBuilder(device);
    auto swapchainResult = swapchainBuilder
            .set_old_swapchain(swapchain)
			.set_desired_extent(width, height)
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
		.usage = VMA_MEMORY_USAGE_GPU_ONLY,
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
	vk::checkResult(result, "Failed to create descriptor pool");

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
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
			.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		};
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(Camera),
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
	vk::checkResult(result, "Failed to create mesh pipeline layout");
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

    auto builder = vk::GraphicsPipelineBuilder(device, nullptr)
        .setPipelineCount(2);

	// Create mesh pipeline
	builder
        .setPipelineLayout(0, meshPipelineLayout)
        .pushPNext(0, &renderingCreateInfo)
        .addDynamicState(0, VK_DYNAMIC_STATE_SCISSOR)
        .addDynamicState(0, VK_DYNAMIC_STATE_VIEWPORT)
        .setBlendAttachment(0, &blendAttachment)
        .setTopology(0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setDepthState(0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL)
        .setRasterState(0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .setMultisampleCount(0, VK_SAMPLE_COUNT_1_BIT)
        .setScissorCount(0, 1U)
        .setViewportCount(0, 1U)
        .addShaderStage(0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main")
        .addShaderStage(0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule, "main")
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
		.setDepthState(1, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL)
		.setRasterState(1, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
		.setMultisampleCount(1, VK_SAMPLE_COUNT_1_BIT)
		.setScissorCount(1, 1U)
		.setViewportCount(1, 1U)
		.addShaderStage(1, VK_SHADER_STAGE_FRAGMENT_BIT, aabbFragModule, "main")
		.addShaderStage(1, VK_SHADER_STAGE_VERTEX_BIT, aabbVertModule, "main");

	std::array<VkPipeline, 2> pipelines {};
    result = builder.build(pipelines.data());
    if (result != VK_SUCCESS) {
        throw vulkan_error("Failed to create mesh and aabb visualizing pipeline", result);
    }

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
        VkSemaphoreCreateInfo semaphoreCreateInfo = {};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        auto semaphoreResult = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.imageAvailable);
		vk::checkResult(semaphoreResult, "Failed to create image semaphore");
		vk::setDebugUtilsName(device, frame.imageAvailable, "Image acquire semaphore");

        semaphoreResult = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.renderingFinished);
		vk::checkResult(semaphoreResult, "Failed to create rendering semaphore");
		vk::setDebugUtilsName(device, frame.renderingFinished, "Rendering finished semaphore");

        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        auto fenceResult = vkCreateFence(device, &fenceCreateInfo, nullptr, &frame.presentFinished);
		vk::checkResult(fenceResult, "Failed to create present fence");
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
        VkCommandPoolCreateInfo commandPoolInfo = {};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolInfo.queueFamilyIndex = 0;
        auto createResult = vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frame.pool);
		vk::checkResult(createResult, "Failed to create frame command pool");

        VkCommandBufferAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = frame.pool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        frame.commandBuffers.resize(1);
        auto allocateResult = vkAllocateCommandBuffers(device, &allocateInfo, frame.commandBuffers.data());
		vk::checkResult(allocateResult, "Failed to allocate frame command buffers");
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

	std::vector<Vertex> globalVertices;
	std::vector<Meshlet> globalMeshlets;
	std::vector<unsigned int> globalMeshletVertices;
	std::vector<unsigned char> globalMeshletTriangles;

	CompressedBufferDataAdapter adapter;
	if (!adapter.decompress(asset))
		throw std::runtime_error("Failed to decompress all glTF buffers");

	// Generate the meshes
	for (auto& gltfMesh : asset.meshes) {
		auto& mesh = meshes.emplace_back();

		// We need this as we require pointer-stability for the generate task.
		mesh.primitives.reserve(gltfMesh.primitives.size());
		for (auto& gltfPrimitive : gltfMesh.primitives) {
			if (!gltfPrimitive.indicesAccessor.has_value()) {
				throw std::runtime_error("Every primitive should have a value.");
			}

			auto* positionIt = gltfPrimitive.findAttribute("POSITION");
			if (positionIt == gltfPrimitive.attributes.end()) {
				throw std::runtime_error("Every primitive has a POSITION attribute.");
			}

			auto& primitive = mesh.primitives.emplace_back();
			if (gltfPrimitive.materialIndex.has_value()) {
				primitive.materialIndex = gltfPrimitive.materialIndex.value() + numDefaultMaterials;
			} else {
				primitive.materialIndex = 0;
			}

			// Copy the positions and indices
			auto& posAccessor = asset.accessors[positionIt->second];
			std::vector<Vertex> vertices; vertices.reserve(posAccessor.count);
			fastgltf::iterateAccessor<glm::vec3>(asset, posAccessor, [&](glm::vec3 val) {
				auto& vertex = vertices.emplace_back();
				vertex.position = glm::vec4(val, 1.0f);
				vertex.color = glm::vec4(1.0f);
				vertex.uv = glm::vec2(0.0f);
			}, adapter);

			auto& indicesAccessor = asset.accessors[gltfPrimitive.indicesAccessor.value()];
			std::vector<std::uint32_t> indices(indicesAccessor.count);
			fastgltf::copyFromAccessor<std::uint32_t>(asset, indicesAccessor, indices.data(), adapter);

			if (auto* colorAttribute = gltfPrimitive.findAttribute("COLOR_0"); colorAttribute != gltfPrimitive.attributes.end()) {
				// The glTF spec allows VEC3 and VEC4 for COLOR_n, with VEC3 data having to be extended with 1.0f for the fourth component.
				auto& colorAccessor = asset.accessors[colorAttribute->second];
				if (colorAccessor.type == fastgltf::AccessorType::Vec4) {
					fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, asset.accessors[colorAttribute->second], [&](glm::vec4 val, std::size_t idx) {
						vertices[idx].color = val;
					}, adapter);
				} else if (colorAccessor.type == fastgltf::AccessorType::Vec3) {
					fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[colorAttribute->second], [&](glm::vec3 val, std::size_t idx) {
						vertices[idx].color = glm::vec4(val, 1.0f);
					}, adapter);
				}
			}

			if (auto* uvAttribute = gltfPrimitive.findAttribute("TEXCOORD_0"); uvAttribute != gltfPrimitive.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, asset.accessors[uvAttribute->second], [&](glm::vec2 val, std::size_t idx) {
					vertices[idx].uv = val;
				}, adapter);
			}

			// These are the optimal values for NVIDIA. What about the others?
			const std::size_t maxVertices = 64;
			const std::size_t maxTriangles = 124; // NVIDIA wants 126 but meshopt only allows 124 for alignment reasons.
			const float coneWeight = 0.0f; // We leave this as 0 because we're not using cluster cone culling.

			// TODO: Meshlet generation and data resizing should probably be threaded, too.
			std::size_t maxMeshlets = meshopt_buildMeshletsBound(indicesAccessor.count, maxVertices, maxTriangles);
			std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
			std::vector<unsigned int> meshlet_vertices(maxMeshlets * maxVertices);
			std::vector<unsigned char> meshlet_triangles(maxMeshlets * maxTriangles * 3);

			// Generate the meshlets for this primitive
			primitive.meshlet_count = meshopt_buildMeshlets(
				meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
				indices.data(), indices.size(),
				&vertices[0].position.x, vertices.size(), sizeof(decltype(vertices)::value_type),
				maxVertices, maxTriangles, coneWeight);

			primitive.descOffset = globalMeshlets.size();
			primitive.vertexIndicesOffset = globalMeshletVertices.size();
			primitive.triangleIndicesOffset = globalMeshletTriangles.size();
			primitive.verticesOffset = globalVertices.size();

			// Trim the buffers
			const auto& lastMeshlet = meshlets[primitive.meshlet_count - 1];
			meshlet_vertices.resize(lastMeshlet.vertex_count + lastMeshlet.vertex_offset);
			meshlet_triangles.resize(((lastMeshlet.triangle_count * 3 + 3) & ~3) + lastMeshlet.triangle_offset);
			meshlets.resize(primitive.meshlet_count);

			std::vector<Meshlet> finalMeshlets; finalMeshlets.reserve(primitive.meshlet_count);
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

				glm::vec3 center = (min + max) * 0.5f;
				finalMeshlets.emplace_back(Meshlet {
					.meshlet = meshlet,
					.aabbExtents = max - center,
					.aabbCenter = center,
				});
			}

			// Append the data to the end of the global buffers.
			globalVertices.insert(globalVertices.end(), vertices.begin(), vertices.end());
			globalMeshlets.insert(globalMeshlets.end(), finalMeshlets.begin(), finalMeshlets.end());
			globalMeshletVertices.insert(globalMeshletVertices.end(), meshlet_vertices.begin(), meshlet_vertices.end());
			globalMeshletTriangles.insert(globalMeshletTriangles.end(), meshlet_triangles.begin(), meshlet_triangles.end());
		}
	}

	uploadMeshlets(globalMeshlets, globalMeshletVertices, globalMeshletTriangles, globalVertices);
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
	auto fence = uploadFences[taskScheduler.GetThreadNum()];
	vkResetFences(device, 1, &fence);
	vkResetCommandBuffer(cmd, 0);

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
		auto submitResult = vkQueueSubmit(queue.handle, 1, &submitInfo, fence);
		vk::checkResult(submitResult, "Failed to submit buffer copy: {}");
	}

	// We always wait for this operation to complete here, to free up the command buffer and fence for the next iteration.
	vkWaitForFences(device, 1, &fence, VK_TRUE, 9999999999);

	vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
}

void Viewer::uploadImageToDevice(std::size_t stagingBufferSize, std::function<void(VkCommandBuffer, VkBuffer, VmaAllocation)> commands) {
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

	// Reset fences and command buffers.
	auto cmd = uploadCommandPools[taskScheduler.GetThreadNum()].buffer;
	auto fence = uploadFences[taskScheduler.GetThreadNum()];
	vkResetFences(device, 1, &fence);
	vkResetCommandBuffer(cmd, 0);

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
		vkQueueSubmit2(queue.handle, 1, &submitInfo, fence);
	}

	// We always wait for this operation to complete here, to free up the command buffer and fence for the next iteration.
	vkWaitForFences(device, 1, &fence, VK_TRUE, 9999999999);

	// Destroy the staging buffer
	vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
}

void Viewer::uploadMeshlets(std::vector<Meshlet>& meshlets,
							std::vector<unsigned int>& meshletVertices, std::vector<unsigned char>& meshletTriangles,
							std::vector<Vertex>& vertices) {
	ZoneScoped;
	{
		// Create the meshlet description buffer
		uploadBufferToDevice(std::as_bytes(std::span(meshlets.begin(), meshlets.end())),
							 &globalMeshBuffers.descHandle, &globalMeshBuffers.descAllocation);
		vk::setDebugUtilsName(device, globalMeshBuffers.descHandle, "Meshlet descriptions");
	}
	{
		// Create the vertex index buffer
		uploadBufferToDevice(std::as_bytes(std::span(meshletVertices.begin(), meshletVertices.end())),
							 &globalMeshBuffers.vertexIndiciesHandle, &globalMeshBuffers.vertexIndiciesAllocation);
		vk::setDebugUtilsName(device, globalMeshBuffers.vertexIndiciesHandle, "Meshlet vertex indices");
	}
	{
		// Create the meshlet description buffer
		uploadBufferToDevice(std::as_bytes(std::span(meshletTriangles.begin(), meshletTriangles.end())),
							 &globalMeshBuffers.triangleIndicesHandle, &globalMeshBuffers.triangleIndicesAllocation);
		vk::setDebugUtilsName(device, globalMeshBuffers.triangleIndicesHandle, "Meshlet triangle indices");
	}
	{
		// Create the vertex buffer
		uploadBufferToDevice(std::as_bytes(std::span(vertices.begin(), vertices.end())),
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

struct ImageLoadTask : public enki::ITaskSet {
	Viewer* viewer;
	std::size_t imageIdx;

	explicit ImageLoadTask(Viewer* viewer, std::size_t imageIdx) noexcept : viewer(viewer), imageIdx(imageIdx) {
		m_SetSize = 1;
	}

	void loadWithStb(std::span<std::uint8_t> encodedImageData, std::uint32_t threadnum) const {
		ZoneScoped;
		static constexpr VkFormat imageFormat = VK_FORMAT_R8G8B8A8_SRGB;
		static constexpr auto channels = 4;

		// Load and decode the image data using stbi
		int width = 0, height = 0, nrChannels = 0;
		auto* ptr = stbi_load_from_memory(encodedImageData.data(), static_cast<int>(encodedImageData.size()), &width, &height, &nrChannels, channels);

		std::span<std::byte> imageData = std::span(reinterpret_cast<std::byte*>(ptr),
												   width * height * sizeof(std::byte) * channels);

		auto& sampledImage = viewer->images[imageIdx];

		const VkImageCreateInfo imageInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = imageFormat,
			.extent = {
				.width = static_cast<std::uint32_t>(width),
				.height = static_cast<std::uint32_t>(height),
				.depth = 1,
			},
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
		const VmaAllocationCreateInfo allocationInfo {
			.usage = VMA_MEMORY_USAGE_GPU_ONLY,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		};
		vmaCreateImage(viewer->allocator, &imageInfo, &allocationInfo,
					   &sampledImage.image, &sampledImage.allocation, nullptr);

		const VkImageViewCreateInfo imageViewInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = sampledImage.image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = imageInfo.format,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.layerCount = 1,
			},
		};
		vkCreateImageView(viewer->device, &imageViewInfo, VK_NULL_HANDLE, &sampledImage.imageView);

		viewer->uploadImageToDevice(imageData.size_bytes(), [&](VkCommandBuffer cmd, VkBuffer stagingBuffer, VmaAllocation stagingAllocation) {
			{
				vk::ScopedMap map(viewer->allocator, stagingAllocation);
				std::memcpy(map.get(), imageData.data(), imageData.size_bytes());
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
				.image = sampledImage.image,
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
			vkCmdCopyBufferToImage(cmd, stagingBuffer, sampledImage.image,
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

		stbi_image_free(imageData.data());
	}

	void loadDds(std::span<std::uint8_t> encodedImageData, std::uint32_t threadnum) const {
		ZoneScoped;
		dds::Image image;
		auto ddsResult = dds::readImage(encodedImageData.data(), encodedImageData.size_bytes(), &image);
		if (ddsResult != dds::ReadResult::Success) {
			// TODO
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
			.usage = VMA_MEMORY_USAGE_GPU_ONLY,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		};
		vmaCreateImage(viewer->allocator, &imageInfo, &allocationInfo,
					   &sampledImage.image, &sampledImage.allocation, nullptr);

		auto imageViewInfo = dds::getVulkanImageViewCreateInfo(&image, vkFormat);
		imageViewInfo.image = sampledImage.image;
		vkCreateImageView(viewer->device, &imageViewInfo, nullptr, &sampledImage.imageView);

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

			// For each mip, submit a command buffer copying the staging buffer to the device image
			glm::u32vec2 extent(image.width, image.height);
			for (std::uint32_t i = 0; i < image.numMips; ++i) {
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
					.image = sampledImage.image,
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = i,
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

				// Transition the image into the destinationLayout
				imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
				imageBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
				imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
				imageBarrier.dstAccessMask = VK_ACCESS_2_NONE;
				imageBarrier.oldLayout = imageBarrier.newLayout;
				imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				vkCmdPipelineBarrier2(cmd, &dependencyInfo);

				extent /= 2;
			}
		});
	}

	void load(std::span<std::uint8_t> data, fastgltf::MimeType mimeType, std::uint32_t threadnum) {
		switch (mimeType) {
			case fastgltf::MimeType::PNG:
			case fastgltf::MimeType::JPEG: {
				loadWithStb(data, threadnum);
				break;
			}
			case fastgltf::MimeType::DDS: {
				loadDds(data, threadnum);
				break;
			}
			default: {
				assert(false);
				break;
			}
		}
	}

	// We'll use the range to operate over multiple images
	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override {
		ZoneScoped;
		auto& image = viewer->asset.images[imageIdx - Viewer::numDefaultTextures];

		std::visit(fastgltf::visitor {
			[](auto& arg) {
				assert(false && "Got unexpected image data source.");
				return;
			},
			[&](fastgltf::sources::Array& array) {
				load(std::span(array.bytes.data(), array.bytes.size()), array.mimeType, threadnum);
			},
			[&](fastgltf::sources::BufferView& bufferView) {
				auto& view = viewer->asset.bufferViews[bufferView.bufferViewIndex];
				auto& buffer = viewer->asset.buffers[view.bufferIndex];
				std::visit(fastgltf::visitor {
					[](auto& arg) {
						assert(false && "Got unexpected image data source.");
						return;
					},
					[&](fastgltf::sources::Array& array) {
						load(std::span(array.bytes.data(), array.bytes.size()), bufferView.mimeType, threadnum);
					}
				}, buffer.data);
			}
		}, image.data);

		auto& sampledImage = viewer->images[imageIdx];
		vk::setDebugUtilsName(viewer->device, sampledImage.image, image.name.c_str());
		vk::setDebugUtilsName(viewer->device, sampledImage.imageView, fmt::format("View of {}", image.name.c_str()));
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
	vkCreateImageView(device, &imageViewInfo, VK_NULL_HANDLE, &defaultTexture.imageView);
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
	images.resize(numDefaultTextures + asset.images.size());
	std::vector<std::unique_ptr<ImageLoadTask>> loadTasks; loadTasks.reserve(asset.images.size());
	for (auto i = numDefaultTextures; i < asset.images.size() + numDefaultTextures; ++i) {
		auto task = std::make_unique<ImageLoadTask>(this, i);
		taskScheduler.AddTaskSetToPipe(task.get());
		loadTasks.emplace_back(std::move(task));
	}

	createDefaultImages();

	// Create the material descriptor layout
	// TODO: We currently use a fixed size for the descriptorCount of the image samplers.
	//       Using VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT we could change the descriptor size.
	// TODO: We currently dont use UPDATE_AFTER_BIND, making us use either frameOverlap count of sets, or restricting
	//       us to a fixed set of textures for rendering.
	std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings = {{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = static_cast<std::uint32_t>(asset.textures.size() + numDefaultTextures),
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		}
	}};
	std::array<VkDescriptorBindingFlags, layoutBindings.max_size()> layoutBindingFlags = {{
		0,
		0, // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
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
		.maxLod = VK_LOD_CLAMP_NONE,
	};
	result = vkCreateSampler(device, &samplerInfo, nullptr, &samplers[0]);

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
	}

	// Finish all texture decode and upload tasks
	for (auto& task : loadTasks) {
		taskScheduler.WaitforTask(task.get());
	}

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
		.dstBinding = 1,
		.dstArrayElement = 0U,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &infos.back(),
	});

	// Write the glTF textures
	for (std::size_t i = 0; i < asset.textures.size(); ++i) {
		auto& texture = asset.textures[i];

		// Well map a glTF texture to a single combined image sampler
		infos.emplace_back(VkDescriptorImageInfo {
			.sampler = samplers[texture.samplerIndex.has_value() ? *texture.samplerIndex + numDefaultSamplers : 0],
			.imageView = images[texture.imageIndex.has_value() ? *texture.imageIndex + numDefaultTextures : 0].imageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		});

		writes.emplace_back(VkWriteDescriptorSet {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = materialSet,
			.dstBinding = 1,
			.dstArrayElement = static_cast<std::uint32_t>(i),
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &infos.back(),
		});
	}
	vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
}

void Viewer::loadGltfMaterials() {
	ZoneScoped;
	// Create the material buffer data
	std::vector<Material> materials; materials.reserve(asset.materials.size());

	// Add the default material
	materials.emplace_back(Material {
		.albedoFactor = glm::vec4(1.0f),
		.albedoIndex = 0,
		.alphaCutoff = 0.5f,
	});

	for (auto& gltfMaterial : asset.materials) {
		auto& mat = materials.emplace_back();
		mat.albedoFactor = glm::make_vec4(gltfMaterial.pbrData.baseColorFactor.data());
		if (gltfMaterial.pbrData.baseColorTexture.has_value()) {
			mat.albedoIndex = gltfMaterial.pbrData.baseColorTexture->textureIndex;
		} else {
			mat.albedoIndex = 0;
		}
		mat.alphaCutoff = gltfMaterial.alphaCutoff;
	}

	// Create the material buffer
	const VmaAllocationCreateInfo allocationCreateInfo {
		.usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
		.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
	};
	const VkBufferCreateInfo bufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = materials.size() * sizeof(decltype(materials)::value_type),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	auto result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
								  &materialBuffer, &materialAllocation, VK_NULL_HANDLE);
	vk::checkResult(result, "Failed to allocate material buffer");
	vk::setDebugUtilsName(device, materialBuffer, "Material buffer");

	deletionQueue.push([this]() {
		vmaDestroyBuffer(allocator, materialBuffer, materialAllocation);
	});

	// Copy the material data to the buffer
	{
		vk::ScopedMap<Material> map(allocator, materialAllocation);
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

glm::mat4 Viewer::getCameraProjectionMatrix(fastgltf::Camera& camera) const {
	ZoneScoped;
	// The following matrix math is for the projection matrices as defined by the glTF spec:
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#projection-matrices
	return std::visit(fastgltf::visitor {
		[&](fastgltf::Camera::Perspective& perspective) {
			glm::mat4x4 mat(0.0f);

			assert(swapchain.extent.width != 0 && swapchain.extent.height != 0);
			auto aspectRatio = perspective.aspectRatio.value_or(
				static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height));
			mat[0][0] = 1.f / (aspectRatio * tan(0.5f * perspective.yfov));
			mat[1][1] = 1.f / (tan(0.5f * perspective.yfov));
			mat[2][3] = -1;

			if (perspective.zfar.has_value()) {
				// Finite projection matrix
				mat[2][2] = (*perspective.zfar + perspective.znear) / (perspective.znear - *perspective.zfar);
				mat[3][2] = (2 * *perspective.zfar * perspective.znear) / (perspective.znear - *perspective.zfar);
			} else {
				// Infinite projection matrix
				mat[2][2] = -1;
				mat[3][2] = -2 * perspective.znear;
			}
			return mat;
		},
		[&](fastgltf::Camera::Orthographic& orthographic) {
			glm::mat4x4 mat(1.0f);
			mat[0][0] = 1.f / orthographic.xmag;
			mat[1][1] = 1.f / orthographic.ymag;
			mat[2][2] = 2.f / (orthographic.znear - orthographic.zfar);
			mat[3][2] = (orthographic.zfar + orthographic.znear) / (orthographic.znear - orthographic.zfar);
			return mat;
		},
	}, camera.camera);
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

void Viewer::drawNode(std::vector<PrimitiveDraw>& cmd, std::vector<VkDrawIndirectCommand>& aabbCmd, std::size_t nodeIndex, glm::mat4 matrix) {
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

void Viewer::drawMesh(std::vector<PrimitiveDraw>& cmd, std::vector<VkDrawIndirectCommand>& aabbCmd, std::size_t meshIndex, glm::mat4 matrix) {
	assert(meshes.size() > meshIndex);
	ZoneScoped;

	auto& mesh = meshes[meshIndex];

	for (auto& primitive : mesh.primitives) {
		auto& draw = cmd.emplace_back();

		// Dispatch so many groups that we only have to use up to 128 16-bit indices in the shared payload.
		const VkDrawMeshTasksIndirectCommandEXT indirectCommand {
			.groupCountX = static_cast<std::uint32_t>((primitive.meshlet_count + 128 - 1) / 128),
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
		draw.materialIndex = primitive.materialIndex;

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

	std::vector<PrimitiveDraw> draws;
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
			.usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
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
		vk::ScopedMap<PrimitiveDraw> map(allocator, currentDrawBuffer.primitiveDrawAllocation);
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
			.usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
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

void Viewer::updateCameraBuffer(std::size_t currentFrame) {
	assert(cameraBuffers.size() > currentFrame);
	ZoneScoped;

	// Calculate new camera matrices, upload to GPU
	auto& cameraBuffer = cameraBuffers[currentFrame];
	vk::ScopedMap<Camera> map(allocator, cameraBuffer.allocation);
	auto& camera = *map.get();

	if (cameraIndex.has_value()) {
		auto& scene = asset.scenes[sceneIndex];

		// Get the view matrix by traversing the node tree and finding the selected cameraNode
		glm::mat4 viewMatrix(1.0f);
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

		auto projectionMatrix = getCameraProjectionMatrix(asset.cameras[*cameraIndex]);

		projectionMatrix[1][1] *= -1;
		camera.viewProjectionMatrix = projectionMatrix * viewMatrix;
	} else {
		movement.velocity += (movement.accelerationVector * movement.speedMultiplier);
		// Lerp the velocity to 0, adding deceleration.
		movement.velocity = movement.velocity + (5.0f * deltaTime) * (-movement.velocity);
		// Add the velocity into the position
		movement.position += movement.velocity * deltaTime;
		auto viewMatrix = glm::lookAt(movement.position, movement.position + movement.direction, cameraUp);

		static constexpr auto zNear = 0.01f;
		static constexpr auto zFar = 10000.0f;
		static constexpr auto fov = glm::radians(75.0f);
		const auto aspectRatio = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);
		auto projectionMatrix = glm::perspective(fov, aspectRatio, zNear, zFar);

		// Invert the Y-Axis to use the same coordinate system as glTF.
		projectionMatrix[1][1] *= -1;
		camera.viewProjectionMatrix = projectionMatrix * viewMatrix;
	}

	if (!freezeCameraFrustum) {
		// This plane extraction code is from https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
		const auto& vp = camera.viewProjectionMatrix;
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

		ImGui::DragFloat("Camera speed", &movement.speedMultiplier, 0.1f, 0.5f, 50.0f, "%.0f");

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

		glfwSetKeyCallback(viewer.window, keyCallback);
		glfwSetCursorPosCallback(viewer.window, cursorCallback);
		// glfwSetInputMode(viewer.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();

        // Create the Vulkan surface
        auto surfaceResult = glfwCreateWindowSurface(viewer.instance, viewer.window, nullptr, &viewer.surface);
        if (surfaceResult != VK_SUCCESS) {
            throw vulkan_error("Failed to create window surface", surfaceResult);
        }
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
				// Reset the acceleration before updating it through input events
				viewer.movement.accelerationVector = glm::vec3(0.0f);

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
            if (acquireResult != VK_SUCCESS) {
                throw vulkan_error("Failed to acquire swapchain image", acquireResult);
            }

            // Begin the command buffer
            VkCommandBufferBeginInfo beginInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, // We're only using once, then resetting.
            };
            vkBeginCommandBuffer(cmd, &beginInfo);

            {
				TracyVkZone(viewer.tracyCtx, cmd, "Mesh shading");

				// Transition the swapchain image from UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL for rendering
				// Transition the depth image from UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL
				std::array<VkImageMemoryBarrier2, 2> imageBarriers = {{
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
					.clearValue = {1.0f, 0.0f},
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
											  sizeof(PrimitiveDraw));

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
				if (submitResult != VK_SUCCESS) {
					throw vulkan_error("Failed to submit to queue", submitResult);
				}

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
				if (presentResult != VK_SUCCESS) {
					throw vulkan_error("Failed to present to queue", presentResult);
				}
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
