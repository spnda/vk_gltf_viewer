#include <deque>
#include <functional>
#include <iostream>
#include <string_view>

#include <TaskScheduler.h>

#include "stb_image.h"

#include <tracy/Tracy.hpp>

#include <vulkan/vk.hpp>
#include <VkBootstrap.h>
#include <vulkan/vma.hpp>
#include <tracy/TracyVulkan.hpp>

#include <vulkan/pipeline_builder.hpp>
#include <vulkan/debug_utils.hpp>

#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>

#include <glm/glm.hpp>
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
#include <vk_gltf_viewer/buffer_uploader.hpp>
#include <vk_gltf_viewer/scheduler.hpp>

enki::TaskScheduler taskScheduler;

struct Viewer;

void glfwErrorCallback(int errorCode, const char* description) {
    if (errorCode != GLFW_NO_ERROR) {
        std::cout << "GLFW error: " << errorCode;

        if (description != nullptr) {
            std::cout << ": " << description;
        }

        std::cout << '\n';
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
    std::cout << pCallbackData->pMessage << '\n';
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
    deletionQueue.push([&]() {
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
		.hostQueryReset = VK_TRUE,
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

	VmaAllocatorCreateFlags allocatorFlags;
	auto& physicalDevice = selectionResult.value();
	if (physicalDevice.enable_extension_if_present(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
		allocatorFlags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	// Generate the queue descriptions for vkb. Use one queue for everything except
	// for dedicated transfer queues.
	std::vector<vkb::CustomQueueDescription> queues;
	auto queueFamilies = physicalDevice.get_queue_families();
	for (uint32_t i = 0; i < queueFamilies.size(); i++) {
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
    deletionQueue.push([&]() {
        vkb::destroy_device(device);
    });

    volkLoadDevice(device);

	// Initialize the tracy context
	tracyCtx = TracyVkContextHostCalibrated(device.physical_device, device, vkResetQueryPool, vkGetPhysicalDeviceCalibrateableTimeDomainsEXT, vkGetCalibratedTimestampsEXT);

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

	deletionQueue.push([&]() {
		vmaDestroyAllocator(allocator);
	});

	// Get the queues
    auto graphicsQueueRes = device.get_queue(vkb::QueueType::graphics);
    checkResult(graphicsQueueRes);
    graphicsQueue = graphicsQueueRes.value();

	auto transferQueueIndexRes = device.get_dedicated_queue_index(vkb::QueueType::transfer);
	checkResult(transferQueueIndexRes);

	BufferUploader::getInstance().init(device, allocator, transferQueueIndexRes.value(),
									   queueFamilies[transferQueueIndexRes.value()].queueCount);
	deletionQueue.push([&]() {
		BufferUploader::getInstance().destroy();
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

    // The swapchain is not added to the deletionQueue, as it gets recreated throughout the application's lifetime.
    vkb::destroy_swapchain(swapchain);
    swapchain = swapchainResult.value();

    auto imageResult = swapchain.get_images();
    checkResult(imageResult);
    swapchainImages = std::move(imageResult.value());

    auto imageViewResult = swapchain.get_image_views();
    checkResult(imageViewResult);
    swapchainImageViews = std::move(imageViewResult.value());

	if (depthImage != VK_NULL_HANDLE) {
		vkDestroyImageView(device, depthImageView, VK_NULL_HANDLE);
		vmaDestroyImage(allocator, depthImage, depthImageAllocation);
	}

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
}

void Viewer::createDescriptorPool() {
	ZoneScoped;
	auto& limits = device.physical_device.properties.limits;

	// TODO: Do we want to always just use the *mega* descriptor pool?
	std::array<VkDescriptorPoolSize, 3> sizes = {{
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = min(1048576U, limits.maxDescriptorSetUniformBuffers),
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = min(1048576U, limits.maxDescriptorSetStorageBuffers),
		},
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = min(16536U, limits.maxDescriptorSetSampledImages),
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
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
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
    std::array<VkDescriptorSetLayout, 3> layouts = {{ cameraSetLayout, meshletSetLayout, materialSetLayout }};
    const VkPipelineLayoutCreateInfo layoutCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 0,
    };
    auto result = vkCreatePipelineLayout(device, &layoutCreateInfo, VK_NULL_HANDLE, &meshPipelineLayout);
	vk::checkResult(result, "Failed to create mesh pipeline layout");
	vk::setDebugUtilsName(device, meshPipelineLayout, "Mesh shading pipeline layout");

    // Load the mesh pipeline shaders
    VkShaderModule fragModule, meshModule, taskModule;
    vk::loadShaderModule("main.frag.glsl.spv", device, &fragModule);
    vk::loadShaderModule("main.mesh.glsl.spv", device, &meshModule);
	vk::loadShaderModule("main.task.glsl.spv", device, &taskModule);

    // Build the mesh pipeline
    const auto colorAttachmentFormat = swapchain.image_format;
	const auto depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    const VkPipelineRenderingCreateInfo renderingCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &colorAttachmentFormat,
			.depthAttachmentFormat = depthAttachmentFormat,
    };

    const VkPipelineColorBlendAttachmentState blendAttachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    auto builder = vk::GraphicsPipelineBuilder(device, nullptr)
        .setPipelineCount(1)
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
		.addShaderStage(0, VK_SHADER_STAGE_TASK_BIT_EXT, taskModule, "main");

    result = builder.build(&meshPipeline);
    if (result != VK_SUCCESS) {
        throw vulkan_error("Failed to create mesh pipeline", result);
    }

    // We don't need the shader modules after creating the pipeline anymore.
    vkDestroyShaderModule(device, fragModule, VK_NULL_HANDLE);
    vkDestroyShaderModule(device, meshModule, VK_NULL_HANDLE);
	vkDestroyShaderModule(device, taskModule, VK_NULL_HANDLE);

    deletionQueue.push([&]() {
        vkDestroyPipeline(device, meshPipeline, VK_NULL_HANDLE);
        vkDestroyPipelineLayout(device, meshPipelineLayout, VK_NULL_HANDLE);
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

        deletionQueue.push([&]() {
            vkDestroyFence(device, frame.presentFinished, nullptr);
            vkDestroySemaphore(device, frame.renderingFinished, nullptr);
            vkDestroySemaphore(device, frame.imageAvailable, nullptr);
        });
    }

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
        deletionQueue.push([&]() {
            vkDestroyCommandPool(device, frame.pool, nullptr);
        });
    }
}

class Base64DecodeTask final : public enki::ITaskSet {
    std::string_view encodedData;
    uint8_t* outputData;

public:
    // Arbitrarily chosen 1MB. Lower values will cause too many tasks to spawn, slowing down the process.
    // Perhaps even larger values would be necessary, as even this gets decoded incredibly quick and the
    // overhead of launching threaded tasks gets noticeable.
    static constexpr const size_t minBase64DecodeSetSize = 1 * 1024 * 1024; // 1MB.

    explicit Base64DecodeTask(uint32_t dataSize, std::string_view encodedData, uint8_t* outputData)
            : enki::ITaskSet(dataSize, minBase64DecodeSetSize), encodedData(encodedData), outputData(outputData) {}

    void ExecuteRange(enki::TaskSetPartition range, uint32_t threadnum) override {
        fastgltf::base64::decode_inplace(encodedData.substr(static_cast<size_t>(range.start) * 4, static_cast<size_t>(range.end) * 4),
                                         &outputData[range.start * 3], 0);
    }
};

// The custom base64 callback for fastgltf to multithread base64 decoding, to divide the (possibly) large
// input buffer into smaller chunks that can be worked on by multiple threads.
void multithreadedBase64Decoding(std::string_view encodedData, uint8_t* outputData,
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
    auto* editor = static_cast<Viewer*>(userPointer);
    taskScheduler.AddTaskSetToPipe(&task);

    // Finally, wait for all other tasks to finish. enkiTS will use this thread as well to process the tasks.
    taskScheduler.WaitforTask(&task);
}

void Viewer::loadGltf(std::string_view file) {
	ZoneScoped;
    const std::filesystem::path filePath(file);

    fastgltf::GltfDataBuffer fileBuffer;
    if (!fileBuffer.loadFromFile(filePath)) {
        throw std::runtime_error("Failed to load file");
    }

    fastgltf::Parser parser(fastgltf::Extensions::KHR_mesh_quantization);
    parser.setUserPointer(this);
    parser.setBase64DecodeCallback(multithreadedBase64Decoding);

	// TODO: Extract buffer/image loading into async functions in the future
	static constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages | fastgltf::Options::GenerateMeshIndices;

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
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
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
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT,
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
	vk::setDebugUtilsName(device, cameraSetLayout, "Mesh shader pipeline descriptor layout");

	deletionQueue.push([&]() {
		vkDestroyDescriptorSetLayout(device, meshletSetLayout, nullptr);
	});

	std::vector<Vertex> globalVertices;
	std::vector<meshopt_Meshlet> globalMeshlets;
	std::vector<unsigned int> globalMeshletVertices;
	std::vector<unsigned char> globalMeshletTriangles;

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
			});

			auto& indicesAccessor = asset.accessors[gltfPrimitive.indicesAccessor.value()];
			std::vector<std::uint32_t> indices(indicesAccessor.count);
			fastgltf::copyFromAccessor<std::uint32_t>(asset, indicesAccessor, indices.data());

			if (auto* colorAttribute = gltfPrimitive.findAttribute("COLOR_0"); colorAttribute != gltfPrimitive.attributes.end()) {
				// The glTF spec allows VEC3 and VEC4 for COLOR_n, with VEC3 data having to be extended with 1.0f for the fourth component.
				auto& colorAccessor = asset.accessors[colorAttribute->second];
				if (colorAccessor.type == fastgltf::AccessorType::Vec4) {
					fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, asset.accessors[colorAttribute->second], [&](glm::vec4 val, std::size_t idx) {
						vertices[idx].color = val;
					});
				} else if (colorAccessor.type == fastgltf::AccessorType::Vec3) {
					fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[colorAttribute->second], [&](glm::vec3 val, std::size_t idx) {
						vertices[idx].color = glm::vec4(val, 1.0f);
					});
				}
			}

			if (auto* uvAttribute = gltfPrimitive.findAttribute("TEXCOORD_0"); uvAttribute != gltfPrimitive.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, asset.accessors[uvAttribute->second], [&](glm::vec2 val, std::size_t idx) {
					vertices[idx].uv = val;
				});
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

			// Append the data to the end of the global buffers.
			globalVertices.insert(globalVertices.end(), vertices.begin(), vertices.end());
			globalMeshlets.insert(globalMeshlets.end(), meshlets.begin(), meshlets.end());
			globalMeshletVertices.insert(globalMeshletVertices.end(), meshlet_vertices.begin(), meshlet_vertices.end());
			globalMeshletTriangles.insert(globalMeshletTriangles.end(), meshlet_triangles.begin(), meshlet_triangles.end());
		}
	}

	uploadMeshlets(globalMeshlets, globalMeshletVertices, globalMeshletTriangles, globalVertices);
}

VkResult Viewer::createGpuTransferBuffer(std::size_t byteSize, VkBuffer *buffer, VmaAllocation *allocation) noexcept {
	const VmaAllocationCreateInfo allocationCreateInfo {
		.usage = VMA_MEMORY_USAGE_GPU_ONLY,
	};
	const VkBufferCreateInfo bufferCreateInfo {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = byteSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	return vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo,
						   buffer, allocation, VK_NULL_HANDLE);
}

void Viewer::uploadMeshlets(std::vector<meshopt_Meshlet>& meshlets,
							std::vector<unsigned int>& meshletVertices, std::vector<unsigned char>& meshletTriangles,
							std::vector<Vertex>& vertices) {
	ZoneScoped;
	std::vector<std::unique_ptr<BufferUploadTask>> uploadTasks;
	{
		// Create the meshlet description buffer
		auto result = createGpuTransferBuffer(meshlets.size() * sizeof(std::remove_reference_t<decltype(meshlets)>::value_type),
											  &globalMeshBuffers.descHandle, &globalMeshBuffers.descAllocation);
		vk::checkResult(result, "Failed to allocate meshlet description buffer: {}");
		vk::setDebugUtilsName(device, globalMeshBuffers.descHandle, "Meshlet descriptions");

		auto task = BufferUploader::getInstance().uploadToBuffer(
			std::as_bytes(std::span{meshlets.begin(), meshlets.end()}),
			globalMeshBuffers.descHandle);
		uploadTasks.emplace_back(std::move(task));
	}
	{
		// Create the vertex index buffer
		auto result = createGpuTransferBuffer(meshletVertices.size() * sizeof(std::remove_reference_t<decltype(meshletVertices)>::value_type),
											  &globalMeshBuffers.vertexIndiciesHandle, &globalMeshBuffers.vertexIndiciesAllocation);
		vk::checkResult(result, "Failed to allocate vertex index buffer: {}");
		vk::setDebugUtilsName(device, globalMeshBuffers.vertexIndiciesHandle, "Meshlet vertex indices");

		auto task = BufferUploader::getInstance().uploadToBuffer(
			std::as_bytes(std::span{meshletVertices.begin(), meshletVertices.end()}),
			globalMeshBuffers.vertexIndiciesHandle);
		uploadTasks.emplace_back(std::move(task));
	}
	{
		// Create the meshlet description buffer
		auto result = createGpuTransferBuffer(meshletTriangles.size() * sizeof(std::remove_reference_t<decltype(meshletTriangles)>::value_type),
											  &globalMeshBuffers.triangleIndicesHandle, &globalMeshBuffers.triangleIndicesAllocation);
		vk::checkResult(result, "Failed to allocate triangle index buffer: {}");
		vk::setDebugUtilsName(device, globalMeshBuffers.triangleIndicesHandle, "Meshlet triangle indices");

		auto task = BufferUploader::getInstance().uploadToBuffer(
			std::as_bytes(std::span{meshletTriangles.begin(), meshletTriangles.end()}),
			globalMeshBuffers.triangleIndicesHandle);
		uploadTasks.emplace_back(std::move(task));
	}
	{
		// Create the vertex buffer
		auto result = createGpuTransferBuffer(vertices.size() * sizeof(std::remove_reference_t<decltype(vertices)>::value_type),
											  &globalMeshBuffers.verticesHandle, &globalMeshBuffers.verticesAllocation);
		vk::checkResult(result, "Failed to allocate vertex buffer: {}");
		vk::setDebugUtilsName(device, globalMeshBuffers.verticesHandle, "Meshlet vertices");

		auto task = BufferUploader::getInstance().uploadToBuffer(
			std::as_bytes(std::span{vertices.begin(), vertices.end()}),
			globalMeshBuffers.verticesHandle);
		uploadTasks.emplace_back(std::move(task));
	}

	deletionQueue.push([&]() {
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

	for (auto& task : uploadTasks) {
		taskScheduler.WaitforTask(task.get());
	}
}

#include <stb_image.h>

struct ImageLoadTask : public enki::ITaskSet {
	Viewer* viewer;
	std::size_t imageIdx;

	explicit ImageLoadTask(Viewer* viewer, std::size_t imageIdx) noexcept : viewer(viewer), imageIdx(imageIdx) {
		m_SetSize = 1;
	}

	// We'll use the range to operate over multiple images
	void ExecuteRange(enki::TaskSetPartition range, uint32_t threadnum) override {
		ZoneScoped;
		// m_SetSize = 1, so range will always be 0,1
		auto& image = viewer->asset.images[imageIdx - Viewer::numDefaultTextures];

		uint8_t* imageData = nullptr;
		VkExtent3D imageExtent;
		imageExtent.depth = 1;
		static constexpr auto channels = 4;

		// Load and decode the image data using stbi from the various sources.
		std::visit(fastgltf::visitor {
			[](auto arg) { },
			[&](fastgltf::sources::Array& vector) {
				int width = 0, height = 0, nrChannels = 0;
				imageData = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, channels);
				imageExtent.width = width;
				imageExtent.height = height;
			},
			[&](fastgltf::sources::BufferView& view) {
				auto& bufferView = viewer->asset.bufferViews[view.bufferViewIndex];
				auto& buffer = viewer->asset.buffers[bufferView.bufferIndex];
				// Yes, we've already loaded every buffer into some GL buffer. However, with GL it's simpler
				// to just copy the buffer data again for the texture. Besides, this is just an example.
				std::visit(fastgltf::visitor {
					// We only care about VectorWithMime here, because we specify LoadExternalBuffers, meaning
					// all buffers are already loaded into a vector.
					[](auto& arg) {},
					[&](fastgltf::sources::Array& vector) {
						int width = 0, height = 0, nrChannels = 0;
						imageData = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), &width, &height, &nrChannels, channels);
						imageExtent.width = width;
						imageExtent.height = height;
					}
				}, buffer.data);
			},
		}, image.data);

		SampledImage& sampledImage = viewer->images[imageIdx];

		const VkImageCreateInfo imageInfo {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = VK_FORMAT_R8G8B8A8_SRGB,
			.extent = imageExtent,
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

		// Create and schedule the ImageUploadTask.
		auto data = std::span<const std::byte> { reinterpret_cast<std::byte*>(imageData),
			imageExtent.width * imageExtent.height * sizeof(std::byte) * channels };
		ImageUploadTask uploadTask(data, sampledImage.image, imageExtent, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		taskScheduler.AddTaskSetToPipe(&uploadTask);

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

		taskScheduler.WaitforTask(&uploadTask);

		stbi_image_free(imageData);
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
		.usage = VMA_MEMORY_USAGE_GPU_ONLY,
	};
	auto& defaultTexture = images[0];
	auto result = vmaCreateImage(allocator, &imageInfo, &allocationInfo,
				   &defaultTexture.image, &defaultTexture.allocation, nullptr);
	vk::checkResult(result, "Failed to create default image: {}");

	// We use R8G8B8A8_UNORM, so we need to use 8-bit integers for the colors here.
	std::array<std::uint8_t, 4> white {{ 255, 255, 255, 255 }};
	auto data = std::span<const std::byte> { reinterpret_cast<std::byte*>(white.data()), sizeof(white) };
	ImageUploadTask uploadTask(data, defaultTexture.image, imageInfo.extent, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	taskScheduler.AddTaskSetToPipe(&uploadTask);

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

	taskScheduler.WaitforTask(&uploadTask);
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
	deletionQueue.push([&]() {
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
		.alphaCutoff = 0.5,
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

	deletionQueue.push([&]() {
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

void Viewer::drawNode(std::vector<PrimitiveDraw>& cmd, std::size_t nodeIndex, glm::mat4 matrix) {
	assert(asset.nodes.size() > nodeIndex);
	ZoneScoped;

	auto& node = asset.nodes[nodeIndex];
	matrix = getTransformMatrix(node, matrix);

	if (node.meshIndex.has_value()) {
		drawMesh(cmd, node.meshIndex.value(), matrix);
	}

	for (auto& child : node.children) {
		drawNode(cmd, child, matrix);
	}
}

void Viewer::drawMesh(std::vector<PrimitiveDraw>& cmd, std::size_t meshIndex, glm::mat4 matrix) {
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
	}
}

void Viewer::updateDrawBuffer(std::size_t currentFrame) {
	assert(drawBuffers.size() > currentFrame);
	ZoneScoped;

	auto& currentDrawBuffer = drawBuffers[currentFrame];

	std::vector<PrimitiveDraw> draws;

	std::size_t sceneIdx = asset.defaultScene.value_or(0);
	auto& scene = asset.scenes[sceneIdx];
	for (auto& nodeIdx : scene.nodeIndices) {
		drawNode(draws, nodeIdx, glm::mat4(1.0f));
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
		vk::setDebugUtilsName(device, globalMeshBuffers.descHandle, fmt::format("Indirect draw buffer {}", currentFrame));
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

	vk::ScopedMap<PrimitiveDraw> map(allocator, currentDrawBuffer.primitiveDrawAllocation);
	auto* data = map.get();
	std::copy(draws.begin(), draws.end(), data);
}

void Viewer::updateCameraBuffer(std::size_t currentFrame) {
	assert(cameraBuffers.size() > currentFrame);
	ZoneScoped;

	// Calculate new camera matrices, upload to GPU
	auto& cameraBuffer = cameraBuffers[currentFrame];
	vk::ScopedMap<Camera> map(allocator, cameraBuffer.allocation);
	auto& camera = *map.get();

	movement.velocity += (movement.accelerationVector * 2.0f);
	// Lerp the velocity to 0, adding deceleration.
	movement.velocity = movement.velocity + (2.0f * deltaTime) * (-movement.velocity);
	// Add the velocity into the position
	movement.position += movement.velocity * deltaTime;
	auto viewMatrix = glm::lookAt(movement.position, movement.position + movement.direction, cameraUp);

	auto projectionMatrix = glm::perspective(glm::radians(75.0f),
											 static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height),
											 0.01f, 1000.0f);

	// Invert the Y-Axis to use the same coordinate system as glTF.
	projectionMatrix[1][1] *= -1;
	camera.viewProjectionMatrix = projectionMatrix * viewMatrix;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "No gltf file specified." << '\n';
        return -1;
    }
    auto gltfFile = std::string_view { argv[1] };

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
		glfwSetInputMode(viewer.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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

        // Creates the required fences and semaphores for frame sync
        viewer.createFrameData();

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

            currentFrame = ++currentFrame % frameOverlap;
            auto& frameSyncData = viewer.frameSyncData[currentFrame];

            // Wait for the last frame with the current index to have finished presenting, so that we can start
            // using the semaphores and command buffers.
            vkWaitForFences(viewer.device, 1, &frameSyncData.presentFinished, VK_TRUE, UINT64_MAX);
            vkResetFences(viewer.device, 1, &frameSyncData.presentFinished);

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
						.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
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
						.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
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

				vkCmdEndRendering(cmd);
            }

            // Transition the swapchain image from (possibly) UNDEFINED -> PRESENT_SRC_KHR
			const VkImageMemoryBarrier2 swapchainImageBarrier {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
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
            const VkPipelineStageFlags submitWaitStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			const VkSubmitInfo submitInfo {
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &frameSyncData.imageAvailable,
				.pWaitDstStageMask = &submitWaitStages,
				.commandBufferCount = 1,
				.pCommandBuffers = &cmd,
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &frameSyncData.renderingFinished,
			};
            auto submitResult = vkQueueSubmit(viewer.graphicsQueue, 1, &submitInfo, frameSyncData.presentFinished);
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
            auto presentResult = vkQueuePresentKHR(viewer.graphicsQueue, &presentInfo);
            if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
                viewer.swapchainNeedsRebuild = true;
                continue;
            }
            if (presentResult != VK_SUCCESS) {
                throw vulkan_error("Failed to present to queue", presentResult);
            }

			FrameMarkEnd("frame");
        }
    } catch (const vulkan_error& error) {
        std::cerr << error.what() << ": " << error.what_result() << '\n';
    } catch (const std::runtime_error& error) {
        std::cerr << error.what() << '\n';
    }

    vkDeviceWaitIdle(viewer.device); // Make sure everything is done

    taskScheduler.WaitforAll();

	// Destroy the samplers
	for (auto& sampler : viewer.samplers) {
		vkDestroySampler(viewer.device, sampler, VK_NULL_HANDLE);
	}

	// Destroy the images
	for (auto& image : viewer.images) {
		vkDestroyImageView(viewer.device, image.imageView, VK_NULL_HANDLE);
		vmaDestroyImage(viewer.allocator, image.image, image.allocation);
	}

	// Destroy the draw buffers
	for (auto& drawBuffer : viewer.drawBuffers) {
		vmaDestroyBuffer(viewer.allocator, drawBuffer.primitiveDrawHandle, drawBuffer.primitiveDrawAllocation);
	}

    // Destroys everything. We leave this out of the try-catch block to make sure it gets executed.
    // The swapchain is the only exception, as that gets recreated within the render loop. Managing it
    // with this paradigm is quite hard.
    viewer.swapchain.destroy_image_views(viewer.swapchainImageViews);
    vkb::destroy_swapchain(viewer.swapchain);
	vkDestroyImageView(viewer.device, viewer.depthImageView, VK_NULL_HANDLE);
	vmaDestroyImage(viewer.allocator, viewer.depthImage, viewer.depthImageAllocation);

    viewer.flushObjects();
    glfwDestroyWindow(viewer.window);
    glfwTerminate();

    taskScheduler.WaitforAllAndShutdown();

    return 0;
}
