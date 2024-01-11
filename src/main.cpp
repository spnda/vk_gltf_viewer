#include <deque>
#include <functional>
#include <iostream>
#include <string_view>

#include <TaskScheduler.h>

#include "stb_image.h"

#include <vulkan/vk.hpp>
#include <VkBootstrap.h>
#include <vulkan/vma.hpp>

#include <vulkan/pipeline_builder.hpp>

#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fastgltf/base64.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/parser.hpp>

#include <vk_gltf_viewer/viewer.hpp>

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
    VkPhysicalDeviceVulkan12Features vulkan12Features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .bufferDeviceAddress = VK_TRUE,
    };

    VkPhysicalDeviceVulkan13Features vulkan13Features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
		.maintenance4 = VK_TRUE,
    };

    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
        .meshShader = VK_TRUE,
    };

	// Select an appropriate device with the given requirements.
    vkb::PhysicalDeviceSelector selector(instance);
    auto selectionResult = selector
            .set_surface(surface)
            .set_minimum_version(1, 3) // We want Vulkan 1.3.
            .set_required_features_12(vulkan12Features)
            .set_required_features_13(vulkan13Features)
            .add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME)
            .add_required_extension_features(meshShaderFeatures)
            .require_present()
            .require_dedicated_transfer_queue()
            .select();
    checkResult(selectionResult);

    vkb::DeviceBuilder deviceBuilder(selectionResult.value());
    auto creationResult = deviceBuilder
            .build();
    checkResult(creationResult);

    device = creationResult.value();
    deletionQueue.push([&]() {
        vkb::destroy_device(device);
    });

    volkLoadDevice(device);

	// Create the VMA allocator
	// Create the VMA allocator object
	const VmaVulkanFunctions vmaFunctions {
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
	};
	const VmaAllocatorCreateInfo allocatorInfo {
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
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
    auto graphicsQueue = device.get_queue(vkb::QueueType::graphics);
    checkResult(graphicsQueue);
    this->graphicsQueue = graphicsQueue.value();

    auto transferQueue = device.get_dedicated_queue(vkb::QueueType::transfer);
    checkResult(transferQueue);
    this->transferQueue = transferQueue.value();
}

void Viewer::rebuildSwapchain(std::uint32_t width, std::uint32_t height) {
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
}

void Viewer::createDescriptorPool() {
	// TODO: Update this according to the actual data passed to the shader. Currently this is
	//       1 uniform and 1 storage buffer, but that might change.
	std::array<VkDescriptorPoolSize, 2> sizes = {{
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
		}
	}};
	const VkDescriptorPoolCreateInfo poolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = frameOverlap,
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
	// The camera descriptor layout
	std::array<VkDescriptorSetLayoutBinding, 1> layoutBindings = {{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT,
		},
	}};
	const VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = static_cast<std::uint32_t>(layoutBindings.size()),
		.pBindings = layoutBindings.data(),
	};
	auto result = vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfo,
											  VK_NULL_HANDLE, &cameraSetLayout);
	vk::checkResult(result, "Failed to create camera descriptor set layout: {}");

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
	vk::checkResult(result, "Failed to allocate camera descriptor set: {}");

	// Generate descriptor writes to update the descriptor
	std::vector<VkWriteDescriptorSet> descriptorWrites;
	descriptorWrites.reserve(frameOverlap);
	cameraBuffers.resize(frameOverlap);
	for (auto& cameraBuffer : cameraBuffers) {
		// Copy the created camera sets into the every cameraBuffer structs.
		// Small hack to use descriptorWrites.size() here but it represents the same index.
		cameraBuffer.cameraSet = sets[descriptorWrites.size()];

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
		const VkWriteDescriptorSet descriptorWrite = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = cameraBuffer.cameraSet,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &bufferInfo,
		};
		descriptorWrites.emplace_back(descriptorWrite);
	}

	// Update the descriptors
	vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(descriptorWrites.size()),
						   descriptorWrites.data(), 0, nullptr);
}

void Viewer::buildMeshPipeline() {
    // Build the mesh pipeline layout
    std::array<VkDescriptorSetLayout, 1> layouts = {{ cameraSetLayout }};
    const VkPipelineLayoutCreateInfo layoutCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    auto result = vkCreatePipelineLayout(device, &layoutCreateInfo, VK_NULL_HANDLE, &meshPipelineLayout);
    if (result != VK_SUCCESS) {
        throw vulkan_error("Failed to create mesh pipeline layout", result);
    }

    // Load the mesh pipeline shaders
    VkShaderModule fragModule, meshModule;
    vk::loadShaderModule("main.frag.glsl.spv", device, &fragModule);
    vk::loadShaderModule("main.mesh.glsl.spv", device, &meshModule);

    // Build the mesh pipeline
    const auto colorAttachmentFormat = swapchain.image_format;
    const VkPipelineRenderingCreateInfo renderingCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &colorAttachmentFormat,
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
        .setDepthState(0, VK_FALSE, VK_FALSE, VK_COMPARE_OP_GREATER_OR_EQUAL)
        .setRasterState(0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE)
        .setMultisampleCount(0, VK_SAMPLE_COUNT_1_BIT)
        .setScissorCount(0, 1U)
        .setViewportCount(0, 1U)
        .addShaderStage(0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main")
        .addShaderStage(0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule, "main");

    result = builder.build(&meshPipeline);
    if (result != VK_SUCCESS) {
        throw vulkan_error("Failed to create mesh pipeline", result);
    }

    // We don't need the shader modules after creating the pipeline anymore.
    vkDestroyShaderModule(device, fragModule, VK_NULL_HANDLE);
    vkDestroyShaderModule(device, meshModule, VK_NULL_HANDLE);

    deletionQueue.push([&]() {
        vkDestroyPipeline(device, meshPipeline, VK_NULL_HANDLE);
        vkDestroyPipelineLayout(device, meshPipelineLayout, VK_NULL_HANDLE);
    });
}

void Viewer::createFrameData() {
    frameSyncData.resize(frameOverlap);
    for (auto& frame : frameSyncData) {
        VkSemaphoreCreateInfo semaphoreCreateInfo = {};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        auto semaphoreResult = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.imageAvailable);
        if (semaphoreResult != VK_SUCCESS) {
            throw vulkan_error("Failed to create image semaphore", semaphoreResult);
        }
        semaphoreResult = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.renderingFinished);
        if (semaphoreResult != VK_SUCCESS) {
            throw vulkan_error("Failed to create rendering semaphore", semaphoreResult);
        }

        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        auto fenceResult = vkCreateFence(device, &fenceCreateInfo, nullptr, &frame.presentFinished);
        if (fenceResult != VK_SUCCESS) {
            throw vulkan_error("Failed to create present fence", fenceResult);
        }

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
        if (createResult != VK_SUCCESS) {
            throw vulkan_error("Failed to create command pool", createResult);
        }

        VkCommandBufferAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = frame.pool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        frame.commandBuffers.resize(1);
        auto allocateResult = vkAllocateCommandBuffers(device, &allocateInfo, frame.commandBuffers.data());
        if (allocateResult != VK_SUCCESS) {
            throw vulkan_error("Failed to allocate command buffers", allocateResult);
        }
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
    editor->taskScheduler.AddTaskSetToPipe(&task);

    // Finally, wait for all other tasks to finish. enkiTS will use this thread as well to process the tasks.
    editor->taskScheduler.WaitforTask(&task);
}


void loadGltf(Viewer& viewer, std::string_view file) {
    const std::filesystem::path filePath(file);

    fastgltf::GltfDataBuffer fileBuffer;
    if (!fileBuffer.loadFromFile(filePath)) {
        throw std::runtime_error("Failed to load file");
    }

    fastgltf::Parser parser(fastgltf::Extensions::KHR_mesh_quantization);
    parser.setUserPointer(&viewer);
    parser.setBase64DecodeCallback(multithreadedBase64Decoding);

    auto asset = parser.loadGltf(&fileBuffer, filePath.parent_path());
    if (asset.error() != fastgltf::Error::None) {
        auto message = fastgltf::getErrorMessage(asset.error());
        throw std::runtime_error(std::string("Failed to load glTF") + std::string(message));
    }

    viewer.asset = std::move(asset.get());

    // We'll always do additional validation
    if (auto validation = fastgltf::validate(viewer.asset); validation != fastgltf::Error::None) {
        auto message = fastgltf::getErrorMessage(asset.error());
        throw std::runtime_error(std::string("Asset failed validation") + std::string(message));
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "No gltf file specified." << '\n';
        return -1;
    }
    auto gltfFile = std::string_view { argv[1] };

    Viewer viewer {};
    viewer.taskScheduler.Initialize();

    glfwSetErrorCallback(glfwErrorCallback);

    try {
        // Initialize GLFW
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("Failed to initialize glfw");
        }

        // Load the glTF asset
        loadGltf(viewer, gltfFile);

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

        // Create the swapchain
        viewer.rebuildSwapchain(videoMode->width, videoMode->height);

		viewer.createDescriptorPool();
		viewer.buildCameraDescriptor();
        viewer.buildMeshPipeline();

        // Creates the required fences and semaphores for frame sync
        viewer.createFrameData();

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

            currentFrame = ++currentFrame % frameOverlap;
            auto& frameSyncData = viewer.frameSyncData[currentFrame];

            // Wait for the last frame with the current index to have finished presenting, so that we can start
            // using the semaphores and command buffers.
            vkWaitForFences(viewer.device, 1, &frameSyncData.presentFinished, VK_TRUE, UINT64_MAX);
            vkResetFences(viewer.device, 1, &frameSyncData.presentFinished);

			// Calculate new camera matrices, upload to GPU
			{
				auto& cameraBuffer = viewer.cameraBuffers[currentFrame];
				vk::ScopedMap<Camera> map(viewer.allocator, cameraBuffer.allocation);
				auto& camera = *map.get();

				// TODO: Allow camera movement
				auto viewMatrix = glm::lookAt(glm::vec3(5.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

				auto projectionMatrix = glm::perspective(glm::radians(75.0f),
														 static_cast<float>(viewer.swapchain.extent.width) / static_cast<float>(viewer.swapchain.extent.height),
														 0.01f, 1000.0f);

				camera.viewProjectionMatrix = projectionMatrix * viewMatrix;
			}

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
				// Transition the swapchain image from UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL for rendering
				const VkImageMemoryBarrier2 swapchainAttachmentImageBarrier {
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
                };
				const VkDependencyInfo dependencyInfo {
					.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
					.imageMemoryBarrierCount = 1,
					.pImageMemoryBarriers = &swapchainAttachmentImageBarrier,
				};
				vkCmdPipelineBarrier2(cmd, &dependencyInfo);

				const VkRenderingAttachmentInfo swapchainAttachment = {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.imageView = viewer.swapchainImageViews[swapchainImageIndex],
					.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
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
				};
				vkCmdBeginRendering(cmd, &renderingInfo);

				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, viewer.meshPipeline);

				// Bind the camera descriptor set
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, viewer.meshPipelineLayout,
										0, 1, &viewer.cameraBuffers[currentFrame].cameraSet,
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

				vkCmdDrawMeshTasksEXT(cmd, 1, 1, 1);

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

            vkEndCommandBuffer(cmd);

            // Submit the command buffer
            const VkPipelineStageFlags submitWaitStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            VkSubmitInfo submitInfo {
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
            VkPresentInfoKHR presentInfo {
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
        }
    } catch (const vulkan_error& error) {
        std::cerr << error.what() << ": " << error.what_result() << '\n';
    } catch (const std::runtime_error& error) {
        std::cerr << error.what() << '\n';
    }

    vkDeviceWaitIdle(viewer.device); // Make sure everything is done

    viewer.taskScheduler.WaitforAll();

    // Destroys everything. We leave this out of the try-catch block to make sure it gets executed.
    // The swapchain is the only exception, as that gets recreated within the render loop. Managing it
    // with this paradigm is quite hard.
    viewer.swapchain.destroy_image_views(viewer.swapchainImageViews);
    vkb::destroy_swapchain(viewer.swapchain);
    viewer.flushObjects();
    glfwDestroyWindow(viewer.window);
    glfwTerminate();

    viewer.taskScheduler.WaitforAllAndShutdown();

    return 0;
}
