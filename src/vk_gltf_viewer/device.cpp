#include <tracy/Tracy.hpp>

#include <vulkan/vk.hpp>
#include <GLFW/glfw3.h>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/buffer.hpp>

#include <nvidia/dlss.hpp>

#if defined(VKV_NV_AFTERMATH)
#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#endif

VkBool32 vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT          messageSeverity,
							 VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
							 const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
							 void*                                            pUserData) {
	auto* destination = [&]() -> std::FILE* {
		switch (messageSeverity) {
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
				return stderr;
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
				return stdout;
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
			default:
				std::unreachable();
		}
	}();

	fmt::print(destination, "{}\n", pCallbackData->pMessage);
	fmt::print(destination, "\tObjects: {}\n", pCallbackData->objectCount);
	for (std::size_t i = 0; i < pCallbackData->objectCount; ++i) {
		auto& obj = pCallbackData->pObjects[i];
		fmt::print(destination, "\t\t[{}] 0x{:x}, {}, {}\n", i,
				   obj.objectHandle, obj.objectType, obj.pObjectName == nullptr ? "nullptr" : obj.pObjectName);
	}
	return VK_FALSE; // Beware: VK_TRUE here and the layers will kill the app instantly.
}

Instance::Instance() {
	ZoneScoped;
	vk::checkResult(volkInitialize(), "No compatible Vulkan loader or driver found");

	auto version = volkGetInstanceVersion();
	if (version < VK_API_VERSION_1_1) {
		throw std::runtime_error("The Vulkan loader only supports version 1.0.");
	}

	vk::initVulkanAllocationCallbacks();

	vkb::InstanceBuilder builder;

	// Enable GLFW extensions
	{
		std::uint32_t glfwExtensionCount = 0;
		const auto* glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		builder.enable_extensions(glfwExtensionCount, glfwExtensions);
	}

#if defined(VKV_NV_DLSS)
	// Enable NGX/DLSS extensions
	{
		dlss::initFeatureInfo();
		auto extensions = dlss::getRequiredInstanceExtensions();
		for (auto& ext : extensions)
			builder.enable_extension(ext.extensionName);
	}
#endif

	builder.enable_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	auto result = builder
		.set_app_name("vk_viewer")
		.require_api_version(1, 3, 0)
#if DEBUG
		.request_validation_layers()
		.set_debug_callback(vulkanDebugCallback)
#endif
		.set_allocation_callbacks(vk::allocationCallbacks.get())
		.build();
	if (!result)
		throw vulkan_error(result.error().message(), result.vk_result());

	instance = std::move(result).value();

	volkLoadInstanceOnly(instance);
}

Instance::~Instance() noexcept {
	ZoneScoped;
	vkb::destroy_instance(instance);
}

Device::Device(const Instance& instance, VkSurfaceKHR surface) {
	ZoneScoped;

#if defined(VKV_NV_AFTERMATH)
	aftermathCrashTracker = std::make_unique<AftermathCrashTracker>();
#endif

	const VkPhysicalDeviceFeatures vulkan10Features {
		.shaderInt64 = VK_TRUE,
	};
	const VkPhysicalDeviceVulkan11Features vulkan11Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.storageBuffer16BitAccess = VK_TRUE,
	};
	const VkPhysicalDeviceVulkan12Features vulkan12Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.storageBuffer8BitAccess = VK_TRUE,
		.uniformAndStorageBuffer8BitAccess = VK_TRUE,
		.shaderFloat16 = VK_TRUE,
		.shaderInt8 = VK_TRUE,
		.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
		.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
		.descriptorBindingPartiallyBound = VK_TRUE,
		.runtimeDescriptorArray = VK_TRUE,
		.samplerFilterMinmax = VK_TRUE,
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

	// Select a physical device with the given requirements
	vkb::PhysicalDeviceSelector selector(instance.instance);
	auto result = selector
		.set_surface(surface)
		.set_minimum_version(1, 3)
		.set_required_features(vulkan10Features)
		.set_required_features_11(vulkan11Features)
		.set_required_features_12(vulkan12Features)
		.set_required_features_13(vulkan13Features)
		.add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME)
		.add_required_extension_features(meshShaderFeatures)
#if defined(TRACY_ENABLE)
		.add_required_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
#endif
		.require_present()
		.require_dedicated_transfer_queue()
		.select();
	if (!result)
		throw vulkan_error(result.error().message(), result.vk_result());

	physicalDevice = std::move(result).value();

	VmaAllocatorCreateFlags allocatorFlags = 0;
	if (physicalDevice.enable_extension_if_present(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
		allocatorFlags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

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

#if defined(VKV_NV_DLSS)
	// Enable NGX/DLSS extensions
	{
		auto extensions = dlss::getRequiredDeviceExtensions(instance, physicalDevice);
		for (auto& ext : extensions)
			physicalDevice.enable_extension_if_present(ext.extensionName);
	}
#endif

#if defined(VKV_NV_AFTERMATH)
	VkDeviceDiagnosticsConfigCreateInfoNV diagnosticsConfig {
		.sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
		.flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV
			| VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV
			| VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV,
	};

	physicalDevice.enable_extension_if_present(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
#endif

	// Create the logical device
	vkb::DeviceBuilder builder(physicalDevice);
	auto buildResult = builder
		.custom_queue_setup(queues)
#if defined(VKV_NV_AFTERMATH)
		.add_pNext(&diagnosticsConfig)
#endif
		.set_allocation_callbacks(vk::allocationCallbacks.get())
		.build();
	if (!buildResult)
		throw vulkan_error(buildResult.error().message(), buildResult.vk_result());

	device = std::move(buildResult).value();

	// Load all function pointers
	volkLoadDevice(device);

	VkPhysicalDeviceProperties2 properties {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &vulkan12Properties,
	};
	vkGetPhysicalDeviceProperties2(physicalDevice, &properties);

#if defined(VKV_NV_DLSS)
	dlss::initSdk(instance, *this);
#endif

	resourceTable = std::make_unique<graphics::vulkan::VkResourceTable>(*this);

	// Create the VMA allocator
	const VmaVulkanFunctions vmaFunctions {
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
	};
	const VmaAllocatorCreateInfo allocatorInfo {
		.flags = allocatorFlags | VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = physicalDevice,
		.device = device,
		.pAllocationCallbacks = vk::allocationCallbacks.get(),
		.pVulkanFunctions = &vmaFunctions,
		.instance = instance,
		.vulkanApiVersion = VK_API_VERSION_1_3,
	};
	vk::checkResult(vmaCreateAllocator(&allocatorInfo, &allocator), "Failed to create VMA allocator");

	// Get the graphics queue handle
	auto graphicsQueueIndexResult = device.get_queue_index(vkb::QueueType::graphics);
	if (!graphicsQueueIndexResult)
		throw vulkan_error(graphicsQueueIndexResult.error().message(), graphicsQueueIndexResult.vk_result());
	graphicsQueueFamily = graphicsQueueIndexResult.value();

	vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue.handle);
	graphicsQueue.lock = std::make_unique<std::mutex>();
	vk::setDebugUtilsName(device, graphicsQueue.handle, "Graphics queue");

	// Get the transfer queue handle
	auto transferQueueIndexResult = device.get_dedicated_queue_index(vkb::QueueType::transfer);
	if (!transferQueueIndexResult)
		throw vulkan_error(transferQueueIndexResult.error().message(), transferQueueIndexResult.vk_result());
	transferQueueFamily = transferQueueIndexResult.value();

	transferQueues.resize(queueFamilies[transferQueueFamily].queueCount);
	for (std::uint32_t i = 0; auto& queue : transferQueues) {
		vkGetDeviceQueue(device, transferQueueFamily, i, &queue.handle);
		queue.lock = std::make_unique<std::mutex>();
		vk::setDebugUtilsName(device, queue.handle, fmt::format("Transfer queue {}", i));
		++i;
	}

	// Create the Tracy Vulkan context
#if defined(TRACY_ENABLE)
	tracyCtx = TracyVkContextHostCalibrated(physicalDevice, device,
											vkResetQueryPool,
											vkGetPhysicalDeviceCalibrateableTimeDomainsEXT,
											vkGetCalibratedTimestampsEXT);
#endif

	// Create the timeline deletion queue
	timelineDeletionQueue = std::make_unique<TimelineDeletionQueue>(*this);

	// Create the sync pools
	globalFencePool.init(device);

	// Create the command pools for the transfer queues
	uploadCommandPools.resize(taskScheduler.GetNumTaskThreads());
	for (auto& pool : uploadCommandPools)
		pool.create(device, transferQueueFamily);
}

Device::~Device() noexcept {
	ZoneScoped;
#if defined(VKV_NV_AFTERMATH)
	aftermathCrashTracker->waitToFinish();
#endif

#if defined(VKV_NV_DLSS)
	dlss::shutdown(device);
#endif
	for (auto& pool : uploadCommandPools)
		pool.destroy();
	globalFencePool.destroy();
	timelineDeletionQueue.reset(); // Needs to happen before device destruction
	resourceTable.reset();
#if defined(TRACY_ENABLE)
	if (tracyCtx != nullptr)
		DestroyVkContext(tracyCtx);
#endif
	vmaDestroyAllocator(allocator);
	vkb::destroy_device(device);
}

std::unique_ptr<ScopedBuffer> Device::createHostStagingBuffer(std::size_t byteSize) const noexcept {
	ZoneScoped;
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
	return std::make_unique<ScopedBuffer>(*this, &bufferCreateInfo, &allocationCreateInfo);
}

void Device::immediateSubmit(vk::Queue& queue, vk::CommandPool& commandPool, std::function<void(VkCommandBuffer)> commands) {
	ZoneScoped;
	auto cmd = commandPool.allocate();
	constexpr VkCommandBufferBeginInfo beginInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &beginInfo);

	commands(cmd);

	vkEndCommandBuffer(cmd);

	const VkCommandBufferSubmitInfo bufferSubmitInfo {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
		.commandBuffer = cmd,
	};
	const VkSubmitInfo2 submitInfo {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.commandBufferInfoCount = 1,
		.pCommandBufferInfos = &bufferSubmitInfo,
	};
	auto fence = globalFencePool.acquire();
	vk::checkResult(queue.submit(submitInfo, fence->handle), "Failed to submit immediate command buffer");

	globalFencePool.wait_and_free(fence);
}
