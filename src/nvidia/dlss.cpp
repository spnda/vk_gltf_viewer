#if defined(VKV_NV_DLSS)
#include <fmt/xchar.h>

#include <filesystem>

#include <tracy/Tracy.hpp>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/scheduler.hpp>
#include <nvidia/dlss.hpp>

#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers.h>

namespace dlss {
	void checkNgxResult(NVSDK_NGX_Result result, const std::string& message) {
		if (NVSDK_NGX_SUCCEED(result))
			return;
		throw std::runtime_error(message);
	}

	void logCallback(const char* message, NVSDK_NGX_Logging_Level level, NVSDK_NGX_Feature sourceFeature) {
		ZoneScoped;
		fmt::print("{}", message);
	}

	static constexpr std::string_view projectId = "5a0dbedb-9ed7-4a09-ada5-7ab0feb8ff82\0";
	NVSDK_NGX_FeatureCommonInfo featureCommonInfo {
		.LoggingInfo = {
			.LoggingCallback = logCallback,
			.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF, // NVSDK_NGX_LOGGING_LEVEL_ON,
		}
	};
	NVSDK_NGX_FeatureDiscoveryInfo featureDiscoveryInfo {
		.SDKVersion = NVSDK_NGX_Version_API,
		.FeatureID = NVSDK_NGX_Feature_SuperSampling,
	};

	bool dlssSupported = false;

	bool isSupported() noexcept { return dlssSupported; }
}

void dlss::initFeatureInfo() noexcept {
	ZoneScoped;
	featureDiscoveryInfo.Identifier = {
		.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id,
		.v = {
			.ProjectDesc = {
				.ProjectId = projectId.data(),
				.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM,
				.EngineVersion = "1.0"
			}
		}
	};

	static auto dir = std::filesystem::current_path();
	featureDiscoveryInfo.ApplicationDataPath = dir.c_str();

	featureDiscoveryInfo.FeatureInfo = &featureCommonInfo;
}

std::span<VkExtensionProperties> dlss::getRequiredInstanceExtensions() {
	ZoneScoped;
	std::uint32_t extensionCount = 0;
	VkExtensionProperties* properties = nullptr;
	auto result = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
		&featureDiscoveryInfo, &extensionCount, &properties);
	checkNgxResult(result, "Failed to get required DLSS instance extensions");
	return {properties, extensionCount};
}

std::span<VkExtensionProperties> dlss::getRequiredDeviceExtensions(VkInstance instance, VkPhysicalDevice device) {
	ZoneScoped;
	std::uint32_t extensionCount = 0;
	VkExtensionProperties* properties = nullptr;
	auto result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
		instance, device, &featureDiscoveryInfo, &extensionCount, &properties);
	checkNgxResult(result, "Failed to get required DLSS physical device extensions");
	return {properties, extensionCount};
}

void dlss::initSdk(VkInstance instance, Device& device) {
	ZoneScoped;
	checkNgxResult(NVSDK_NGX_VULKAN_Init_with_ProjectID(
		projectId.data(),
		NVSDK_NGX_ENGINE_TYPE_CUSTOM,
		"1.0",
		featureDiscoveryInfo.ApplicationDataPath,
		instance, device.physicalDevice, device,
		vkGetInstanceProcAddr, vkGetDeviceProcAddr,
		&featureCommonInfo), "Failed to init NGX SDK");

	{
		checkNgxResult(NVSDK_NGX_VULKAN_GetCapabilityParameters(&ngxParams), "Failed to query capability parameters");

		{
			int support = 0;
			auto res = ngxParams->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &support);

			// This means the SDK and hardware support DLSS
			dlssSupported = NVSDK_NGX_SUCCEED(res) && static_cast<bool>(support);

			if (dlssSupported) {
				res = ngxParams->Get(NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &support);

				// This means DLSS was disabled or denied.
				dlssSupported = NVSDK_NGX_SUCCEED(res) && static_cast<bool>(support);
			}
		}
	}
}

NVSDK_NGX_Handle* dlss::initFeature(Device& device, glm::u32vec2 inputSize, glm::u32vec2 outputSize) {
	ZoneScoped;
	if (!dlssSupported) {
		return nullptr;
	}

	// Create the NGX feature handle. The DLSS docs do not mention if a transfer queue is enough, but it seems to work for me.
	NVSDK_NGX_Handle* ret = nullptr;
	device.immediateSubmit(device.getNextTransferQueueHandle(), device.uploadCommandPools[taskScheduler.GetThreadNum()], [&](auto cmd) {
		NVSDK_NGX_DLSS_Create_Params createParams {
			.Feature = {
				.InWidth = inputSize.x,
				.InHeight = inputSize.y,
				.InTargetWidth = outputSize.x,
				.InTargetHeight = outputSize.y,
			},
			.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure,
			.InEnableOutputSubrects = false,
		};

		auto res = NGX_VULKAN_CREATE_DLSS_EXT1(device, cmd, 1, 1, &ret, ngxParams, &createParams);
		if (NVSDK_NGX_FAILED(res))
			fmt::print("DLSS Creation failed: {}", res);
	});
	return ret;
}

dlss::DlssRecommendedSettings dlss::getRecommendedSettings(NVSDK_NGX_PerfQuality_Value quality, glm::u32vec2 windowSize) {
	ZoneScoped;
	dlss::DlssRecommendedSettings settings;
	auto res = NGX_DLSS_GET_OPTIMAL_SETTINGS(ngxParams,
		windowSize.x, windowSize.y, quality,
		&settings.optimalRenderSize.x, &settings.optimalRenderSize.y,
		&settings.maxRenderSize.x, &settings.maxRenderSize.y,
		&settings.minRenderSize.x, &settings.minRenderSize.y,
		&settings.sharpness);

	if (NVSDK_NGX_FAILED(res)) {
		fmt::print("Querying Optimal Settings failed! code = {}, info: {}", std::to_underlying(res), res);
		return dlss::DlssRecommendedSettings {
			.sharpness = 0.f,
			.optimalRenderSize = windowSize,
			.maxRenderSize = windowSize,
			.minRenderSize = windowSize,
		};
	}

	assert(settings.optimalRenderSize.x != 0 && settings.optimalRenderSize.y != 0);
	return settings;
}

void dlss::releaseFeature(NVSDK_NGX_Handle* handle) {
	ZoneScoped;
	NVSDK_NGX_VULKAN_ReleaseFeature(handle);
}

void dlss::shutdown(VkDevice device) {
	ZoneScoped;
	NVSDK_NGX_VULKAN_DestroyParameters(ngxParams);
	NVSDK_NGX_VULKAN_Shutdown1(device);
}
#endif // defined(VKV_NV_DLSS)
