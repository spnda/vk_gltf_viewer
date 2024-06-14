#if defined(VKV_NV_DLSS)
#pragma once

#include <span>

#include <glm/vec2.hpp>

#include <fmt/format.h>
#include <fmt/xchar.h>

#include <vulkan/vk.hpp>
#include <nvsdk_ngx_defs.h>

struct Device;
struct NVSDK_NGX_Parameter;

extern "C" {
const wchar_t* GetNGXResultAsString(NVSDK_NGX_Result res);
}

template <>
struct fmt::formatter<NVSDK_NGX_Result> {
	template <typename ParseContext>
	constexpr auto parse(ParseContext& ctx) {
		return ctx.begin();
	}

	template <typename FormatContext>
	inline auto format(NVSDK_NGX_Result const& result, FormatContext& ctx) const {
		return fmt::format_to(ctx.out(), GetNGXResultAsString(result));
	}
};

namespace dlss {
	struct DlssRecommendedSettings {
		float sharpness = 0.01f; // in ngx sdk 3.1, dlss sharpening is deprecated
		glm::u32vec2 optimalRenderSize = {~(0u), ~(0u)};
		glm::u32vec2 maxRenderSize     = {~(0u), ~(0u)};
		glm::u32vec2 minRenderSize     = {~(0u), ~(0u)};
	};

	constexpr auto modes = std::to_array<std::pair<NVSDK_NGX_PerfQuality_Value, std::string_view>>({
		//{ NVSDK_NGX_PerfQuality_Value_UltraQuality, "Ultra-Quality" },
		{ NVSDK_NGX_PerfQuality_Value_MaxQuality, "Quality\0" },
		{ NVSDK_NGX_PerfQuality_Value_Balanced, "Balanced\0" },
		{ NVSDK_NGX_PerfQuality_Value_MaxPerf, "Performance\0" },
		{ NVSDK_NGX_PerfQuality_Value_UltraPerformance, "Ultra-Performance\0" },
		{ NVSDK_NGX_PerfQuality_Value_DLAA, "DLAA\0" },
	});

	inline NVSDK_NGX_Parameter* ngxParams;

	void initFeatureInfo() noexcept;
	std::span<VkExtensionProperties> getRequiredInstanceExtensions();
	std::span<VkExtensionProperties> getRequiredDeviceExtensions(VkInstance instance, VkPhysicalDevice device);
	void initSdk(VkInstance instance, Device& device);
	NVSDK_NGX_Handle* initFeature(Device& device, glm::u32vec2 inputSize, glm::u32vec2 outputSize);
	DlssRecommendedSettings getRecommendedSettings(NVSDK_NGX_PerfQuality_Value quality, glm::u32vec2 windowSize);
	void releaseFeature(NVSDK_NGX_Handle* handle);
	void shutdown(VkDevice device);

	bool isSupported() noexcept;
}
#endif // defined(VKV_NV_DLSS)
