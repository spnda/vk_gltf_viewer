#if defined(VKV_NV_AFTERMATH)
#include <fmt/format.h>

#include <fstream>
#include <functional>
#include <thread>

#include <tracy/Tracy.hpp>

#include <vulkan/vk.hpp>
#include <nvidia/aftermath.hpp>

#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

void aftermathCrashDumpCallback(const void* pGpuCrashDump, std::uint32_t dumpSize, void* pUserData) {
	ZoneScoped;
	auto& tracker = *static_cast<AftermathCrashTracker*>(pUserData);
	std::lock_guard<std::mutex> lock(tracker.mutex);

	GFSDK_Aftermath_GpuCrashDump_Decoder decoder = {};
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GpuCrashDump_CreateDecoder(
		GFSDK_Aftermath_Version_API,
		pGpuCrashDump,
		dumpSize,
		&decoder));

	GFSDK_Aftermath_GpuCrashDump_BaseInfo baseInfo = {};
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GpuCrashDump_GetBaseInfo(decoder, &baseInfo));

	// Get the application name description
	std::uint32_t applicationNameLength = 0;
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GpuCrashDump_GetDescriptionSize(
		decoder,
		GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName,
		&applicationNameLength));

	std::vector<char> applicationName(applicationNameLength, '\0');
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GpuCrashDump_GetDescription(
		decoder,
		GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName,
		static_cast<std::uint32_t>(applicationName.size()),
		applicationName.data()));

	static int count = 0; // Is this bug fixed yet?
	const std::string filePath = fmt::format("{}-{}-{}.nv-gpudmp", applicationName.data(), baseInfo.pid, count++);
	std::ofstream dumpFile(filePath, std::ios::binary);
	if (dumpFile) {
		dumpFile.write(static_cast<const char*>(pGpuCrashDump), dumpSize);
		dumpFile.close();
	}

	// Also generate a JSON dump
	uint32_t jsonSize = 0;
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
		decoder,
		GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO,
		GFSDK_Aftermath_GpuCrashDumpFormatterFlags_UTF8_OUTPUT,
		nullptr,
		nullptr,
		nullptr,
		pUserData,
		&jsonSize));

	std::vector<char> json(jsonSize);
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GpuCrashDump_GetJSON(
		decoder,
		uint32_t(json.size()),
		json.data()));

	const std::string jsonFileName = filePath + ".json";
	std::ofstream jsonFile(jsonFileName, std::ios::binary);
	if (jsonFile) {
		jsonFile.write(json.data(), static_cast<std::streamsize>(json.size()));
		jsonFile.close();
	}

	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder));
}

void aftermathShaderDebugInfoCallback(const void* pShaderDebugInfo, uint32_t shaderDebugInfoSize, void* pUserData) {
	ZoneScoped;
	auto& tracker = *static_cast<AftermathCrashTracker*>(pUserData);
	std::lock_guard<std::mutex> lock(tracker.mutex);

	// Get shader debug information identifier.
	GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier = {};
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GetShaderDebugInfoIdentifier(
		GFSDK_Aftermath_Version_API, pShaderDebugInfo, shaderDebugInfoSize, &identifier));

	std::vector<std::byte> data(static_cast<const std::byte*>(pShaderDebugInfo),
								static_cast<const std::byte*>(pShaderDebugInfo) + shaderDebugInfoSize);
	tracker.shaderDebugInfo[identifier] = std::move(data);

	// Finally write the data into a nvdbg file
	std::string filePath = fmt::format("shader-{:x}-{:x}.nvdbg", identifier.id[0], identifier.id[1]);
	std::ofstream file(filePath, std::ios::binary);
	if (file)
		file.write(static_cast<const char*>(pShaderDebugInfo), shaderDebugInfoSize);
}

void aftermathCrashDumpDescriptionCallback(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void* pUserData) {
	ZoneScoped;
	auto& tracker = *static_cast<AftermathCrashTracker*>(pUserData);
	std::lock_guard<std::mutex> lock(tracker.mutex);

	addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "vk_gltf_viewer");
	addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, "v1.0");
}

void aftermathResolveMarkerCallback(const void* pMarkerData, const uint32_t markerDataSize, void* pUserData, void** ppResolvedMarkerData, uint32_t* pResolvedMarkerDataSize) {
	ZoneScoped;
	auto& tracker = *static_cast<AftermathCrashTracker*>(pUserData);
	std::lock_guard<std::mutex> lock(tracker.mutex);
}

template <>
struct fmt::formatter<GFSDK_Aftermath_Result> {
	template <typename ParseContext>
	constexpr auto parse(ParseContext& ctx) {
		return ctx.begin();
	}

	template <typename FormatContext>
	inline auto format(GFSDK_Aftermath_Result const& result, FormatContext& ctx) const {
		switch (result) {
			case GFSDK_Aftermath_Result_FAIL_DriverVersionNotSupported:
				return fmt::format_to(ctx.out(), "Unsupported driver version - requires an NVIDIA R495 display driver or newer.");
			case GFSDK_Aftermath_Result_FAIL_D3dDllInterceptionNotSupported:
				return fmt::format_to(ctx.out(), "Aftermath is incompatible with D3D API interception, such as PIX or Nsight Graphics.");
			default:
				return fmt::format_to(ctx.out(), "Aftermath Error 0x{:x}", std::to_underlying(result));
		}
	}
};

void AftermathCrashTracker::checkResult(GFSDK_Aftermath_Result result) {
	if (GFSDK_Aftermath_SUCCEED(result))
		return;

	throw std::runtime_error(fmt::format("{}", result));
}

AftermathCrashTracker::AftermathCrashTracker() {
	ZoneScoped;
	// Enable GPU crash dumps for any devices creates after this call
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_EnableGpuCrashDumps(
		GFSDK_Aftermath_Version_API,
		GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
		GFSDK_Aftermath_GpuCrashDumpFeatureFlags_Default,
		aftermathCrashDumpCallback,
		aftermathShaderDebugInfoCallback,
		aftermathCrashDumpDescriptionCallback,
		aftermathResolveMarkerCallback,
		this));
}

AftermathCrashTracker::~AftermathCrashTracker() {
	ZoneScoped;
	GFSDK_Aftermath_DisableGpuCrashDumps();
}

void AftermathCrashTracker::waitToFinish() {
	ZoneScoped;
	GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
	AftermathCrashTracker::checkResult(GFSDK_Aftermath_GetCrashDumpStatus(&status));

	if (status == GFSDK_Aftermath_CrashDump_Status_NotStarted)
		return;

	while (status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed && status != GFSDK_Aftermath_CrashDump_Status_Finished) {
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		AftermathCrashTracker::checkResult(GFSDK_Aftermath_GetCrashDumpStatus(&status));
	}
}
#endif // defined(VKV_NV_AFTERMATH)
