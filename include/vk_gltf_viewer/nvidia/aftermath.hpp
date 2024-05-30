#pragma once

#include <mutex>
#include <unordered_map>
#include <memory>

#include <GFSDK_Aftermath_Defines.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

namespace std {
	template <>
	struct hash<GFSDK_Aftermath_ShaderDebugInfoIdentifier> {
		std::size_t operator()(const GFSDK_Aftermath_ShaderDebugInfoIdentifier& identifier) const {
			static constexpr auto h = hash<std::uint64_t>{};
			return h(identifier.id[0]) ^ h(identifier.id[1]);
		}
	};
}

inline bool operator==(const GFSDK_Aftermath_ShaderDebugInfoIdentifier& lhs, const GFSDK_Aftermath_ShaderDebugInfoIdentifier& rhs) {
	return lhs.id[0] == rhs.id[0] && lhs.id[1] == rhs.id[1];
}

// TODO: Add support for custom pipeline markers
struct AftermathCrashTracker {
	std::mutex mutex;

	std::unordered_map<GFSDK_Aftermath_ShaderDebugInfoIdentifier, std::vector<std::byte>> shaderDebugInfo;

	static void checkResult(GFSDK_Aftermath_Result result);

	explicit AftermathCrashTracker();
	~AftermathCrashTracker();

	// When a DEVICE_LOST occurs, the crash handler needs some time to collect data.
	// We therefore need to wait before destroying the logical device.
	void waitToFinish();
};
