#pragma once

#include <tracy/Tracy.hpp>

#include <vulkan/vk.hpp>

namespace vk {
	template <typename T>
	[[gnu::always_inline]] inline void setDebugUtilsName(VkDevice device, T handle, std::string string) {
		ZoneScoped;
		if (vkSetDebugUtilsObjectNameEXT == nullptr || handle == VK_NULL_HANDLE) {
			return;
		}

		// We use the other header with macros to construct a long chain of constexpr conditions
		// to determine the VkObjectType for the given T in the template.
#define VK_OBJECT_TYPE_CASE_FIRST(id, newObjectType)                                                                                       \
		if constexpr (std::is_same_v<T, id>) {                                                                                                 \
			objectType = newObjectType;                                                                                                        \
		}
#define VK_OBJECT_TYPE_CASE(id, newObjectType)                                                                                             \
		else if constexpr (std::is_same_v<T, id>) {                                                                                            \
			objectType = newObjectType;                                                                                                        \
		}

		VkObjectType objectType = VK_OBJECT_TYPE_UNKNOWN;
#include <vulkan/object_types.hpp>

		const VkDebugUtilsObjectNameInfoEXT info = {
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
			.objectType = objectType,
			.objectHandle = reinterpret_cast<std::uint64_t>(handle),
			.pObjectName = string.c_str(),
		};
		vkSetDebugUtilsObjectNameEXT(device, &info);
	}
} // namespace vk
