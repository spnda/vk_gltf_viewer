#pragma once

#include <tracy/Tracy.hpp>

#include <vulkan/vk.hpp>

#define VK_OBJECT_TYPE_CASE(x, y)
#include <vulkan/object_types.hpp>

namespace vk {
	template <typename T>
	[[gnu::always_inline]] inline void setDebugUtilsName(VkDevice device, T handle, std::string string) {
		ZoneScoped;
		if (vkSetDebugUtilsObjectNameEXT == nullptr || handle == VK_NULL_HANDLE) {
			return;
		}

		const auto objectType = VulkanObjectTypeV<T>;
		const VkDebugUtilsObjectNameInfoEXT info = {
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
			.objectType = objectType,
			.objectHandle = reinterpret_cast<std::uint64_t>(handle),
			.pObjectName = string.c_str(),
		};
		vkSetDebugUtilsObjectNameEXT(device, &info);
	}
} // namespace vk
