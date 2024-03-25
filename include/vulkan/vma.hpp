#pragma once

// Simple header for including VMA. We include volk.h beforehand to avoid the VMA to
// include vulkan.h and windows.h.
#define VK_NO_PROTOTYPES
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_VULKAN_VERSION 1003000

// VMA includes vulkan.h directly, which might contain Windows.h
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

// Set to 1 if VMA debug output is necessary
#if 0
#include <fmt/printf.h>
#define VMA_DEBUG_LOG_FORMAT(format, ...) do { \
	fmt::printf((format), __VA_ARGS__); \
	fmt::printf("\n"); \
} while(false)
#endif

// clang-format off
#include <volk.h>
#include <vk_mem_alloc.h>
// clang-format on

namespace vk {
	template<typename T = void>
	class ScopedMap {
		VmaAllocator allocator;
		VmaAllocation allocation;
		void* data;

	public:
		ScopedMap(VmaAllocator allocator, VmaAllocation allocation) : allocator(allocator), allocation(allocation),
																	  data(nullptr) {
			vmaMapMemory(allocator, allocation, &data);
		}

		T *get() {
			return static_cast<T*>(data);
		}

		~ScopedMap() {
			vmaUnmapMemory(allocator, allocation);
		}
	};

	[[gnu::always_inline]] inline void setAllocationName(VmaAllocator allocator, VmaAllocation allocation, std::string string) {
		vmaSetAllocationName(allocator, allocation, string.c_str());
	}
}
