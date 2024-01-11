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
}
