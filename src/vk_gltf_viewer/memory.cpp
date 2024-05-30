#include <string_view>

static constexpr std::string_view cpuMemoryPool = "cpu malloc\0";

#include <tracy/Tracy.hpp>
#if defined(TRACY_ENABLE)
void* operator new(std::size_t count)
{
	auto ptr = malloc(count);
	TracyAllocN(ptr, count, cpuMemoryPool.data());
	return ptr;
}
void operator delete(void* ptr) noexcept
{
	TracyFreeN(ptr, cpuMemoryPool.data());
	free(ptr);
}
#endif

#include <vulkan/vk.hpp>

void* vkAlloc(void* userData, std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) noexcept {
#ifdef _WIN32
	auto* ptr = _aligned_malloc(size, alignment);
#else
	auto* ptr = std::aligned_alloc(alignment, size);
#endif
#ifdef TRACY_ENABLE
	TracyAllocN(ptr, size, cpuMemoryPool.data());
#endif
	return ptr;
}

void* vkRealloc(void* userData, void* original, std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) noexcept {
	return _aligned_realloc(original, size, alignment);
}

void vkFree(void* userData, void* alloc) noexcept {
#ifdef TRACY_ENABLE
	TracyFreeN(alloc, cpuMemoryPool.data());
#endif
#ifdef _WIN32
	_aligned_free(alloc);
#else
	std::free(alloc);
#endif
}

void vk::initVulkanAllocationCallbacks() {
	vk::allocationCallbacks = VkAllocationCallbacks {
		.pUserData = nullptr,
		.pfnAllocation = &vkAlloc,
		.pfnReallocation = &vkRealloc,
		.pfnFree = &vkFree,
	};
}
