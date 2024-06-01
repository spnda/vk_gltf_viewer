#include <string_view>

#include <tracy/Tracy.hpp>
#if defined(TRACY_ENABLE)
static constexpr std::string_view cpuMemoryPool = "cpu malloc\0";

void* operator new(std::size_t count)
{
	auto ptr = ::malloc(count);
	TracyAllocN(ptr, count, cpuMemoryPool.data());
	return ptr;
}
void operator delete(void* ptr) noexcept
{
	TracyFreeN(ptr, cpuMemoryPool.data());
	::free(ptr);
}
#endif

#include <vulkan/vk.hpp>
#include <fastgltf/util.hpp>

#if defined(_WIN32) || defined(__APPLE__)
void* vkAlloc(void* userData, std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) noexcept {
#if defined(_WIN32)
	auto* ptr = _aligned_malloc(size, alignment);
#else
	auto* ptr = std::aligned_alloc(alignment, size);
#endif

#ifdef TRACY_ENABLE
	TracyAllocN(ptr, size, cpuMemoryPool.data());
#endif
	return ptr;
}

void vkFree(void* userData, void* alloc) noexcept {
#ifdef TRACY_ENABLE
	TracyFreeN(alloc, cpuMemoryPool.data());
#endif

#if defined(_WIN32)
	_aligned_free(alloc);
#else
	std::free(alloc);
#endif
}

#if defined(__APPLE__)
#include <malloc/malloc.h>
#endif

// Tracy does not support realloc, and therefore we'll emulate this function
// using vkAlloc and vkFree.
void* vkRealloc(void* userData, void* original, std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) noexcept {
	if (original == nullptr) {
		// If pOriginal is NULL, then pfnReallocation must behave equivalently to a call to PFN_vkAllocationFunction with the same parameter values (without pOriginal).
		return vkAlloc(userData, size, alignment, scope);;
	}

	if (size == 0) {
		// If size is zero, then pfnReallocation must behave equivalently to a call to PFN_vkFreeFunction with the same pUserData parameter value,
		// and pMemory equal to pOriginal.
		vkFree(userData, original);
		return nullptr;
	}

	auto* ptr = vkAlloc(userData, size, alignment, scope);
	if (ptr == nullptr)
		// If this function fails and pOriginal is non-NULL the application must not free the old allocation.
		return ptr;

#if defined(_WIN32)
	auto originalSize = _aligned_msize(original, alignment, 0);
#elif defined(__APPLE__)
	auto originalSize = malloc_size(original);
#endif

	std::memcpy(ptr, original, fastgltf::min(originalSize, size) - 1);
	vkFree(userData, original);
	return ptr;
}
#endif // defined(_WIN32) || defined(__APPLE__)

void vk::initVulkanAllocationCallbacks() {
#if defined(_WIN32) || defined(__APPLE__)
	vk::allocationCallbacks = std::make_unique<VkAllocationCallbacks>(VkAllocationCallbacks {
		.pUserData = nullptr,
		.pfnAllocation = &vkAlloc,
		.pfnReallocation = &vkRealloc,
		.pfnFree = &vkFree,
	});
#endif
}
