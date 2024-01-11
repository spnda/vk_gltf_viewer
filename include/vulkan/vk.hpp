#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#if defined(WIN32)
// volk doesn't use vulkan.h and therefore doesn't include windows.h.
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#endif

#define VK_NO_PROTOTYPES
#include <volk.h>

// This is to keep any other files from including vulkan/vulkan.h directly, which could include Vulkan.h
// volk.h already does a correct setup of all Vulkan headers.
#define VULKAN_H_ 1

#include <vulkan/vk_fmt.hpp>

namespace vk {
    template <typename R, typename F, typename... Args>
    requires std::is_constructible_v<R> && requires(F func, Args... args, std::uint32_t count) {
        { func(args..., &count, nullptr) };
    }
    std::vector<R> enumerateVector(F func, Args... args) {
        std::uint32_t count = 0;
        func(args..., &count, nullptr);
        std::vector<R> ret(count);
        func(args..., &count, ret.data());
        return ret;
    }

    [[gnu::always_inline]] inline void checkResult(VkResult result, const char* message) noexcept(false) {
        if (result != VK_SUCCESS) {
            throw std::runtime_error(fmt::format(fmt::runtime(message), result));
        }
    }

    [[gnu::always_inline]] inline void checkResult(VkResult result, const std::string& message) noexcept(false) {
        if (result != VK_SUCCESS) {
            throw std::runtime_error(message);
        }
    }
} // namespace vk
