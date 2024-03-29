#pragma once

// This header should be included everywhere you use Vulkan structs inside a fmtlib-formatted
// string. This will format the VkResult parameter to the corresponding enum name.

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <vulkan/vk.hpp>
#include <vulkan/vk_enum_string_helper.h>

namespace vk {
    // Represents a Vulkan version but wraps it in another type so that we can format it.
    struct Version {
        std::uint32_t version;

        Version(std::uint32_t value) : version(value) {}

        inline operator std::uint32_t() const {
            return version;
        }
    };
} // namespace vk

template <>
struct fmt::formatter<vk::Version> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    inline auto format(vk::Version const& result, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}.{}.{}", (result.version >> 22) & 0x7FU, (result.version >> 12) & 0x3FFU,
                              result.version & 0xFFFU);
    }
};

template <>
struct fmt::formatter<VkResult> : formatter<std::string_view> {
    template <typename FormatContext>
    inline auto format(VkResult const& result, FormatContext& ctx) const {
		return formatter<string_view>::format(string_VkResult(result), ctx);
    }
};

template <>
struct fmt::formatter<VkExtensionProperties> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    inline auto format(VkExtensionProperties const& result, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}:v{}", result.extensionName, result.specVersion);
    }
};

template <>
struct fmt::formatter<VkLayerProperties> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    inline auto format(VkLayerProperties const& result, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}:v{}", result.layerName, result.implementationVersion);
    }
};

template <>
struct fmt::formatter<VkObjectType> : formatter<std::string_view> {
    template <typename FormatContext>
    inline auto format(VkObjectType const& result, FormatContext& ctx) const {
		return formatter<string_view>::format(string_VkObjectType(result), ctx);
    }
};
