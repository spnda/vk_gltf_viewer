#pragma once
// This header is defined to be used in CMake while configuring the imgui target.
// If this header is moved, or renamed that CMake script will have to be updated.
// This also boldly assumes that nothing in this project includes imgui in a C file.

#include <cstdint>

#include <glm/glm.hpp>

// This needs to stay the same size as shaders::ResourceTableHandle
#define ImTextureID std::uint32_t

#define IMGUI_INCLUDE_IMGUI_USER_H
#define IMGUI_USER_H_FILENAME "imgui/user.hpp"

#define IM_VEC2_CLASS_EXTRA                                                       \
        constexpr ImVec2(const glm::fvec2& v) : x(v.x), y(v.y) {}                 \
        operator glm::fvec2() const { return glm::fvec2(x, y); }

#define IM_VEC4_CLASS_EXTRA                                                       \
        constexpr ImVec4(const glm::fvec4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {} \
        operator glm::fvec4() const { return glm::fvec4(x, y, z, w); }

#define IMGUI_DEFINE_MATH_OPERATORS
