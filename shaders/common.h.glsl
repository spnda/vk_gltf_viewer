#ifndef COMMON_GLSL_H
#define COMMON_GLSL_H

/** Header for common macros and constants for use in shared GLSL headers */

#if defined(__cplusplus)
#define GLSL_NAMESPACE_BEGIN namespace glsl {
#define GLSL_NAMESPACE_END } // namespace glsl

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vk.hpp>
namespace glsl {
	using uint = std::uint32_t;
	using namespace std;
	using namespace glm;
}

#define GLSL_CONSTANT static constexpr
#define GLSL_ARRAY(Type, Name, Count) std::array<Type, Count> Name
#define BUFFER_REF(Name) VkDeviceAddress
#define GLSL_BOOL alignas(4) bool
#define MEMBER_INIT(Value) = Value

#define PARAMETER_COPY(Name) Name
#define PARAMETER_REF(Name) Name&
#define PARAMETER_CREF(Name) const Name&

#else
#define GLSL_NAMESPACE_BEGIN
#define GLSL_NAMESPACE_END

#define GLSL_CONSTANT const
#define GLSL_ARRAY(Type, Name, Count) Type Name [Count]
#define BUFFER_REF(Name) Name
#define GLSL_BOOL bool
#define MEMBER_INIT(Value)

#define PARAMETER_COPY(Name) in Name
#define PARAMETER_REF(Name) inout Name
#define PARAMETER_CREF(Name) inout Name
#endif

#endif
