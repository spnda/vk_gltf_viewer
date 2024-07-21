#ifndef COMMON_GLSL_H
#define COMMON_GLSL_H

/** Header for common macros and constants for use in shared GLSL headers */

#if defined(__METAL_VERSION__)
#define SHADER_METAL 1
#include <metal_stdlib>

#define GLSL_NAMESPACE_BEGIN namespace glsl {
#define GLSL_NAMESPACE_END } // namespace glsl

namespace glsl {
	using namespace metal;

	// TODO: Do we want to keep all these aliases here?
	using vec2 = packed_float2;
	using vec3 = packed_float3;
	using vec4 = packed_float4;
	using bvec3 = bool3;
	using u8vec3 = packed_uchar3;
	using u8vec4 = packed_uchar4;
	using u16vec2 = packed_ushort2;
	using mat3 = float3x3;
	using mat4 = float4x4;
}

#define GLSL_CONSTANT static constant
#define GLSL_ARRAY(Type, Name, Count) metal::array<Type, Count> Name
#define BUFFER_REF(Name, Type) device Type*
#define GLSL_BOOL alignas(4) bool
#define MEMBER_INIT(Value) = Value
#define ALIGN_AS(Value)

#define PARAMETER_COPY(Name) Name
#define PARAMETER_REF(Name) device Name&
#define PARAMETER_CREF(Name) device const Name&

#elif defined(__cplusplus)
#define SHADER_CPP 1
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
#define BUFFER_REF(Name, Type) VkDeviceAddress
#define GLSL_BOOL alignas(4) bool
#define MEMBER_INIT(Value) = Value
#define ALIGN_AS(Value) alignas(Value)

#define PARAMETER_COPY(Name) Name
#define PARAMETER_REF(Name) Name&
#define PARAMETER_CREF(Name) const Name&

#else
#define SHADER_GLSL 1
#define GLSL_NAMESPACE_BEGIN
#define GLSL_NAMESPACE_END

#define GLSL_CONSTANT const
#define GLSL_ARRAY(Type, Name, Count) Type Name [Count]
#define BUFFER_REF(Name, Type) Name
#define GLSL_BOOL bool
#define MEMBER_INIT(Value)
#define ALIGN_AS(Value)

#define PARAMETER_COPY(Name) in Name
#define PARAMETER_REF(Name) inout Name
#define PARAMETER_CREF(Name) inout Name
#endif

#endif
