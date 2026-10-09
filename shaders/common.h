#ifndef SHADERS_COMMON_H
#define SHADERS_COMMON_H

/** Header for common macros and constants for use in shared shader headers */

#if defined(__METAL_VERSION__)
#define SHADER_METAL 1
#define SHADER_NAMESPACE_BEGIN namespace shaders {
#define SHADER_NAMESPACE_END } // namespace shaders

#include <metal_stdlib>

namespace shaders {
	namespace mtl = metal;
	using namespace mtl;

	using fvec2 = mtl::float2;
	using fvec3 = mtl::float3;
	using fvec4 = mtl::float4;
	using packed_fvec2 = mtl::packed_float2;
	using packed_fvec3 = mtl::packed_float3;
	using packed_fvec4 = mtl::packed_float4;

	using bvec2 = mtl::bool2;
	using bvec3 = mtl::bool3;
	using bvec4 = mtl::bool4;

	using u8vec2 = mtl::uchar2;
	using u8vec3 = mtl::uchar3;
	using u8vec4 = mtl::uchar4;
	using packed_u8vec2 = mtl::packed_uchar2;
	using packed_u8vec3 = mtl::packed_uchar3;
	using packed_u8vec4 = mtl::packed_uchar4;

	using u16vec2 = mtl::ushort2;
	using u16vec3 = mtl::ushort3;
	using u16vec4 = mtl::ushort4;
	using packed_u16vec2 = mtl::packed_ushort2;
	using packed_u16vec3 = mtl::packed_ushort3;
	using packed_u16vec4 = mtl::packed_ushort4;

	using u32vec2 = mtl::uint2;
	using u32vec3 = mtl::uint3;
	using u32vec4 = mtl::uint4;
	using packed_u32vec2 = mtl::packed_uint2;
	using packed_u32vec3 = mtl::packed_uint3;
	using packed_u32vec4 = mtl::packed_uint4;

	using f16vec2 = mtl::half2;
	using f16vec3 = mtl::half3;
	using f16vec4 = mtl::half4;
	using packed_f16vec2 = mtl::packed_half2;
	using packed_f16vec3 = mtl::packed_half3;
	using packed_f16vec4 = mtl::packed_half4;

	using fmat2 = mtl::float2x2;
	using fmat3 = mtl::float3x3;
	using fmat4 = mtl::float4x4;
}

#define SHADER_CONSTANT static constant
#define SHADER_ARRAY(Type, Name, Count) mtl::array<Type, Count> Name
#define BUFFER_REF(Name, Type) device Type*
#define SHADER_BOOL alignas(4) bool
#define MEMBER_INIT(Value) = Value
#define ALIGN_AS(Value) alignas(Value)
#define FUNCTION_INLINE inline
#define FUNCTION_CONSTEXPR constexpr

#define PARAMETER_COPY(Name) Name
#define PARAMETER_REF(Name) thread Name&
#define PARAMETER_CREF(Name) thread const Name&

#elif defined(__cplusplus)
#define SHADER_CPP 1
#define SHADER_NAMESPACE_BEGIN namespace shaders {
#define SHADER_NAMESPACE_END } // namespace shaders

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>
#include <vulkan/vk.hpp>
namespace shaders {
	using int8_t = std::int8_t;
	using uint8_t = std::uint8_t;
	using int16_t = std::int16_t;
	using uint16_t = std::uint16_t;
	using int32_t = std::int32_t;
	using uint32_t = std::uint32_t;
	using int64_t = std::int64_t;
	using uint64_t = std::uint64_t;

	using namespace glm;
	using packed_fvec2 = fvec2;
	using packed_fvec3 = fvec3;
	using packed_fvec4 = fvec4;

	using packed_u8vec2 = u8vec2;
	using packed_u8vec3 = u8vec3;
	using packed_u8vec4 = u8vec4;

	using packed_u16vec2 = u16vec2;
	using packed_u16vec3 = u16vec3;
	using packed_u16vec4 = u16vec4;

	using packed_u32vec2 = u32vec2;
	using packed_u32vec3 = u32vec3;
	using packed_u32vec4 = u32vec4;
}

#define SHADER_CONSTANT static constexpr const
#define SHADER_ARRAY(Type, Name, Count) std::array<Type, Count> Name
#define BUFFER_REF(Name, Type) VkDeviceAddress
#define SHADER_BOOL alignas(4) bool
#define MEMBER_INIT(Value) = Value
#define ALIGN_AS(Value) alignas(Value)
#define FUNCTION_INLINE inline
#define FUNCTION_CONSTEXPR constexpr

#define PARAMETER_COPY(Name) Name
#define PARAMETER_REF(Name) Name&
#define PARAMETER_CREF(Name) const Name&

#else
#define SHADER_GLSL 1
#define SHADER_NAMESPACE_BEGIN
#define SHADER_NAMESPACE_END

// GLSL is such a garbage language
#define uint32_t uint

#define fvec2 vec2
#define fvec3 vec3
#define fvec4 vec4

#define packed_fvec2 vec2
#define packed_fvec3 vec3
#define packed_fvec4 vec4

#define packed_u8vec2 u8vec2
#define packed_u8vec3 u8vec3
#define packed_u8vec4 u8vec4

#define packed_u16vec2 u16vec2
#define packed_u16vec3 u16vec3
#define packed_u16vec4 u16vec4

#define packed_f16vec2 f16vec2
#define packed_f16vec3 f16vec3
#define packed_f16vec4 f16vec4

#define fmat2 mat2
#define fmat3 mat3
#define fmat4 mat4

#define SHADER_CONSTANT const
#define SHADER_ARRAY(Type, Name, Count) Type Name [Count]
#define BUFFER_REF(Name, Type) Name
#define SHADER_BOOL bool
#define MEMBER_INIT(Value)
#define ALIGN_AS(Value)
#define FUNCTION_INLINE
#define FUNCTION_CONSTEXPR

#define PARAMETER_COPY(Name) in Name
#define PARAMETER_REF(Name) inout Name
#define PARAMETER_CREF(Name) inout Name

#define saturate(Value) clamp(Value, 0.f, 1.f)
#endif

// If 1 use octahedral encoding, if 0 just store the normal
// This only really exists for testing purposes but could be useful to determine a difference
#define GBUFFER_NORMAL_ENCODING 1
#define VERTEX_BUFFER_NORMAL_ENCODING 1
#define VERTEX_BUFFER_TANGENT_ENCODING 1

#endif
