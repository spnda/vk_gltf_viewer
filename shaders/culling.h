#ifndef SHADERS_CULLING_H
#define SHADERS_CULLING_H

#include "common.h"
SHADER_NAMESPACE_BEGIN

// Frustum culling using 6 planes on an AABB
#if defined(SHADER_METAL)
FUNCTION_INLINE bool isAabbInFrustum(metal::float3 center, metal::float3 extents, device const metal::array<metal::packed_float4, 6>& frustum) {
#else
FUNCTION_INLINE bool isAabbInFrustum(fvec3 center, fvec3 extents, fvec4 frustum[6]) {
#endif
	for (uint i = 0; i < 6; ++i) {
		const fvec4 plane = frustum[i];

		const float radius = dot(extents, abs(plane.xyz));
		const float distance = dot(plane.xyz, center) - plane.w;
		if (-radius > distance) {
			return false;
		}
	}
	return true;
}

// See https://gist.github.com/cmf028/81e8d3907035640ee0e3fdd69ada543f#file-aabb_transform-comp-L109-L132
FUNCTION_INLINE fvec3 getWorldSpaceAabbExtent(fvec3 extent, fmat4 transform) {
	const fmat3 transformExtents = fmat3(
		abs(transform[0].xyz),
		abs(transform[1].xyz),
		abs(transform[2].xyz)
	);
	return transformExtents * extent;
}

#if defined(SHADER_GLSL)
// Vertices of a basic cube
SHADER_CONSTANT vec3 aabbPositions[8] = vec3[8](
    vec3(1, -1, -1),
    vec3(1, 1, -1),
    vec3(-1, 1, -1),
    vec3(-1, -1, -1),
    vec3(1, -1, 1),
    vec3(1, 1, 1),
    vec3(-1, -1, 1),
    vec3(-1, 1, 1)
);

/** This projects the given AABB into screen coordinates */
vec3[2] projectAabb(vec3 center, vec3 extent, mat4 viewProjection) {
	vec3 ssMin = vec3(1.f), ssMax = vec3(-1.f);
	for (uint i = 0; i < 8; ++i) {
		const vec3 pos = aabbPositions[i] * extent + center;
		const vec4 clip = viewProjection * vec4(pos, 1);

		const vec2 ndc = clamp(clip.xy / clip.w, -1.f, 1.f);
		const vec2 uv = ndc * 0.5f + 0.5f;
		ssMin = min(ssMin, vec3(uv, clip.z / clip.w));
		ssMax = max(ssMax, vec3(uv, clip.z / clip.w));
	}
	return vec3[2](ssMin, ssMax);
}
#endif

SHADER_NAMESPACE_END
#endif
