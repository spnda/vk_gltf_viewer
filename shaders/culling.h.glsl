#ifndef CULLING_GLSL_H
#define CULLING_GLSL_H

#include "common.h.glsl"
GLSL_NAMESPACE_BEGIN

// Frustum culling using 6 planes on an AABB
#if defined(SHADER_METAL)
bool isAabbInFrustum(metal::float3 center, metal::float3 extents, device const metal::array<metal::packed_float4, 6>& frustum) {
#else
bool isAabbInFrustum(PARAMETER_COPY(vec3) center, PARAMETER_COPY(vec3) extents, PARAMETER_COPY(vec4) frustum[6]) {
#endif
	for (uint i = 0; i < 6; ++i) {
		const vec4 plane = frustum[i];

		const float radius = dot(extents, abs(plane.xyz));
		const float distance = dot(plane.xyz, center) - plane.w;
		if (-radius > distance) {
			return false;
		}
	}
	return true;
}

// See https://gist.github.com/cmf028/81e8d3907035640ee0e3fdd69ada543f#file-aabb_transform-comp-L109-L132
vec3 getWorldSpaceAabbExtent(PARAMETER_COPY(vec3) extent, PARAMETER_COPY(mat4) transform) {
	const mat3 transformExtents = mat3(
		abs(transform[0].xyz),
		abs(transform[1].xyz),
		abs(transform[2].xyz)
	);
#if !defined(SHADER_METAL)
	return transformExtents * extent;
#else
	return transformExtents * metal::float3(extent);
#endif
}

#if !defined(SHADER_METAL)
// Vertices of a basic cube
GLSL_CONSTANT vec3 aabbPositions[8] = vec3[8](
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

GLSL_NAMESPACE_END
#endif
