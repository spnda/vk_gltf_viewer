#ifndef FRUSTUM_CULLING_GLSL_H
#define FRUSTUM_CULLING_GLSL_H
#include "common.glsl.h"
GLSL_NAMESPACE_BEGIN

// Frustum culling using 6 planes on an AABB
bool isAabbInFrustum(PARAMETER_COPY(vec3) center, PARAMETER_COPY(vec3) extents, PARAMETER_COPY(vec4) frustum[6]) {
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
		abs(vec3(transform[0])),
		abs(vec3(transform[1])),
		abs(vec3(transform[2]))
	);
	return transformExtents * extent;
}

GLSL_NAMESPACE_END
#endif
