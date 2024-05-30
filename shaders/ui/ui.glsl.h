#ifndef UI_GLSL_H
#define UI_GLSL_H

#if !defined(__cplusplus)
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#endif

#include "../common.glsl.h"
GLSL_NAMESPACE_BEGIN

#if !defined(__cplusplus)
// TODO: Find a way to share the ImDrawVert definition with this shader header?
struct ImDrawVert {
	vec2 pos;
	vec2 uv;
	// The color is always in sRGB currently with ImGui.
	uint col;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Vertices {
	ImDrawVert v[];
};
#endif

struct UiPushConstants {
	vec2 scale;
	vec2 translate;
	BUFFER_REF(Vertices) vertices MEMBER_INIT(0);
	uint imageIndex;
};

struct FragmentInput {
	vec4 color;
	vec2 uv;
};

GLSL_NAMESPACE_END
#endif
