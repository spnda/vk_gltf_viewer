#ifndef SHADERS_UI_H
#define SHADERS_UI_H

#include "common.h"

#if defined(SHADER_GLSL)
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#endif

#include "resource_table.h"
SHADER_NAMESPACE_BEGIN

#if !defined(SHADER_CPP)
// TODO: Find a way to share the ImDrawVert definition with this shader header?
struct ImDrawVert {
	packed_fvec2 pos;
	packed_fvec2 uv;
	// The color is always in sRGB currently with ImGui.
	uint32_t col;
};

#if defined(SHADER_GLSL)
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Vertices {
	ImDrawVert v[];
};
#endif
#endif

struct UiPushConstants {
	packed_fvec2 scale;
	packed_fvec2 translate;
	BUFFER_REF(Vertices, ImDrawVert) vertices MEMBER_INIT(0);
	ResourceTableHandle imageIndex MEMBER_INIT(invalidHandle);
};

struct FragmentInput {
	fvec4 color;
	fvec2 uv;
};

SHADER_NAMESPACE_END
#endif
