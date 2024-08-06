#version 460
#extension GL_GOOGLE_include_directive : require

#include "ui.h"
#include "srgb.h"

layout(location = 0) out FragmentInput outp;

layout(push_constant) uniform Constants {
	UiPushConstants pushConstants;
};

void main() {
	restrict const ImDrawVert vert = pushConstants.vertices.v[gl_VertexIndex];
	outp.color = unpackUnorm4x8(vert.col);
	outp.uv = vert.uv;
	gl_Position = vec4(vert.pos * pushConstants.scale + pushConstants.translate, 0, 1);
}
