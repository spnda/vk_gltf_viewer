#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "ui.h"
#include "srgb.h"

layout(location = 0) in FragmentInput inp;

layout(location = 0) out vec4 fragColor;

layout(push_constant, scalar) uniform Constants {
	UiPushConstants pushConstants;
};

void main() {
	fragColor = inp.color * texture(sampled_textures_heap[pushConstants.imageIndex], inp.uv.st);
}
