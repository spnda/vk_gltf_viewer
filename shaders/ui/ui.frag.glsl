#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "ui.glsl.h"
#include "srgb.glsl.h"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(location = 0) in FragmentInput inp;

layout(location = 0) out vec4 fragColor;

layout(push_constant, scalar) uniform Constants {
	UiPushConstants pushConstants;
};

void main() {
	fragColor = inp.color * texture(textures[pushConstants.imageIndex], inp.uv.st);
}
