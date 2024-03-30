#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "ui.glsl.h"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(location = 0) out vec4 fragColor;

layout(location = 0) in FragmentInput inp;

layout(push_constant) uniform Constants {
    PushConstants pushConstants;
};

void main() {
    fragColor = inp.color * texture(textures[pushConstants.imageIndex], inp.uv.st);
}
