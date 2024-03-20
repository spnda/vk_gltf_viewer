#version 460
#extension GL_GOOGLE_include_directive : require

#include "ui.glsl.h"

layout(set = 0, binding = 0) uniform sampler2D sTexture;

layout(location = 0) out vec4 fragColor;

layout(location = 0) in FragmentInput inp;

void main() {
    // With ImGuiConfigFlags_IsSRGB enabled, ImGui will not encode vertex colours as sRGB itself.
    fragColor = inp.color * texture(sTexture, inp.uv.st);
}
