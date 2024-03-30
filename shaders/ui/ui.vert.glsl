#version 460
#extension GL_GOOGLE_include_directive : require

#include "ui.glsl.h"

layout(location = 0) out FragmentInput outp;

layout(push_constant) uniform Constants {
    PushConstants pushConstants;
};

// Converts a color from sRGB gamma to linear light gamma
vec4 toLinear(vec4 sRGB) {
    bvec3 cutoff = lessThan(sRGB.rgb, vec3(0.04045));
    vec3 higher = pow((sRGB.rgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    vec3 lower = sRGB.rgb / vec3(12.92);

    return vec4(mix(higher, lower, cutoff), sRGB.a);
}

void main() {
    restrict ImDrawVert vert = pushConstants.vertices.v[gl_VertexIndex];
    outp.color = toLinear(unpackUnorm4x8(vert.col));
    outp.uv = vert.uv;
    gl_Position = vec4(vert.pos * pushConstants.scale + pushConstants.translate, 0, 1);
}
