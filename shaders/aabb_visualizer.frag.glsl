#version 460
#extension GL_GOOGLE_include_directive : require

#include "mesh_common.glsl.h"

layout (location = 0) out vec4 outFragColor;

void main() {
    outFragColor = vec4(1.f,0.f,0.f,0.5f);
}
