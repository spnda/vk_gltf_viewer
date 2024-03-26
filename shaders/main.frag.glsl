#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : require

#include "mesh_common.glsl.h"

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;
layout(location = 2) flat in uint materialIndex;

layout(location = 0) out vec4 fragColor;

layout(set = 2, binding = 0, scalar) readonly buffer Materials {
    Material materials[];
};

layout (set = 2, binding = 1) uniform sampler2D textures[];

vec2 transformUv(in Material material, vec2 uv) {
    mat2 rotationMat = mat2(
        cos(material.uvRotation), -sin(material.uvRotation),
        sin(material.uvRotation), cos(material.uvRotation)
    );
    return rotationMat * uv * material.uvScale + material.uvOffset;
}

void main() {
    Material material = materials[materialIndex];
    vec4 sampled = texture(textures[material.albedoIndex], transformUv(material, uv));
    vec4 outColor = color * material.albedoFactor * sampled;

    if (outColor.a < material.alphaCutoff)
        discard;
    fragColor = outColor;
}
