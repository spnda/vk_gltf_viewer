#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;
layout(location = 2) flat in uint materialIndex;

layout(location = 0) out vec4 fragColor;

struct Material {
    vec4 albedoFactor;
    uint albedoIdx;
    float alphaCutoff;
};

layout(set = 2, binding = 0, scalar) readonly buffer Materials {
    Material materials[];
};

layout (set = 2, binding = 1) uniform sampler2D textures[];

void main() {
    Material material = materials[materialIndex];
    vec4 outColor = color * material.albedoFactor * texture(textures[nonuniformEXT(material.albedoIdx)], uv);

    if (outColor.a < material.alphaCutoff)
        discard;
    fragColor = outColor;
}
