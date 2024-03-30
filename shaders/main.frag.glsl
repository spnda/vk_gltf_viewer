#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : require

#include "mesh_common.glsl.h"

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;
layout(location = 2) flat in uint materialIndex;
layout(location = 3) in vec4 lightSpacePos;

layout(location = 0) out vec4 fragColor;

layout(set = 2, binding = 0, scalar) readonly buffer Materials {
    Material materials[];
};

layout (set = 2, binding = 1) uniform sampler2D shadowMap;

layout (set = 2, binding = 2) uniform sampler2D textures[];

vec2 transformUv(in Material material, vec2 uv) {
    mat2 rotationMat = mat2(
        cos(material.uvRotation), -sin(material.uvRotation),
        sin(material.uvRotation), cos(material.uvRotation)
    );
    return rotationMat * uv * material.uvScale + material.uvOffset;
}

float shadow(vec4 lightSpacePos) {
    // Perspective divide
    vec3 coords = lightSpacePos.xyz / lightSpacePos.w;

    // Transform from NDC into UV coordinates. Note that with Vulkan the Z range is [0,1] already,
    // so we only need to transform X and Y.
    coords.xy = coords.xy * 0.5 + 0.5;

    // Sample from the shadow map and determine if we are the closest fragment for the light
    // We, also, use reversed Z for the shadow maps
    float closestDepth = texture(shadowMap, coords.xy).r;
    float currentDepth = coords.z;

    // Prevent oversampling when fragment is behind the far plane of the light matrix
    if (coords.z > 1.0f)
        return 0.0f;
    return currentDepth - 0.005f > closestDepth ? 0.5f : 0.0f;
}

void main() {
    Material material = materials[materialIndex];
    vec4 sampled = texture(textures[material.albedoIndex], transformUv(material, uv));
    vec4 outColor = color * material.albedoFactor * sampled;
    outColor = vec4(outColor.xyz * (1.0f - shadow(lightSpacePos)), outColor.w);

    if (outColor.a < material.alphaCutoff)
        discard;
    fragColor = outColor;
}
