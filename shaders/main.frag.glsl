#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : require

#include "mesh_common.glsl.h"

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 worldSpacePos;
layout(location = 3) in vec3 normal;
layout(location = 4) flat in uint materialIndex;

layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0, scalar) uniform CameraUniform {
    Camera camera;
};

layout(set = 2, binding = 0, scalar) readonly buffer Materials {
    Material materials[];
};

layout (set = 2, binding = 1) uniform sampler2DArray shadowMap;

layout (set = 2, binding = 2) uniform sampler2D textures[];

vec2 transformUv(in Material material, vec2 uv) {
    mat2 rotationMat = mat2(
        cos(material.uvRotation), -sin(material.uvRotation),
        sin(material.uvRotation), cos(material.uvRotation)
    );
    return rotationMat * uv * material.uvScale + material.uvOffset;
}

float shadow(in vec3 normal, in vec3 worldSpacePos) {
    // Select cascade layer (fallback to the last layer)
    float depthValue = abs((camera.view * vec4(worldSpacePos, 1.0f)).z);
    uint layer = shadowMapCount - 1;
    for (int i = 0; i < shadowMapCount; ++i) {
        if (depthValue < camera.splitDistances[i]) {
            layer = i;
            break;
        }
    }

    // Get the current fragment's position for the light's view.
    vec4 lightSpacePos = camera.views[layer + 1].viewProjection * vec4(worldSpacePos, 1.0f);
    vec3 coords = lightSpacePos.xyz / lightSpacePos.w; // Perspective divide to get screen coordinates

    // Transform from NDC into UV coordinates. Note that with Vulkan the Z range is [0,1] already,
    // so we only need to transform X and Y.
    coords.xy = coords.xy * 0.5 + 0.5;

    // Prevent oversampling when fragment is behind the far plane of the light matrix
    if (coords.z > 1.0f)
        return 0.0f;

    // Basic PCSS (PCF with soft shadows), additionally sampling 8 texels
    // Compute a kernel size, depending on the distance from the light blocker as described here:
    // https://developer.download.nvidia.com/shaderlibrary/docs/shadow_PCSS.pdf
    const float receiverDepth = coords.z;
    const float depthBlocker = texture(shadowMap, vec3(coords.xy, layer)).r;
    int kernelSize = clamp(int((receiverDepth - depthBlocker) * 50.f / depthBlocker), 1, 3); // Clamp between 1 and 3 for performance reasons
    vec2 texelSize = 1.f / textureSize(shadowMap, 0).xy;

    // Calculate the "perfect" shadow bias as per https://www.desmos.com/calculator/nbhoiubvfj
    const vec3 L = -normalize(camera.lightDirection);
    const float b = 1.41411356f * texelSize.x / 2.0f; // *sqrt(2) for diagonal length, effectively just length(texelSize)
    const float NoL = clamp(abs(dot(normal, L)), 0.0001f, 1.f);
    float bias = 2.f / (1 << 23) + b * length(cross(normal, L)) / NoL;
    bias = (0.01f + bias) / camera.views[layer + 1].projectionZLength;

    // Run the PCF kernel
    float shadow = 0.0f;
    for (int x = -kernelSize; x <= kernelSize; ++x) {
        for (int y = -kernelSize; y <= kernelSize; ++y) {
            // Sample from the shadow map with an offset and determine if we are the closest fragment for the light
            float pcfDepth = texture(shadowMap, vec3(coords.xy + vec2(x, y) * texelSize, layer)).r;
            shadow += receiverDepth < pcfDepth - bias ? 1.f : 0.f;
        }
    }
    return shadow / pow(2 * kernelSize + 1, 2);
}

// Converts a color from sRGB gamma to linear light gamma
// See https://gamedev.stackexchange.com/a/194038/159451
vec4 toLinear(vec4 sRGB) {
    bvec3 cutoff = lessThan(sRGB.rgb, vec3(0.04045));
    vec3 higher = pow((sRGB.rgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    vec3 lower = sRGB.rgb / vec3(12.92);

    return vec4(mix(higher, lower, cutoff), sRGB.a);
}

void main() {
    const Material material = materials[materialIndex];

    vec3 ambient = vec3(0.1, 0.1, 0.1);

    // the glTF baseColorTexture contains sRGB encoded values.
    vec4 sampled = texture(textures[material.albedoIndex], transformUv(material, uv));
    vec4 albedoColor = color * material.albedoFactor * toLinear(sampled);
    if (albedoColor.a < material.alphaCutoff)
        discard;

    vec3 diffuse = vec3(max(dot(normal, -camera.lightDirection), 0.f));

    // We use the vertex normals for the shadow bias calculation, as the self shadowing is caused by the geometry.
    vec3 result = (ambient + (1.0f - shadow(normal, worldSpacePos)) * diffuse) * albedoColor.xyz;

    // Reinhard tonemapping
    const float exposure = 1.f;
    vec3 mapped = vec3(1.f) - exp(-result * exposure);
    fragColor = vec4(mapped, 1);
}
