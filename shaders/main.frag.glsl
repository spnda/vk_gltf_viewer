#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_fragment_shader_barycentric : require

#include "mesh_common.glsl.h"

layout(location = 0) pervertexEXT in uint vertexIndex[];
layout(location = 1) in vec3 worldSpacePos;
layout(location = 2) flat in uint materialIndex;

vec4 interpolateColor(const in u8vec4 quantizedColor[3]) {
    return gl_BaryCoordEXT.x * unpackVertexColor(quantizedColor[0])
        + gl_BaryCoordEXT.y * unpackVertexColor(quantizedColor[1])
        + gl_BaryCoordEXT.z * unpackVertexColor(quantizedColor[2]);
}
vec3 interpolateNormal(const in u8vec3 quantizedNormal[3]) {
    return gl_BaryCoordEXT.x * unpackVertexNormal(quantizedNormal[0])
        + gl_BaryCoordEXT.y * unpackVertexNormal(quantizedNormal[1])
        + gl_BaryCoordEXT.z * unpackVertexNormal(quantizedNormal[2]);
}
vec2 interpolateUv(const in f16vec2 quantizedUv[3]) {
    return gl_BaryCoordEXT.x * vec2(quantizedUv[0])
        + gl_BaryCoordEXT.y * vec2(quantizedUv[1])
        + gl_BaryCoordEXT.z * vec2(quantizedUv[2]);
}

layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0, scalar) restrict readonly uniform CameraUniform {
    Camera camera;
};

layout(set = 1, binding = 3, scalar) restrict readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(set = 2, binding = 0, scalar) restrict readonly buffer Materials {
    Material materials[];
};

layout (set = 2, binding = 1) uniform sampler2DArrayShadow shadowMap;

layout (set = 2, binding = 2) uniform sampler2D textures[];

vec2 transformUv(in Material material, vec2 uv) {
    mat2 rotationMat = mat2(
        cos(material.uvRotation), -sin(material.uvRotation),
        sin(material.uvRotation), cos(material.uvRotation)
    );
    return rotationMat * uv * material.uvScale + material.uvOffset;
}

// Calculate the "perfect" shadow bias as per https://www.desmos.com/calculator/nbhoiubvfj
// The texelSize needs to be the world width of a single texel of the shadow map.
float getBaseShadowBias(in vec3 L, in vec3 N, in float texelSize) {
    const float b = 1.41411356f * texelSize.x / 2.0f; // *sqrt(2) for diagonal length, effectively just length(texelSize)
    const float NoL = clamp(abs(dot(N, L)), 0.0001f, 1.f);
    // 0.01f as a base bias, and 2.f / (1 << 23) as a basic value to avoid numerical issues
    return (0.01f + 2.f / (1 << 23) + b * length(cross(N, L)) / NoL);
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

    RenderView view = camera.views[layer + 1];

    // Get the current fragment's position for the light's view.
    vec4 lightSpacePos = view.viewProjection * vec4(worldSpacePos, 1.0f);
    vec3 coords = lightSpacePos.xyz / lightSpacePos.w; // Perspective divide to get screen coordinates

    // Transform from NDC into UV coordinates. Note that with Vulkan the Z range is [0,1] already,
    // so we only need to transform X and Y.
    coords.xy = coords.xy * 0.5 + 0.5;

    // Prevent oversampling when fragment is behind the far plane of the light matrix
    if (coords.z > 1.0f)
        return 0.0f;

    const vec3 L = -normalize(camera.lightDirection);
    const vec2 texelSize = 1.f / textureSize(shadowMap, 0).xy;

    // Resize the texelSize to be in world space
    const float baseBias = getBaseShadowBias(L, normal, texelSize.x * view.projectionWidth);

    // Use hardware PCF with sampler2DArrayShadow, and do manual PCF with a smaller kernel.
    const float kernelSize = 1.5;
    float shadow = 0.f;
    for (float x = -kernelSize; x <= kernelSize; ++x) {
        for (float y = -kernelSize; y <= kernelSize; ++y) {
            const vec2 offset = vec2(x, y) * texelSize;
            const float dist = length(offset);

            // We need to compute an additional offset because of PCF, since the nearby texels might require a different bias.
            const float bias = (baseBias) / view.projectionZLength;

            // The texture() function does depth check for us, which allows the hardware to do PCF too.
            shadow += texture(shadowMap, vec4(coords.xy + offset, layer, coords.z + bias));
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
    restrict const Material material = materials[materialIndex];

    // Do vertex fetch for the other attributes apart from position
    restrict const Vertex vertex1 = vertices[vertexIndex[0]];
    restrict const Vertex vertex2 = vertices[vertexIndex[1]];
    restrict const Vertex vertex3 = vertices[vertexIndex[2]];

    vec3 ambient = vec3(0.1, 0.1, 0.1);

    // Interpolate UV coordinates and the quantized vertex color
    vec2 uv = interpolateUv(f16vec2[](vertex1.uv, vertex2.uv, vertex3.uv));
    vec4 vtxColor = interpolateColor(u8vec4[](vertex1.color, vertex2.color, vertex3.color));

    // the glTF baseColorTexture contains sRGB encoded values.
    vec4 sampled = texture(textures[material.albedoIndex], transformUv(material, uv));
    vec4 albedoColor = vtxColor * material.albedoFactor * toLinear(sampled);
    if (albedoColor.a < material.alphaCutoff)
        discard;

    // Interpolate the quantized normal
    vec3 normal = interpolateNormal(u8vec3[](vertex1.normal, vertex2.normal, vertex3.normal));
    vec3 diffuse = vec3(max(dot(normal, -camera.lightDirection), 0.f));

    // We use the vertex normals for the shadow bias calculation, as the self shadowing is caused by the geometry.
    vec3 result = (ambient + (1.0f - shadow(normal, worldSpacePos)) * diffuse) * albedoColor.xyz;

    // Reinhard tonemapping
    const float exposure = 1.f;
    vec3 mapped = vec3(1.f) - exp(-result * exposure);
    fragColor = vec4(mapped, 1);
}
