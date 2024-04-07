#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_scalar_block_layout : require

#include "mesh_common.glsl.h"

layout(set = 0, binding = 0, scalar) uniform CameraUniform {
    Camera camera;
};

layout(set = 1, binding = 0, scalar) readonly buffer MeshletDescBuffer {
    Meshlet meshlets[];
};

layout(set = 1, binding = 4, scalar) readonly buffer PrimitiveDrawBuffer {
    PrimitiveDraw primitives[];
};

// Vertices of a basic cube
const vec3 positions[8] = vec3[8](
    vec3(1, -1, -1),
    vec3(1, 1, -1),
    vec3(-1, 1, -1),
    vec3(-1, -1, -1),
    vec3(1, -1, 1),
    vec3(1, 1, 1),
    vec3(-1, -1, 1),
    vec3(-1, 1, 1)
);

// Edge indices for a basic cube
const uint edges[12 * 2] = uint[12 * 2](
    0, 1,
    0, 3,
    0, 4,
    2, 1,
    2, 3,
    2, 7,
    6, 3,
    6, 4,
    6, 7,
    5, 1,
    5, 4,
    5, 7
);

// Simple shader to take meshlet AABBs and transform them into a visible cube using line topology.
void main() {
    PrimitiveDraw primitive = primitives[gl_DrawID];
    Meshlet meshlet = meshlets[primitive.descOffset + gl_InstanceIndex];

    vec3 position = positions[edges[gl_VertexIndex]];
    vec3 pos = position * meshlet.aabbExtents.xyz + meshlet.aabbCenter.xyz;

    gl_Position = camera.viewProjection * primitive.modelMatrix * vec4(pos, 1.0f);
}
