#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes: require

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(constant_id = 0) const uint subgroupSize = 32;

// This now uses the spec constant 0 for the x size
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

#include "mesh_common.glsl.h"

layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProjection;

    // We represent a plane using a single vec4, in the form of ax + by + cz + d = 0
    vec4 frustum[6];
} camera;

layout(set = 1, binding = 0, scalar) readonly buffer MeshletDescBuffer {
    Meshlet meshlets[];
};

layout(set = 1, binding = 4, scalar) readonly buffer PrimitiveDrawBuffer {
    PrimitiveDraw primitives[];
};

// This is essentially a replacement for gl_WorkGroupID.x, but one which can store any index
// between 0..256 instead of the linear requirement of the work group ID.
struct Task {
    uint baseID;
    uint8_t deltaIDs[maxMeshlets];
};

taskPayloadSharedEXT Task taskPayload;

// Frustum culling using 6 planes on an AABB
bool isMeshletVisibleAabb(in vec3 center, in vec3 extents) {
    [[unroll]] for (uint i = 0; i < 6; ++i) {
        const vec4 plane = camera.frustum[i];

        const float radius = dot(extents, abs(plane.xyz));
        const float distance = dot(plane.xyz, center) - plane.w;
        if (-radius > distance) {
            return false;
        }
    }
    return true;
}

// See https://gist.github.com/cmf028/81e8d3907035640ee0e3fdd69ada543f#file-aabb_transform-comp-L109-L132
vec3 getWorldSpaceAabbExtent(in vec3 extent, in mat4 transform) {
    const mat3 transformExtents = mat3(
        abs(vec3(transform[0])),
        abs(vec3(transform[1])),
        abs(vec3(transform[2]))
    );
    return transformExtents * extent;
}

void main() {
    const PrimitiveDraw primitive = primitives[gl_DrawID];

    // Every task shader workgroup only gets 128 meshlets to handle. This calculates how many
    // this specific work group should handle, and sets the baseID accordingly.
    uint meshletCount = min(maxMeshlets, primitive.meshletCount - (gl_WorkGroupID.x * maxMeshlets));
    taskPayload.baseID = gl_WorkGroupID.x * maxMeshlets;

    // Generate the delta IDs by iterating over every meshlet.
    const uint meshletLoops = (meshletCount + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    uint visibleMeshlets = 0;
    [[unroll]] for (uint i = 0; i < meshletLoops; ++i) {
        uint idx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;
        idx = min(idx, meshletCount - 1);
        const Meshlet meshlet = meshlets[primitive.descOffset + taskPayload.baseID + idx];

        // Do some culling
        const vec3 worldAabbCenter = (primitive.modelMatrix * vec4(meshlet.aabbCenter, 1.0f)).xyz;
        const vec3 worldAabbExtent = getWorldSpaceAabbExtent(meshlet.aabbExtents.xyz, primitive.modelMatrix);
        const bool visible = isMeshletVisibleAabb(worldAabbCenter, worldAabbExtent);

        // Get the index for this thread for this subgroup
        uint payloadIndex = subgroupExclusiveAdd(uint(visible));
        if (visible) {
            taskPayload.deltaIDs[visibleMeshlets + payloadIndex] = uint8_t(idx);
        }
        visibleMeshlets += subgroupAdd(uint(visible));
    }

    EmitMeshTasksEXT(visibleMeshlets, 1, 1);
}
