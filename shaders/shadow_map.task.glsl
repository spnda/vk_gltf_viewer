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

layout(set = 0, binding = 0, scalar) uniform CameraUniform {
    Camera camera;
};

layout(set = 1, binding = 0, scalar) readonly buffer MeshletDescBuffer {
    Meshlet meshlets[];
};

layout(set = 1, binding = 4, scalar) readonly buffer PrimitiveDrawBuffer {
    PrimitiveDraw primitives[];
};

layout(push_constant) uniform Constants {
    uint layerIndex;
};

taskPayloadSharedEXT TaskPayload taskPayload;

#include "frustum_culling.glsl.h"

void main() {
    const PrimitiveDraw primitive = primitives[gl_DrawID];

    // Early return if entire primitive is outside of the frustum
    const vec3 primWorldAabbCenter = (primitive.modelMatrix * vec4(primitive.aabbCenter, 1.0f)).xyz;
    const vec3 primWorldAabbExtent = getWorldSpaceAabbExtent(primitive.aabbExtents.xyz, primitive.modelMatrix);
    if (!isAabbInFrustum(primWorldAabbCenter, primWorldAabbExtent, 0)) {
        return;
    }

    taskPayload.drawID = gl_DrawID;

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
        const bool visible = isAabbInFrustum(worldAabbCenter, worldAabbExtent, layerIndex + 1);

        // Get the index for this thread for this subgroup
        const uint payloadIndex = subgroupExclusiveAdd(uint(visible));
        if (visible) {
            taskPayload.deltaIDs[visibleMeshlets + payloadIndex] = uint8_t(idx);
        }
        visibleMeshlets += subgroupAdd(uint(visible));
    }

    EmitMeshTasksEXT(visibleMeshlets, 1, 1);
}
