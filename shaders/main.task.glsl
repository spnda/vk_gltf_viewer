#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes: require

#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

struct Meshlet {
    uint vertexOffset;
    uint triangleOffset;

    uint vertexCount;
    uint triangleCount;
};

struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

struct Primitive {
    VkDrawMeshTasksIndirectCommandEXT command;

    mat4x4 modelMatrix;

    uint descOffset;
    uint vertexIndicesOffset;
    uint triangleIndicesOffset;
    uint verticesOffset;

    uint meshletCount;
    uint materialIndex;
};

layout(set = 1, binding = 0, scalar) buffer MeshletDescBuffer {
    Meshlet meshlets[];
};
layout(set = 1, binding = 4, scalar) buffer PrimitiveDrawBuffer {
    Primitive primitives[];
};

const uint maxMeshlets = 128;
// This is essentially a replacement for gl_WorkGroupID.x, but one which can store any index
// between 0..256 instead of the linear requirement of the work group ID.
struct Task {
    uint baseID;
    uint8_t deltaIDs[maxMeshlets];
};

taskPayloadSharedEXT Task taskPayload;

void main() {
    Primitive primitive = primitives[gl_DrawID];

    // Every task shader workgroup only gets 128 meshlets to handle. This calculates how many
    // this specific work group should handle, and sets the baseID accordingly.
    uint meshletCount = min(maxMeshlets, primitive.meshletCount - (gl_WorkGroupID.x * maxMeshlets));
    taskPayload.baseID = gl_WorkGroupID.x * maxMeshlets;

    // Generate the delta IDs by iterating over every meshlet.
    const uint meshletLoops = (meshletCount + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    [[unroll]] for (uint i = 0; i < meshletLoops; ++i) {
        uint idx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;
        idx = min(idx, meshletCount - 1);
        uint meshletID = taskPayload.baseID + idx;
        taskPayload.deltaIDs[idx] = uint8_t(idx);
    }

    EmitMeshTasksEXT(meshletCount, 1, 1);
}
