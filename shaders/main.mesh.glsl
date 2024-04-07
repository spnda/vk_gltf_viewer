#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

#include "mesh_common.glsl.h"

layout(triangles, max_vertices = maxVertices, max_primitives = maxPrimitives) out;

layout(set = 0, binding = 0, scalar) uniform CameraUniform {
    Camera camera;
};

layout(set = 1, binding = 0, scalar) readonly buffer MeshletDescBuffer {
    Meshlet meshlets[];
};

layout(set = 1, binding = 1, scalar) readonly buffer VertexIndexBuffer {
    uint vertexIndices[];
};

layout(set = 1, binding = 2, scalar) readonly buffer PrimitiveIndexBuffer {
    uint8_t primitiveIndices[];
};

layout(set = 1, binding = 3, scalar) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(set = 1, binding = 4, scalar) readonly buffer PrimitiveDrawBuffer {
    PrimitiveDraw primitives[];
};

layout(set = 2, binding = 0, scalar) readonly buffer Materials {
    Material materials[];
};

taskPayloadSharedEXT TaskPayload taskPayload;

layout(location = 0) out vec4 colors[];
layout(location = 1) out vec2 uvs[];
layout(location = 2) out vec3 worldSpacePos[];
layout(location = 3) perprimitiveEXT flat out uint materialIndex[];

shared vec3 clipVertices[maxPrimitives];

void main() {
    const PrimitiveDraw primitive = primitives[gl_DrawID];
    uint deltaId = taskPayload.baseID + uint(taskPayload.deltaIDs[gl_WorkGroupID.x]);
    const Meshlet meshlet = meshlets[primitive.descOffset + deltaId];

    const Material material = materials[primitive.materialIndex];

    // This defines the array size of gl_MeshVerticesEXT
    if (gl_LocalInvocationID.x == 0) {
        SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);
    }

    mat4 mvp = camera.viewProjection * primitive.modelMatrix;

    // The max_vertices does not match the local workgroup size.
    // Therefore, we'll have this loop that will run over all possible vertices.
    const uint vertexLoops = (meshlet.vertexCount + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    [[unroll]] for (uint i = 0; i < vertexLoops; ++i) {
        // Distribute each vertex of the loop over the workgroup.
        uint vidx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;

        // Avoid branching but also respect the meshlet vertexCount.
        // This will redundantly compute the last vertex multiple times.
        // Lowering the workgroup size will reduce this over-computation.
        vidx = min(vidx, meshlet.vertexCount - 1);

        uint vertexIndex = vertexIndices[primitive.vertexIndicesOffset + meshlet.vertexOffset + vidx];
        Vertex vertex = vertices[primitive.verticesOffset + vertexIndex];

        vec4 pos = mvp * vec4(vertex.position, 1.0f);
        gl_MeshVerticesEXT[vidx].gl_Position = pos;
        worldSpacePos[vidx] = (primitive.modelMatrix * vec4(vertex.position, 1.0f)).xyz;
        clipVertices[vidx] = pos.xyw;

        colors[vidx] = vertex.color;
        uvs[vidx] = vertex.uv;
    }

    const float transformDet = determinant(primitive.modelMatrix);
    const uint primitiveLoops = (meshlet.triangleCount + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    [[unroll]] for (uint i = 0; i < primitiveLoops; ++i) {
        uint pidx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;

        pidx = min(pidx, meshlet.triangleCount - 1);

        uvec3 indices = uvec3(primitiveIndices[primitive.triangleIndicesOffset + meshlet.triangleOffset + pidx * 3 + 0],
                              primitiveIndices[primitive.triangleIndicesOffset + meshlet.triangleOffset + pidx * 3 + 1],
                              primitiveIndices[primitive.triangleIndicesOffset + meshlet.triangleOffset + pidx * 3 + 2]);

        gl_PrimitiveTriangleIndicesEXT[pidx] = indices;
        materialIndex[pidx] = primitive.materialIndex;

        if (!material.doubleSided) {
            // The glTF spec says:
            // If the determinant of the transform is a negative value, the winding order of the mesh triangle faces should be reversed.
            // This supports negative scales for mirroring geometry.
            const vec3 v0 = clipVertices[indices.x];
            const vec3 v1 = clipVertices[indices.y];
            const vec3 v2 = clipVertices[indices.z];
            const float det = determinant(mat3(v0, v1, v2));
            if (transformDet < 0.0f) {
                gl_MeshPrimitivesEXT[pidx].gl_CullPrimitiveEXT = det < 0.0f; // Front face culling
            } else {
                gl_MeshPrimitivesEXT[pidx].gl_CullPrimitiveEXT = det > 0.0f; // Back face culling
            }
        }
    }
}
