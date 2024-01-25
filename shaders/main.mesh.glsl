#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_control_flow_attributes: require

const uint maxVertices = 64;
const uint maxPrimitives = 126;

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = maxVertices, max_primitives = maxPrimitives) out;

layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProjection;
} camera;

// This is the definition of meshopt_Meshlet
struct Meshlet {
    uint vertexOffset;
    uint triangleOffset;

    uint vertexCount;
    uint triangleCount;
};

struct Vertex {
    vec4 position;
    vec4 color;
    vec2 uv;
};

struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

struct Primitive {
    // TODO: Get rid of this here?
    VkDrawMeshTasksIndirectCommandEXT command;

    mat4x4 modelMatrix;

    uint descOffset;
    uint vertexIndicesOffset;
    uint triangleIndicesOffset;
    uint verticesOffset;

    uint materialIndex;
};

layout(set = 1, binding = 0, scalar) buffer MeshletDescBuffer {
    Meshlet meshlets[];
};

layout(set = 1, binding = 1, scalar) buffer VertexIndexBuffer {
    uint vertexIndices[];
};

layout(set = 1, binding = 2, scalar) buffer PrimitiveIndexBuffer {
    uint8_t primitiveIndices[];
};

layout(set = 1, binding = 3, scalar) buffer VertexBuffer {
    Vertex vertices[];
};

layout(set = 1, binding = 4, scalar) buffer PrimitiveDrawBuffer {
    Primitive primitives[];
};

layout(location = 0) out vec4 colors[];
layout(location = 1) out vec2 uvs[];
layout(location = 2) flat out uint materialIndex[];

void main() {
    Primitive primitive = primitives[gl_DrawID];
    Meshlet meshlet = meshlets[primitive.descOffset + gl_WorkGroupID.x];

    // This defines the array size of gl_MeshVerticesEXT
    if (gl_LocalInvocationID.x == 0) {
        SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);
    }

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

        gl_MeshVerticesEXT[vidx].gl_Position = camera.viewProjection * primitive.modelMatrix * vertex.position;

        colors[vidx] = vertex.color;
        uvs[vidx] = vertex.uv;
        materialIndex[vidx] = primitive.materialIndex;
    }

    const uint primitiveLoops = (meshlet.triangleCount + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    [[unroll]] for (uint i = 0; i < primitiveLoops; ++i) {
        uint pidx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;

        pidx = min(pidx, meshlet.triangleCount - 1);

        uvec3 indices = uvec3(primitiveIndices[primitive.triangleIndicesOffset + meshlet.triangleOffset + pidx * 3 + 0],
                              primitiveIndices[primitive.triangleIndicesOffset + meshlet.triangleOffset + pidx * 3 + 1],
                              primitiveIndices[primitive.triangleIndicesOffset + meshlet.triangleOffset + pidx * 3 + 2]);

        gl_PrimitiveTriangleIndicesEXT[pidx] = indices;
    }
}
