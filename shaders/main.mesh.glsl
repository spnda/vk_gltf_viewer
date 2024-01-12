#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_8bit_storage : require

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

// Fragment input
layout (location = 0) out Outputs {
    vec4 color;
} outp[];

#define MAX_COLORS 10
vec3 meshletcolors[MAX_COLORS] = {
    vec3(1,0,0),
    vec3(0,1,0),
    vec3(0,0,1),
    vec3(1,1,0),
    vec3(1,0,1),
    vec3(0,1,1),
    vec3(1,0.5,0),
    vec3(0.5,1,0),
    vec3(0,0.5,1),
    vec3(1,1,1)
};

void main() {
    Meshlet meshlet = meshlets[gl_WorkGroupID.x];

    // This defines the array size of gl_MeshVerticesEXT
    if (gl_LocalInvocationID.x == 0) {
        SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);
    }

    // The max_vertices does not match the local workgroup size.
    // Therefore, we'll have this loop that will run over all possible vertices.
    const uint vertexLoops = (maxVertices + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    for (uint i = 0; i < vertexLoops; ++i) {
        // Distribute each vertex of the loop over the workgroup.
        uint vidx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;

        // Avoid branching but also respect the meshlet vertexCount.
        // This will redundantly compute the last vertex multiple times.
        // Lowering the workgroup size will reduce this over-computation.
        vidx = min(vidx, meshlet.vertexCount - 1);

        uint vertexIndex = vertexIndices[meshlet.vertexOffset + vidx];
        vec4 vertex = vertices[vertexIndex].position;

        gl_MeshVerticesEXT[vidx].gl_Position = camera.viewProjection * vertex;

        // TODO: Assign actual vertex colors.
        outp[vidx].color = vec4(meshletcolors[gl_WorkGroupID.x % MAX_COLORS], 1.0f); // vec3(vidx / maxVertices);
    }

    const uint primitiveLoops = (maxPrimitives + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    for (uint i = 0; i < primitiveLoops; ++i) {
        uint pidx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;

        pidx = min(pidx, meshlet.triangleCount - 1);

        uvec3 indices = uvec3(primitiveIndices[meshlet.triangleOffset + pidx * 3 + 0],
                              primitiveIndices[meshlet.triangleOffset + pidx * 3 + 1],
                              primitiveIndices[meshlet.triangleOffset + pidx * 3 + 2]);

        gl_PrimitiveTriangleIndicesEXT[pidx] = indices;
    }
}
