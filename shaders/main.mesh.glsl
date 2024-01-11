#version 460
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;

// Fragment input
layout (location = 0) out Outputs {
    vec3 color;
} outp[];

void main() {
    // This defines the array size of gl_MeshVerticesEXT
    uint vertexCount = 3;
    uint triangleCount = 1;
    SetMeshOutputsEXT(vertexCount, triangleCount);

    const vec3 positions[3] = vec3[3](
        vec3(1.f,1.f, 0.0f),
        vec3(-1.f,1.f, 0.0f),
        vec3(0.f,-1.f, 0.0f)
    );

    const vec3 colors[3] = vec3[3](
        vec3(1.0f, 0.0f, 0.0f),
        vec3(0.0f, 1.0f, 0.0f),
        vec3(00.f, 0.0f, 1.0f)
    );

    gl_MeshVerticesEXT[0].gl_Position = vec4(positions[0], 1);
    gl_MeshVerticesEXT[1].gl_Position = vec4(positions[1], 1);
    gl_MeshVerticesEXT[2].gl_Position = vec4(positions[2], 1);
    gl_PrimitiveTriangleIndicesEXT[0] =  uvec3(0, 1, 2);

    outp[0].color = colors[0];
    outp[1].color = colors[1];
    outp[2].color = colors[2];
}
