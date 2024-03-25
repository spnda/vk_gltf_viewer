#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

const uint maxVertices = 64;
const uint maxPrimitives = 126;
const uint maxMeshlets = 128;

struct Meshlet {
    uint vertexOffset;
    uint triangleOffset;

    uint8_t vertexCount;
    uint8_t triangleCount;

    vec3 aabbExtents;
    vec3 aabbCenter;
};

struct Vertex {
    vec4 position;
    vec4 color;
    f16vec2 uv;
};

struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

struct Primitive {
    // TODO: Get rid of this command struct here?
    VkDrawMeshTasksIndirectCommandEXT command;

    mat4x4 modelMatrix;

    uint descOffset;
    uint vertexIndicesOffset;
    uint triangleIndicesOffset;
    uint verticesOffset;

    uint meshletCount;
    uint materialIndex;
};
