#if !defined(__cplusplus)
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#define GLSL_CONSTANT const
#else
// We put all shader declarations into the "glsl" namespace.
// The following using namespace declarations then don't leak outside the header.
namespace glsl {
using uint = std::uint32_t;
using namespace std;
using namespace glm;
#define GLSL_CONSTANT static constexpr
#endif

GLSL_CONSTANT uint maxVertices = 64;
GLSL_CONSTANT uint maxPrimitives = 126;
GLSL_CONSTANT uint maxMeshlets = 128;

struct Meshlet {
    uint vertexOffset;
    uint triangleOffset;

    uint8_t vertexCount;
    uint8_t triangleCount;

    vec3 aabbExtents;
    vec3 aabbCenter;
};

struct Vertex {
    vec3 position;
    vec4 color;

// Quantized float16_t vec2. TODO: glm has half float types, use those?
#if defined(__cplusplus)
    u16vec2 uv;
#else
    f16vec2 uv;
#endif
};

#if !defined(__cplusplus)
struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};
#endif

struct PrimitiveDraw {
    // TODO: Get rid of this command struct here?
    VkDrawMeshTasksIndirectCommandEXT command;

    mat4x4 modelMatrix;

	// TODO: Switch these to VkDeviceSize/uint64_t
    uint descOffset;
    uint vertexIndicesOffset;
    uint triangleIndicesOffset;
    uint verticesOffset;

    uint meshletCount;
    uint materialIndex;
};

struct Material {
    vec4 albedoFactor;

    // Albedo texture
    uint albedoIndex;
    vec2 uvOffset;
    vec2 uvScale;
    float uvRotation;

    float alphaCutoff;
};

#if defined(__cplusplus)
} // namespace shaders
#endif
