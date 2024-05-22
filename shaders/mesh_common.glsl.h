#if !defined(__cplusplus)
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
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

GLSL_CONSTANT uint shadowMapCount = 4;

struct RenderView {
    mat4 viewProjection;

    // We represent a plane using a single vec4, in the form of ax + by + cz + d = 0
    vec4 frustum[6];

    // zFar - zNear
    float projectionZLength;
    float projectionWidth;
};

struct Camera {
    mat4 view;

    vec3 lightDirection;
    float splitDistances[shadowMapCount];

    // The first view is the viewport, the rest are the sun light cascades.
    RenderView views[1 + shadowMapCount];
};

// TODO: These are the optimal values for NVIDIA. What about the others?
GLSL_CONSTANT uint maxVertices = 64;
GLSL_CONSTANT uint maxPrimitives = 126;
GLSL_CONSTANT uint maxMeshlets = 98;

// This is essentially a replacement for gl_WorkGroupID.x, but one which can store any index
// between 0..256 instead of the linear requirement of the work group ID.
// For NVIDIA, we try to keep this structure below 108 bytes to keep it in shared memory.
// Therefore, this is exactly 106 bytes big (4 + 4 + 98 * 1).
struct TaskPayload {
    uint drawID; // gl_DrawID, as its value is undefined in the mesh shader
    uint baseID;
    uint8_t deltaIDs[maxMeshlets];
};

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
    u8vec4 color;
    u8vec3 normal;

// Quantized float16_t vec2. TODO: glm has half float types, use those?
#if defined(__cplusplus)
    u16vec2 uv;
#else
    f16vec2 uv;
#endif
};

#if !defined(__cplusplus)
vec4 unpackVertexColor(in u8vec4 color) {
    return vec4(color) / 255.f;
}

vec3 unpackVertexNormal(in u8vec3 normal) {
    return normalize(vec3(normal) / 127.f - 1.f);
}
#endif

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

    vec3 aabbExtents;
    vec3 aabbCenter;
};

struct Material {
    vec4 albedoFactor;

    // Albedo texture
    uint albedoIndex;
    vec2 uvOffset;
    vec2 uvScale;
    float uvRotation;

    float alphaCutoff;
    bool doubleSided;
};

#if defined(__cplusplus)
} // namespace glsl
#endif
