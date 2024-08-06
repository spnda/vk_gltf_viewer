#ifndef SHADERS_MESH_COMMON_H
#define SHADERS_MESH_COMMON_H

#include "common.h"

#if defined(SHADER_GLSL)
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#include "resource_table.h"
SHADER_NAMESPACE_BEGIN

SHADER_CONSTANT uint32_t shadowMapCount = 4;

struct Camera {
	ALIGN_AS(16) fmat4 prevViewProjection;
	ALIGN_AS(16) fmat4 prevOcclusionViewProjection;

	ALIGN_AS(16) fmat4 viewProjection;
	ALIGN_AS(16) fmat4 occlusionViewProjection;
	SHADER_ARRAY(packed_fvec4, frustum, 6);
};

#if defined(SHADER_GLSL)
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer CameraBuffer {
	Camera camera;
};
#endif

// TODO: These are the optimal values for NVIDIA. What about the others?
#if defined(SHADER_GLSL)
SHADER_CONSTANT uint32_t maxVertices = 64;
SHADER_CONSTANT uint32_t maxPrimitives = 126;
SHADER_CONSTANT uint32_t maxMeshlets = 102;
#else
SHADER_CONSTANT uint32_t maxVertices = 64;
SHADER_CONSTANT uint32_t maxPrimitives = 128;
SHADER_CONSTANT uint32_t maxMeshlets = 64; // This should best be a multiple of the threadgroup size.
#endif

// This is essentially a replacement for gl_WorkGroupID.x, but one which can store any index
// between 0..256 instead of the linear requirement of the work group ID.
// For NVIDIA, we try to keep this structure below 108 bytes to keep it in shared memory.
// Therefore, this is exactly 106 bytes big (4 + 102 * 1).
struct TaskPayload {
	uint32_t baseID;
	uint8_t deltaIDs[maxMeshlets];
};

struct Meshlet {
	uint32_t vertexOffset;
	uint32_t triangleOffset;

	uint8_t vertexCount;
	uint8_t triangleCount;

	packed_fvec3 aabbExtents;
	packed_fvec3 aabbCenter;
};

struct Vertex {
	packed_fvec3 position;
	packed_u8vec4 color;
	packed_u8vec3 normal;

// Quantized float16_t vec2. TODO: glm has half float types, use those?
#if defined(SHADER_CPP)
	packed_u16vec2 uv;
#else
	packed_f16vec2 uv;
#endif
};

#if defined(SHADER_GLSL)
vec4 unpackVertexColor(in u8vec4 color) {
	return vec4(color) / 255.f;
}

vec3 unpackVertexNormal(in u8vec3 normal) {
	return normalize(vec3(normal) / 127.f - 1.f);
}

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer VertexIndices {
	uint vertexIndices[];
};
layout(buffer_reference, scalar, buffer_reference_align = 1) restrict readonly buffer PrimitiveIndices {
	uint8_t primitiveIndices[];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer Vertices {
	Vertex vertices[];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer Meshlets {
	Meshlet meshlets[];
};
#endif

struct MeshletDraw {
	uint32_t primitiveIndex;
	uint32_t meshletIndex;
	uint32_t transformIndex;
};

struct Primitive {
	BUFFER_REF(VertexIndices, uint32_t) vertexIndexBuffer MEMBER_INIT(0);
	BUFFER_REF(PrimitiveIndices, uint8_t) primitiveIndexBuffer MEMBER_INIT(0);
	BUFFER_REF(Vertices, Vertex) vertexBuffer MEMBER_INIT(0);
	BUFFER_REF(Meshlets, Meshlet) meshletBuffer MEMBER_INIT(0);

	packed_fvec3 aabbExtents;
	packed_fvec3 aabbCenter;

	uint32_t meshletCount;
	uint32_t materialIndex;
};

struct Material {
	packed_fvec4 albedoFactor;

	// Albedo texture
	ResourceTableHandle albedoIndex;
	packed_fvec2 uvOffset;
	packed_fvec2 uvScale;
	float uvRotation;

	float alphaCutoff;
	SHADER_BOOL doubleSided;
};

#if defined(SHADER_GLSL)
vec2 transformUv(in Material material, vec2 uv) {
	mat2 rotationMat = mat2(
		cos(material.uvRotation), -sin(material.uvRotation),
		sin(material.uvRotation), cos(material.uvRotation)
	);
	return rotationMat * uv * material.uvScale + material.uvOffset;
}
#endif

SHADER_NAMESPACE_END
#endif
