#ifndef MESH_COMMON_GLSL_H
#define MESH_COMMON_GLSL_H

#if !defined(__cplusplus)
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#include "common.h.glsl"
#include "resource_table.h.glsl"
GLSL_NAMESPACE_BEGIN

GLSL_CONSTANT uint shadowMapCount = 4;

struct Camera {
	ALIGN_AS(16) mat4 prevViewProjection;
	ALIGN_AS(16) mat4 prevOcclusionViewProjection;

	ALIGN_AS(16) mat4 viewProjection;
	ALIGN_AS(16) mat4 occlusionViewProjection;
	GLSL_ARRAY(vec4, frustum, 6);
};

#if !defined(__cplusplus)
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer CameraBuffer {
	Camera camera;
};
#endif

// TODO: These are the optimal values for NVIDIA. What about the others?
#if defined(SHADER_GLSL)
GLSL_CONSTANT uint maxVertices = 64;
GLSL_CONSTANT uint maxPrimitives = 126;
GLSL_CONSTANT uint maxMeshlets = 102;
#else
GLSL_CONSTANT uint maxVertices = 64;
GLSL_CONSTANT uint maxPrimitives = 128;
GLSL_CONSTANT uint maxMeshlets = 64; // This should best be a multiple of the threadgroup size.
#endif

// This is essentially a replacement for gl_WorkGroupID.x, but one which can store any index
// between 0..256 instead of the linear requirement of the work group ID.
// For NVIDIA, we try to keep this structure below 108 bytes to keep it in shared memory.
// Therefore, this is exactly 106 bytes big (4 + 102 * 1).
struct TaskPayload {
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
	uint primitiveIndex;
	uint meshletIndex;
	uint transformIndex;
};

struct Primitive {
	BUFFER_REF(VertexIndices, uint) vertexIndexBuffer MEMBER_INIT(0);
	BUFFER_REF(PrimitiveIndices, uint8_t) primitiveIndexBuffer MEMBER_INIT(0);
	BUFFER_REF(Vertices, Vertex) vertexBuffer MEMBER_INIT(0);
	BUFFER_REF(Meshlets, Meshlet) meshletBuffer MEMBER_INIT(0);

	vec3 aabbExtents;
	vec3 aabbCenter;

	uint meshletCount;
	uint materialIndex;
};

struct Material {
	vec4 albedoFactor;

	// Albedo texture
	ResourceTableHandle albedoIndex;
	vec2 uvOffset;
	vec2 uvScale;
	float uvRotation;

	float alphaCutoff;
	GLSL_BOOL doubleSided;
};

#if !defined(__cplusplus)
vec2 transformUv(in Material material, vec2 uv) {
	mat2 rotationMat = mat2(
		cos(material.uvRotation), -sin(material.uvRotation),
		sin(material.uvRotation), cos(material.uvRotation)
	);
	return rotationMat * uv * material.uvScale + material.uvOffset;
}
#endif

GLSL_NAMESPACE_END
#endif
