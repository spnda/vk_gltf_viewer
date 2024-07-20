#ifndef VISBUFFER_GLSL_H
#define VISBUFFER_GLSL_H

#if !defined(__cplusplus)
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#endif

#include "common.h.glsl"
#include "mesh_common.h.glsl"
#include "resource_table.h.glsl"

GLSL_NAMESPACE_BEGIN

GLSL_CONSTANT uint triangleBits = 7; // Enough to fit 127 unique triangles
GLSL_CONSTANT uint drawIndexBits = 32 - triangleBits; // 25 bits for the drawIndex, which limits us
													  // to 33'554'432 meshlets for rendering.

#if !defined(__cplusplus)
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer Materials {
	Material materials[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer MeshletDraws {
	MeshletDraw draws[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer TransformBuffer {
	mat4 transforms[];
};

layout(buffer_reference, scalar, buffer_reference_align = 8) restrict readonly buffer Primitives {
	Primitive primitives[];
};
#endif

struct VisbufferPushConstants {
	BUFFER_REF(MeshletDraws, MeshletDraw) drawBuffer MEMBER_INIT(0);
	uint meshletDrawCount MEMBER_INIT(0);

	BUFFER_REF(TransformBuffer, mat4) transformBuffer MEMBER_INIT(0);
	BUFFER_REF(Primitives, Primitive) primitiveBuffer MEMBER_INIT(0);
	BUFFER_REF(CameraBuffer, Camera) cameraBuffer MEMBER_INIT(0);
	BUFFER_REF(Materials, Material) materialBuffer MEMBER_INIT(0);

	ResourceTableHandle depthPyramid;
};

struct VisbufferResolvePushConstants {
	ResourceTableHandle visbufferHandle;
	ResourceTableHandle outputImageHandle;

	BUFFER_REF(MeshletDraws, MeshletDraw) drawBuffer MEMBER_INIT(0);
	BUFFER_REF(Primitives, Primitive) primitiveBuffer MEMBER_INIT(0);
	BUFFER_REF(Materials, Material) materialBuffer MEMBER_INIT(0);
};

uint packVisBuffer(PARAMETER_COPY(uint) drawIndex, PARAMETER_COPY(uint) primitiveId) {
	return (drawIndex << triangleBits) | primitiveId;
}

void unpackVisBuffer(PARAMETER_COPY(uint) visBuffer, PARAMETER_REF(uint) drawIndex, PARAMETER_REF(uint) primitiveId) {
	primitiveId = visBuffer & ((1 << triangleBits) - 1);
	drawIndex = visBuffer >> triangleBits;
}

GLSL_CONSTANT uint visbufferClearValue = ~0U;

GLSL_NAMESPACE_END
#endif
