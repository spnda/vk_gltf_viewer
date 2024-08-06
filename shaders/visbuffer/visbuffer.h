#ifndef SHADERS_VISBUFFER_H
#define SHADERS_VISBUFFER_H

#if !defined(__cplusplus)
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#endif

#include "common.h"
#include "mesh_common.h"
#include "resource_table.h"

SHADER_NAMESPACE_BEGIN

SHADER_CONSTANT uint32_t triangleBits = 7; // Enough to fit 127 unique triangles
SHADER_CONSTANT uint32_t drawIndexBits = 32 - triangleBits; // 25 bits for the drawIndex, which limits us
													  // to 33'554'432 meshlets for rendering.
SHADER_CONSTANT uint32_t visbufferClearValue = ~0U;

#if defined(SHADER_GLSL)
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer Materials {
	Material materials[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer MeshletDraws {
	MeshletDraw draws[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer TransformBuffer {
	fmat4 transforms[];
};

layout(buffer_reference, scalar, buffer_reference_align = 8) restrict readonly buffer Primitives {
	Primitive primitives[];
};
#endif

struct VisbufferPushConstants {
	BUFFER_REF(MeshletDraws, MeshletDraw) drawBuffer MEMBER_INIT(0);
	uint32_t meshletDrawCount MEMBER_INIT(0);

	BUFFER_REF(TransformBuffer, fmat4) transformBuffer MEMBER_INIT(0);
	BUFFER_REF(Primitives, Primitive) primitiveBuffer MEMBER_INIT(0);
	BUFFER_REF(CameraBuffer, Camera) cameraBuffer MEMBER_INIT(0);
	BUFFER_REF(Materials, Material) materialBuffer MEMBER_INIT(0);

	ResourceTableHandle depthPyramid MEMBER_INIT(invalidHandle);
};

struct VisbufferResolvePushConstants {
	ResourceTableHandle visbufferHandle MEMBER_INIT(invalidHandle);
	ResourceTableHandle outputImageHandle MEMBER_INIT(invalidHandle);

	BUFFER_REF(MeshletDraws, MeshletDraw) drawBuffer MEMBER_INIT(0);
	BUFFER_REF(Primitives, Primitive) primitiveBuffer MEMBER_INIT(0);
	BUFFER_REF(Materials, Material) materialBuffer MEMBER_INIT(0);
};

FUNCTION_INLINE uint32_t packVisBuffer(uint32_t drawIndex, uint32_t primitiveId) {
	return (drawIndex << triangleBits) | primitiveId;
}

#if !defined(SHADER_GLSL)
struct VisbufferData {
	uint32_t drawIndex;
	uint32_t primitiveId;
};
FUNCTION_INLINE VisbufferData unpackVisBuffer(uint32_t visbuffer) {
	return {
		.drawIndex = visbuffer >> triangleBits,
		.primitiveId = visbuffer & ((1 << triangleBits) - 1),
	};
}
#else
void unpackVisBuffer(uint32_t visBuffer, out uint32_t drawIndex, out uint32_t primitiveId) {
	primitiveId = visBuffer & ((1 << triangleBits) - 1);
	drawIndex = visBuffer >> triangleBits;
}
#endif

SHADER_NAMESPACE_END
#endif
