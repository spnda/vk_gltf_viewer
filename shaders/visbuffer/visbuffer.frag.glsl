#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_mesh_shader : require // For gl_PrimitiveID

#include "mesh_common.h"
#include "srgb.h"

#include "visbuffer.h"

#include "resource_table.h"

layout(location = 0) in flat uint drawIndex;
layout(location = 1) in flat uint materialIndex;
layout(location = 2) in vec4 position;
layout(location = 3) in vec4 prevPosition;
layout(location = 4) in vec4 color;
layout(location = 5) in vec2 uv;

layout(location = 0) out uint visbufferId;
layout(location = 1) out vec2 motionVectors;

layout(push_constant, scalar) readonly uniform PushConstants {
	VisbufferPushConstants pushConstants;
};

void main() {
	//restrict const Material material = pushConstants.materialBuffer.materials[materialIndex];

	//const vec4 sampled = texture(textures[material.albedoIndex], transformUv(material, uv));
	//const vec4 albedoColor = color * material.albedoFactor * toLinear(sampled);
	//if (albedoColor.a < material.alphaCutoff)
	//    discard;

	visbufferId = packVisBuffer(drawIndex, uint(gl_PrimitiveID));

	motionVectors = ((prevPosition.xy / prevPosition.w) * 0.5f - 0.5f) - ((position.xy / position.w) * 0.5f - 0.5f);
}
