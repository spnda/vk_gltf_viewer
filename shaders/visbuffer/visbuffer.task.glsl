#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "visbuffer.glsl.h"
#include "mesh_common.glsl.h"
#include "frustum_culling.glsl.h"

layout(constant_id = 0) const uint subgroupSize = 32;

// This now uses the spec constant 0 for the x size
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant, scalar) readonly uniform PushConstants {
	VisbufferPushConstants pushConstants;
};

taskPayloadSharedEXT TaskPayload taskPayload;

void main() {
	// Every task shader workgroup only gets 128 meshlets to handle. This calculates how many
	// this specific work group should handle, and sets the baseID accordingly.
	uint meshletCount = min(maxMeshlets, pushConstants.meshletDrawCount - (gl_WorkGroupID.x * maxMeshlets));
	uint baseId = taskPayload.baseID = gl_WorkGroupID.x * maxMeshlets;

	// We have workgroup sizes of 32 or whatever the subgroup size is, but we designate one task
	// shader workgroup for maxMeshlets meshlets, to fit efficiently into the task payload.
	const uint meshletLoops = (meshletCount + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
	uint visibleMeshlets = 0;
	[[unroll]] for (uint i = 0; i < meshletLoops; ++i) {
		uint idx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;
		idx = min(idx, meshletCount - 1);

		// TODO: Is this a great option that we fetch the data like this in the loop?
		restrict const MeshletDraw draw = pushConstants.drawBuffer.draws[baseId + idx];
		const mat4 transformMatrix = pushConstants.transformBuffer.transforms[draw.transformIndex];
		restrict Primitive primitive = pushConstants.primitiveBuffer.primitives[draw.primitiveIndex];
		restrict const Meshlet meshlet = primitive.meshletBuffer.meshlets[draw.meshletIndex];

		// Do some culling
		const vec3 worldAabbCenter = (transformMatrix * vec4(meshlet.aabbCenter, 1.0f)).xyz;
		const vec3 worldAabbExtent = getWorldSpaceAabbExtent(meshlet.aabbExtents.xyz, transformMatrix);
		const bool visible = isAabbInFrustum(worldAabbCenter, worldAabbExtent, pushConstants.cameraBuffer.camera.frustum);

		// Get the index for this thread for this subgroup
		uint payloadIndex = subgroupExclusiveAdd(uint(visible));
		if (visible) {
			taskPayload.deltaIDs[visibleMeshlets + payloadIndex] = uint8_t(idx);
		}
		visibleMeshlets += subgroupAdd(uint(visible));
	}

	EmitMeshTasksEXT(visibleMeshlets, 1, 1);
}
