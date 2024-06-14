#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_control_flow_attributes : require

#include "visbuffer.glsl.h"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(triangles, max_vertices = maxVertices, max_primitives = maxPrimitives) out;

// This is not per primitive, but this is a little more memory efficient,
// since there is no perdrawEXT.
layout(location = 0) perprimitiveEXT out flat uint drawIndex[];
layout(location = 1) perprimitiveEXT out flat uint materialIndex[];
layout(location = 2) out vec4 position[];
layout(location = 3) out vec4 prevPosition[];
layout(location = 4) out vec4 color[];
layout(location = 5) out vec2 uv[];

layout(push_constant, scalar) readonly uniform PushConstants {
	VisbufferPushConstants pushConstants;
};

taskPayloadSharedEXT TaskPayload taskPayload;

void main() {
	const uint drawId = taskPayload.baseID + taskPayload.deltaIDs[gl_WorkGroupID.x];
	restrict const MeshletDraw draw = pushConstants.drawBuffer.draws[drawId];

	restrict Primitive primitive = pushConstants.primitiveBuffer.primitives[draw.primitiveIndex];
	restrict const Meshlet meshlet = primitive.meshletBuffer.meshlets[draw.meshletIndex];

	// This defines the array size of gl_MeshVerticesEXT
	if (gl_LocalInvocationID.x == 0) {
		SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);
	}

	mat4 transformMatrix = pushConstants.transformBuffer.transforms[draw.transformIndex];
	mat4 mvp = pushConstants.cameraBuffer.camera.viewProjection * transformMatrix;
	mat4 prevMvp = pushConstants.cameraBuffer.camera.prevViewProjection * transformMatrix; // TODO: We need the transforms from the last frame

	// The max_vertices does not match the local workgroup size.
	// Therefore, we'll have this loop that will run over all possible vertices.
	const uint vertexLoops = (meshlet.vertexCount + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
	[[unroll]] for (uint i = 0; i < vertexLoops; ++i) {
		// Distribute each vertex of the loop over the workgroup
		uint vidx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;

		// Avoid branching but also respect the meshlet vertexCount.
		// This will, however, re-compute some vertices.
		vidx = min(vidx, meshlet.vertexCount - 1);

		const uint vertexIndex = primitive.vertexIndexBuffer.vertexIndices[meshlet.vertexOffset + vidx];
		restrict const Vertex vertex = primitive.vertexBuffer.vertices[vertexIndex];

		vec4 pos = mvp * vec4(vertex.position, 1.f);
		gl_MeshVerticesEXT[vidx].gl_Position = pos;
		position[vidx] = pos;
		prevPosition[vidx] = prevMvp * vec4(vertex.position, 1.f);
		color[vidx] = unpackVertexColor(vertex.color);
		uv[vidx] = vec2(vertex.uv);
	}

	const uint primitiveLoops = (meshlet.triangleCount + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
	[[unroll]] for (uint i = 0; i < primitiveLoops; ++i) {
		uint pidx = gl_LocalInvocationIndex.x + i * gl_WorkGroupSize.x;
		pidx = min(pidx, meshlet.triangleCount - 1);

		uvec3 indices = uvec3(
			primitive.primitiveIndexBuffer.primitiveIndices[meshlet.triangleOffset + pidx * 3 + 0],
			primitive.primitiveIndexBuffer.primitiveIndices[meshlet.triangleOffset + pidx * 3 + 1],
			primitive.primitiveIndexBuffer.primitiveIndices[meshlet.triangleOffset + pidx * 3 + 2]);

		gl_PrimitiveTriangleIndicesEXT[pidx] = indices;
		drawIndex[pidx] = drawId;
		materialIndex[pidx] = primitive.materialIndex;
	}
}
