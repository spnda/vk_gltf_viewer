#include <metal_stdlib>

#include "visbuffer.h.glsl"
#include "mesh_common.h.glsl"
#include "culling.h.glsl"

using namespace metal;

struct MeshletVertex {
	float4 position [[position]];
};

struct MeshletPrimitive {
	uint drawIndex;
	bool culled [[primitive_culled]];
	float4 color [[flat]];
};

using Meshlet = mesh<MeshletVertex, MeshletPrimitive, glsl::maxVertices, glsl::maxPrimitives, topology::triangle>;

struct ObjectPayload {
	uint baseIndex;
	metal::array<uint8_t, glsl::maxMeshlets> indices;
};

// https://www.ronja-tutorials.com/post/041-hsv-colorspace/
static float3 hue2rgb(float hue) {
	hue = fract(hue); //only use fractional part of hue, making it loop
	float r = abs(hue * 6 - 3) - 1; //red
	float g = 2 - abs(hue * 6 - 2); //green
	float b = 2 - abs(hue * 6 - 4); //blue
	float3 rgb = float3(r,g,b); //combine components
	rgb = saturate(rgb); //clamp between 0 and 1
	return rgb;
}

[[object]] void visbuffer_object(
		object_data ObjectPayload& payload [[payload]],
		mesh_grid_properties outGrid,
		constant const ulong& meshletDrawCount [[buffer(0)]],
		device const glsl::Camera& camera [[buffer(1)]],
		device const glsl::MeshletDraw* draws [[buffer(2)]],
		device const float4x4* transforms [[buffer(3)]],
		device const glsl::Primitive* primitives [[buffer(4)]],
		uint groupId [[threadgroup_position_in_grid]],
		uint threadIndex [[thread_position_in_threadgroup]],
		uint threadGroupWidth [[threads_per_threadgroup]]) {
	uint meshletCount = uint(min(ulong(glsl::maxMeshlets), meshletDrawCount - (groupId * glsl::maxMeshlets)));
	uint baseId = payload.baseIndex = groupId * glsl::maxMeshlets;

	uint visibleMeshlets = 0;

	const uint meshletLoops = (meshletCount + threadGroupWidth - 1) / threadGroupWidth;
	for (uint i = 0; i < meshletLoops; ++i) {
		uint tidx = threadIndex + i * threadGroupWidth;
		uint idx = min(tidx, meshletCount - 1U);

		device const auto& draw = draws[baseId + idx];
		device const auto& transformMatrix = transforms[draw.transformIndex];
		device const auto& primitive = primitives[draw.primitiveIndex];
		device const auto& meshlet = primitive.meshletBuffer[draw.meshletIndex];

		// We automatically mark the meshlet as culled if we are a thread that is overprocessing,
		// this prevents any buffer overruns in the payload index array later.
		bool visible = tidx == idx;

		// Frustum culling
		const auto worldAabbCenter = (transformMatrix * float4(meshlet.aabbCenter, 1.0f)).xyz;
		const auto worldAabbExtent = glsl::getWorldSpaceAabbExtent(meshlet.aabbExtents.xyz, transformMatrix);
		visible = visible && glsl::isAabbInFrustum(worldAabbCenter, worldAabbExtent, camera.frustum);

		auto payloadIndex = simd_prefix_exclusive_sum(uint(visible));
		if (visible) {
			payload.indices[visibleMeshlets + payloadIndex] = uint8_t(idx);
		}
		visibleMeshlets += simd_sum(uint(visible));
	}

	threadgroup_barrier(mem_flags::mem_none);
	if (threadIndex == 0) {
		outGrid.set_threadgroups_per_grid(uint3(visibleMeshlets, 1, 1));
	}
}

[[mesh]] void visbuffer_mesh(
		Meshlet meshletOut,
		object_data const ObjectPayload& payload [[payload]],
		device const glsl::MeshletDraw* draws [[buffer(0)]],
		device const float4x4* transforms [[buffer(1)]],
		device const glsl::Primitive* primitives [[buffer(2)]],
		device const glsl::Camera& camera [[buffer(3)]],
		uint payloadIndex [[threadgroup_position_in_grid]],
		uint threadIndex [[thread_position_in_threadgroup]],
		uint threadGroupWidth [[threads_per_threadgroup]]) {
	const auto drawIdx = payload.baseIndex + payload.indices[payloadIndex];

	device auto& draw = draws[drawIdx];

	device auto& primitive = primitives[draw.primitiveIndex];
	device auto& meshlet = primitive.meshletBuffer[draw.meshletIndex];

	if (threadIndex == 0) {
		meshletOut.set_primitive_count(meshlet.triangleCount);
	}

	device auto& transformMatrix = transforms[draw.transformIndex];
	auto mvp = camera.viewProjection * transformMatrix;

	const uint vertexLoops = (meshlet.vertexCount + threadGroupWidth - 1) / threadGroupWidth;
	for (uint i = 0; i < vertexLoops; ++i) {
		uint vidx = threadIndex + i * threadGroupWidth;
		vidx = min(vidx, meshlet.vertexCount - 1U);

		const auto vertexIndex = primitive.vertexIndexBuffer[meshlet.vertexOffset + vidx];
		device const auto& vtx = primitive.vertexBuffer[vertexIndex];

		auto pos = mvp * float4(vtx.position, 1.f);
		meshletOut.set_vertex(vidx, MeshletVertex {
			.position = pos,
		});
	}

	const uint primitiveLoops = (meshlet.triangleCount + threadGroupWidth - 1) / threadGroupWidth;
	for (uint i = 0; i < primitiveLoops; ++i) {
		uint pidx = threadIndex + i * threadGroupWidth;
		pidx = min(pidx, meshlet.triangleCount - 1U);

		auto j = pidx * 3;
		meshletOut.set_index(j + 0, primitive.primitiveIndexBuffer[meshlet.triangleOffset + j + 0]);
		meshletOut.set_index(j + 1, primitive.primitiveIndexBuffer[meshlet.triangleOffset + j + 1]);
		meshletOut.set_index(j + 2, primitive.primitiveIndexBuffer[meshlet.triangleOffset + j + 2]);

		meshletOut.set_primitive(pidx, MeshletPrimitive {
			.drawIndex = drawIdx,
			.culled = false,
			.color = float4(hue2rgb(draw.meshletIndex * 1.71f), 1),
		});
	}
}

struct FragmentIn {
	MeshletVertex vert;
	MeshletPrimitive prim;
};

struct FragmentOut {
	uint visbuffer [[color(0)]];
	float4 color [[color(1)]];
};

/*[[fragment]] uint visbuffer_frag(
		FragmentIn in [[stage_in]],
		uint primitiveId [[primitive_id]]) {
	return glsl::packVisBuffer(in.prim.drawIndex, primitiveId);
}*/

[[fragment]] FragmentOut visbuffer_frag(
		FragmentIn in [[stage_in]],
		uint primitiveId [[primitive_id]]) {
	return {
		.visbuffer = glsl::packVisBuffer(in.prim.drawIndex, primitiveId),
		.color = in.prim.color,
	};
}
