#include <metal_stdlib>

#include "visbuffer.h"
#include "mesh_common.h"
#include "culling.h"

using namespace metal;

struct MeshletVertex {
	float4 position [[position]];
};

struct MeshletPrimitive {
	uint drawIndex;
	bool culled [[primitive_culled]];
};

using Meshlet = mesh<MeshletVertex, MeshletPrimitive, shaders::maxVertices, shaders::maxPrimitives, topology::triangle>;

struct ObjectPayload {
	uint baseIndex;
	metal::array<uint8_t, shaders::maxMeshlets> indices;
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

/// Object shader that handles up to shaders::maxMeshlets meshlets per threadgroup.
/// This uses frustum culling and a modified index array to have a fully GPU-driven
/// pipeline and cull as many meshlets as possible.
[[object]] void visbuffer_object(
		object_data ObjectPayload& payload [[payload]],
		mesh_grid_properties outGrid,
		constant const ulong& meshletDrawCount [[buffer(0)]],
		device const shaders::Camera& camera [[buffer(1)]],
		device const shaders::MeshletDraw* draws [[buffer(2)]],
		device const float4x4* transforms [[buffer(3)]],
		device const shaders::Primitive* primitives [[buffer(4)]],
		uint groupId [[threadgroup_position_in_grid]],
		uint threadIndex [[thread_position_in_threadgroup]],
		uint threadGroupWidth [[threads_per_threadgroup]]) {
	uint meshletCount = uint(min(ulong(shaders::maxMeshlets), meshletDrawCount - (groupId * shaders::maxMeshlets)));
	uint baseId = payload.baseIndex = groupId * shaders::maxMeshlets;

	// We use an atomic counter for how many meshlets are visible instead of SIMD intrinsics,
	// since those caused a GPU hang in some cases, for some reason. Might be useful to investigate
	// that in the future, since I doubt this atomic is absolutely free.
	threadgroup atomic<uint> visibleMeshlets;

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
		const auto worldAabbExtent = shaders::getWorldSpaceAabbExtent(meshlet.aabbExtents.xyz, transformMatrix);
		visible = visible && shaders::isAabbInFrustum(worldAabbCenter, worldAabbExtent, camera.frustum);

		if (visible) {
			payload.indices[atomic_fetch_add_explicit(&visibleMeshlets, 1, memory_order_relaxed)] = uint8_t(idx);
		}
	}

	threadgroup_barrier(mem_flags::mem_none);
	if (threadIndex == 0) {
		outGrid.set_threadgroups_per_grid(uint3(atomic_load_explicit(&visibleMeshlets, memory_order_relaxed), 1, 1));
	}
}

/// Mesh shader that handles a single meshlet/draw per threadgroup.
/// Since a meshlet can have a variable amount of primitives and vertices,
/// we use fancy loops to properly process them, at the cost of potentially
/// processing the last element multiple times.
[[mesh]] void visbuffer_mesh(
		Meshlet meshletOut,
		object_data const ObjectPayload& payload [[payload]],
		device const shaders::MeshletDraw* draws [[buffer(0)]],
		device const float4x4* transforms [[buffer(1)]],
		device const shaders::Primitive* primitives [[buffer(2)]],
		device const shaders::Camera& camera [[buffer(3)]],
		device const shaders::Material* materials [[buffer(4)]],
		uint payloadIndex [[threadgroup_position_in_grid]],
		uint threadIndex [[thread_position_in_threadgroup]],
		uint threadGroupWidth [[threads_per_threadgroup]]) {
	const auto drawIdx = payload.baseIndex + payload.indices[payloadIndex];

	device auto& draw = draws[drawIdx];

	device auto& primitive = primitives[draw.primitiveIndex];
	device auto& meshlet = primitive.meshletBuffer[draw.meshletIndex];
	device auto& material = materials[primitive.materialIndex];

	if (threadIndex == 0) {
		meshletOut.set_primitive_count(meshlet.triangleCount);
	}

	device auto& transformMatrix = transforms[draw.transformIndex];
	auto mvp = camera.viewProjection * transformMatrix;

	threadgroup metal::array<float3, shaders::maxVertices> clipVertices;

	const uint vertexLoops = (meshlet.vertexCount + threadGroupWidth - 1) / threadGroupWidth;
	for (uint i = 0; i < vertexLoops; ++i) {
		uint vidx = threadIndex + i * threadGroupWidth;
		vidx = min(vidx, meshlet.vertexCount - 1U);

		const auto vertexIndex = primitive.vertexIndexBuffer[meshlet.vertexOffset + vidx];
		device const auto& vtx = primitive.vertexBuffer[vertexIndex];

		auto pos = mvp * float4(vtx.position, 1.f);
		clipVertices[vidx] = pos.xyw;
		meshletOut.set_vertex(vidx, MeshletVertex {
			.position = pos,
		});
	}

	const auto transformDeterminant = determinant(transformMatrix);
	const uint primitiveLoops = (meshlet.triangleCount + threadGroupWidth - 1) / threadGroupWidth;
	for (uint i = 0; i < primitiveLoops; ++i) {
		uint pidx = threadIndex + i * threadGroupWidth;
		pidx = min(pidx, meshlet.triangleCount - 1U);

		auto j = pidx * 3;
		auto idx0 = primitive.primitiveIndexBuffer[meshlet.triangleOffset + j + 0];
		auto idx1 = primitive.primitiveIndexBuffer[meshlet.triangleOffset + j + 1];
		auto idx2 = primitive.primitiveIndexBuffer[meshlet.triangleOffset + j + 2];

		meshletOut.set_index(j + 0, idx0);
		meshletOut.set_index(j + 1, idx1);
		meshletOut.set_index(j + 2, idx2);

		if (!material.doubleSided) {
			const auto v0 = clipVertices[idx0];
			const auto v1 = clipVertices[idx1];
			const auto v2 = clipVertices[idx2];
			const auto det = determinant(float3x3(v0, v1, v2));

			bool culled = transformDeterminant < 0.0f
				? det > 0.0f // Front face culling with Y+ as up.
				: det < 0.0f; // Back face culling with Y+ as up.

			meshletOut.set_primitive(pidx, MeshletPrimitive {
				.drawIndex = drawIdx,
				.culled = culled
			});
		} else {
			meshletOut.set_primitive(pidx, MeshletPrimitive {
				.drawIndex = drawIdx,
				.culled = false, // Material is double sided, meaning we can't cull.
			});
		}
	}
}

struct FragmentIn {
	MeshletVertex vert;
	MeshletPrimitive prim;
};

[[fragment]] uint visbuffer_frag(
		FragmentIn in [[stage_in]],
		uint primitiveId [[primitive_id]]) {
	return shaders::packVisBuffer(in.prim.drawIndex, primitiveId);
}

[[kernel]] void visbuffer_resolve(
		device const shaders::MeshletDraw* draws [[buffer(0)]],
		device const shaders::Primitive* primitives [[buffer(1)]],
		device const shaders::Material* materials [[buffer(2)]],
		texture2d<uint, access::read> visbuffer [[texture(0)]],
		texture2d<float, access::write> color [[texture(1)]],
		ushort2 gid [[thread_position_in_grid]]) {
	if (gid.x >= visbuffer.get_width() || gid.y >= visbuffer.get_height()) {
		return;
	}

	auto data = visbuffer.read(gid).r;
	if (data == shaders::visbufferClearValue) {
		color.write(float4(0.f), gid);
		return;
	}

	auto [drawIndex, primitiveId] = shaders::unpackVisBuffer(data);

	device const auto& draw = draws[drawIndex];
	device auto& primitive = primitives[draw.primitiveIndex];
	device auto& material = materials[primitive.materialIndex];

	auto resolved = float4(hue2rgb(draw.meshletIndex * 1.71f), 1.f);
	color.write(resolved, gid);
}

// Vertices of a basic cube
constant const auto positions = metal::array<float3, 8> {
	float3(1, -1, -1),
	float3(1, 1, -1),
	float3(-1, 1, -1),
	float3(-1, -1, -1),
	float3(1, -1, 1),
	float3(1, 1, 1),
	float3(-1, -1, 1),
	float3(-1, 1, 1)
};

// Edge indices for a basic cube
constant const auto edges = metal::array<uint, 12*2> {
	0, 1,
	0, 3,
	0, 4,
	2, 1,
	2, 3,
	2, 7,
	6, 3,
	6, 4,
	6, 7,
	5, 1,
	5, 4,
	5, 7
};

struct MeshletAabbOutput {
	float4 pos [[position]];
};

/// This vertex shader generates a cube for each meshlet AABB, which we use
/// together with the visibility result buffer to perform occlusion culling.
[[vertex]] MeshletAabbOutput meshlet_aabb_vert(
		device const shaders::MeshletDraw* draws [[buffer(0)]],
		device const float4x4* transforms [[buffer(1)]],
		device const shaders::Primitive* primitives [[buffer(2)]],
		device const shaders::Camera& camera [[buffer(3)]],
		uint instanceId [[instance_id]],
		uint vertexId [[vertex_id]]) {

	device const auto& draw = draws[instanceId];
	device const auto& transformMatrix = transforms[draw.transformIndex];
	device const auto& primitive = primitives[draw.primitiveIndex];
	device const auto& meshlet = primitive.meshletBuffer[draw.meshletIndex];

	auto position = positions[edges[vertexId]];
	auto pos = position * meshlet.aabbExtents + meshlet.aabbCenter;

	return {
		.pos = camera.viewProjection * transformMatrix * float4(pos, 1.f),
	};
}
