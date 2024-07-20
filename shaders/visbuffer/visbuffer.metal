#include <metal_stdlib>

#include "visbuffer.h.glsl"
#include "mesh_common.h.glsl"

using namespace metal;

struct MeshletVertex {
	float4 position [[position]];
};

struct MeshletPrimitive {
	uint drawIndex;
	float4 color [[flat]];
};

using Meshlet = mesh<MeshletVertex, MeshletPrimitive, glsl::maxVertices, glsl::maxPrimitives, topology::triangle>;

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

[[object]] void visbuffer_object(object_data glsl::TaskPayload& out [[payload]]) {

}

[[mesh]] void visbuffer_mesh(
		Meshlet meshletOut,
		device const glsl::MeshletDraw* draws [[buffer(0)]],
		device const float4x4* transforms [[buffer(1)]],
		device const glsl::Primitive* primitives [[buffer(2)]],
		device const glsl::Camera& camera [[buffer(3)]],
		uint drawIdx [[threadgroup_position_in_grid]],
		uint threadIndex [[thread_position_in_threadgroup]],
		uint threadGroupWidth [[threads_per_threadgroup]]) {
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
