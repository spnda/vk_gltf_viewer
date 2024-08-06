#include <metal_stdlib>

#include "mesh_common.h"
#include "visbuffer.h"

using namespace metal;

// https://www.ronja-tutorials.com/post/041-hsv-colorspace/
FUNCTION_INLINE float3 hue2rgb(float hue) {
	hue = fract(hue); //only use fractional part of hue, making it loop
	float r = abs(hue * 6 - 3) - 1; //red
	float g = 2 - abs(hue * 6 - 2); //green
	float b = 2 - abs(hue * 6 - 4); //blue
	auto rgb = float3(r,g,b); //combine components
	rgb = saturate(rgb); //clamp between 0 and 1
	return rgb;
}

uint3 getVertexIndices(
		device const shaders::Primitive& primitive,
		device const shaders::Meshlet& meshlet,
		uint32_t primitiveId) {
	uchar3 primitiveIndices(
		primitive.primitiveIndexBuffer[meshlet.triangleOffset + primitiveId * 3 + 0],
		primitive.primitiveIndexBuffer[meshlet.triangleOffset + primitiveId * 3 + 1],
		primitive.primitiveIndexBuffer[meshlet.triangleOffset + primitiveId * 3 + 2]
	);

	return uint3(
		primitive.vertexIndexBuffer[meshlet.vertexOffset + primitiveIndices.x],
		primitive.vertexIndexBuffer[meshlet.vertexOffset + primitiveIndices.y],
		primitive.vertexIndexBuffer[meshlet.vertexOffset + primitiveIndices.z]
	);
}

struct Barycentrics {
	float3 lambda;
	float3 ddx;
	float3 ddy;
};

/// See http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
/// for more details on how this works and the links to the relevant papers.
Barycentrics calculateBarycentrics(float4 v0, float4 v1, float4 v2, float2 pixel, float2 size) {
	pixel.y = -pixel.y; // flip because +Y is up.
	
	auto invW = 1.f / float3(v0.w, v1.w, v2.w);

	auto ndc0 = v0.xy * invW.x;
	auto ndc1 = v1.xy * invW.y;
	auto ndc2 = v2.xy * invW.z;

	auto invDet = 1.f / determinant(float2x2(ndc2 - ndc1, ndc0 - ndc1));
	auto ddx = float3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
	auto ddy = float3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;
	auto ddxSum = dot(ddx, float3(1.f));
	auto ddySum = dot(ddy, float3(1.f));

	auto deltaVec = pixel - ndc0;
	auto interpInvW = invW.x + deltaVec.x * ddxSum + deltaVec.y * ddySum;
	auto interpW = 1.f / interpInvW;

	float3 lambda(
		interpW * (invW.x + deltaVec.x * ddx.x + deltaVec.y * ddy.x),
		interpW * (0.0f   + deltaVec.x * ddx.y + deltaVec.y * ddy.y),
		interpW * (0.0f   + deltaVec.x * ddx.z + deltaVec.y * ddy.z)
	);

	ddx *= 2.f / size.x;
	ddy *= 2.f / size.y;
	ddxSum *= 2.f / size.x;
	ddySum *= 2.f / size.y;

	//ddy *= -1.f;
	//ddySum *= -1.f;

	auto interpW_ddx = 1.f / (interpInvW + ddxSum);
	auto interpW_ddy = 1.f / (interpInvW + ddySum);

	return {
		.lambda = lambda,
		.ddx = interpW_ddx * (lambda * interpInvW + ddx) - lambda,
		.ddy = interpW_ddy * (lambda * interpInvW + ddy) - lambda,
	};
}

template <typename T, size_t N>
vec<T, N> interpolate(thread const Barycentrics& barycentrics, vec<T, N> v0, vec<T, N> v1, vec<T, N> v2) {
	return barycentrics.lambda.x * v0
		+ barycentrics.lambda.y * v1
		+ barycentrics.lambda.z * v2;
}

float3 interpolateWithDeriv(thread const Barycentrics& barycentrics, float3 v) {
	return float3(
		dot(v, barycentrics.lambda),
		dot(v, barycentrics.ddx),
		dot(v, barycentrics.ddy));
}

[[kernel]] void visbuffer_resolve(
		device const shaders::MeshletDraw* draws [[buffer(0)]],
		device const float4x4* transforms [[buffer(1)]],
		device const shaders::Primitive* primitives [[buffer(2)]],
		device const shaders::Camera& camera [[buffer(3)]],
		device const shaders::Material* materials [[buffer(4)]],
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
	device const auto& transformMatrix = transforms[draw.transformIndex];
	device const auto& primitive = primitives[draw.primitiveIndex];
	device const auto& material = materials[primitive.materialIndex];

	device const auto& meshlet = primitive.meshletBuffer[draw.meshletIndex];
	auto indices = getVertexIndices(primitive, meshlet, primitiveId);

	const auto mvp = camera.viewProjection * transformMatrix;
	device const auto& vtx0 = primitive.vertexBuffer[indices.x];
	device const auto& vtx1 = primitive.vertexBuffer[indices.y];
	device const auto& vtx2 = primitive.vertexBuffer[indices.z];

	const auto pos0 = mvp * float4(vtx0.position, 1.f);
	const auto pos1 = mvp * float4(vtx1.position, 1.f);
	const auto pos2 = mvp * float4(vtx2.position, 1.f);

	float2 size(visbuffer.get_width(), visbuffer.get_height());
	float2 pixel = (float2(gid) / size) * 2.f - 1.f;
	const auto barycentrics = calculateBarycentrics(pos0, pos1, pos2, pixel, size);

	const auto interpolatedColor = interpolate(barycentrics,
		metal::unpack_unorm4x8_to_float(vtx0.color),
		metal::unpack_unorm4x8_to_float(vtx1.color),
		metal::unpack_unorm4x8_to_float(vtx2.color));

	const auto albedo = interpolatedColor * float4(material.albedoFactor);

	//auto resolved = float4(hue2rgb(draw.meshletIndex * 1.71f), 1.f);
	color.write(albedo, gid);
}
