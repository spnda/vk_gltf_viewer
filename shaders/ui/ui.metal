#include <metal_math>

#include "ui.h"

namespace mtl = metal;

struct RasterizerOutput {
	float4 position [[position]];
	float4 color;
	float2 uv;
};

[[vertex]] RasterizerOutput ui_vert(
		uint vertexId [[vertex_id]],
		constant const shaders::UiPushConstants& constants [[buffer(0)]]) {
	const device auto& vert = constants.vertices[vertexId];
	auto pos = vert.pos * constants.scale + constants.translate;
	return RasterizerOutput {
		float4(pos.x, -pos.y, 0, 1),
		mtl::unpack_unorm4x8_to_float(vert.col),
		vert.uv,
	};
}

[[fragment]] float4 ui_frag(
		RasterizerOutput input [[stage_in]],
		constant const shaders::UiPushConstants& constants [[buffer(0)]],
		const shaders::ResourceTable resourceTable [[buffer(1)]]) {
	device auto& entry = resourceTable[constants.imageIndex];
	const auto col = input.color * entry.tex.sample(entry.sampler, input.uv);
	const auto linear = shaders::srgb_eotf(col.xyz);
	return float4(
		convert_to_output_color_space(linear, shaders::color_space_e::srgb_linear, constants.output_color_space),
		col.w
	);
}
