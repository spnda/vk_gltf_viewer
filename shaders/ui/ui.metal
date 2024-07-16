#include <metal_math>

#include "ui.h.glsl"

using namespace metal;

struct RasterizerOutput {
	float4 position [[position]];
	float4 color;
	float2 uv;
};

vertex RasterizerOutput ui_vert(uint vertexId [[vertex_id]],
								const constant glsl::UiPushConstants& constants [[buffer(0)]]) {
    const device auto& vert = constants.vertices[vertexId];
	auto pos = vert.pos * constants.scale + constants.translate;
	return RasterizerOutput {
		float4(pos.x, -pos.y, 0, 1),
		metal::unpack_unorm4x8_to_float(vert.col),
		vert.uv,
	};
}

fragment float4 ui_frag(RasterizerOutput input [[stage_in]],
						const constant glsl::UiPushConstants& constants [[buffer(0)]],
						const constant glsl::ResourceTableBuffer& resourceTable [[buffer(1)]]) {
	float4 color = float4(pow(input.color.rgb, float3(2.2)), input.color.a); // TODO: Use srgb.h header, or not convert explicitly.
	device auto& entry = resourceTable.sampled_textures_heap[constants.imageIndex];
	return color * entry.tex.sample(entry.sampler, input.uv);
}
