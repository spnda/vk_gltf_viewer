#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require

#include "resource_table.h"

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

struct HiZReducePushConstants {
	ResourceTableHandle sourceImage;
	ResourceTableHandle outputImage;
	uvec2 imageSize;
};

layout(push_constant, scalar) readonly uniform PushConstants {
	HiZReducePushConstants pushConstants;
};

void main() {
	uvec2 pos = gl_GlobalInvocationID.xy;

	if (any(greaterThan(pos, pushConstants.imageSize)))
		return;

	// Sampler is set up to do min reduction, so this computes the minimum depth of a 2x2 texel quad
	float depth = texture(sampled_textures_heap[pushConstants.sourceImage], (vec2(pos) + vec2(0.5)) / pushConstants.imageSize).x;

	imageStore(writeonly_image2d_r32f_heap[pushConstants.outputImage], ivec2(pos), vec4(depth));
}
