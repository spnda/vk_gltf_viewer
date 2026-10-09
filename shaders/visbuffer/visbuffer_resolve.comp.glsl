#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_nonuniform_qualifier : require

#include "visbuffer.h"
#include "color.h"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

#include "resource_table.h"

layout(push_constant, scalar) uniform readonly PushConstants {
	VisbufferResolvePushConstants pushConstants;
};

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(readonly_uimage2d_r32ui_heap[pushConstants.visbufferHandle]);
	if (any(greaterThanEqual(pixel, size)))
		return;

	// TODO: Is there some other way to clear the destination image?
	imageStore(writeonly_image2d_rgba8_heap[pushConstants.outputImageHandle], pixel, vec4(0.f));

	uint visBuffer = imageLoad(readonly_uimage2d_r32ui_heap[pushConstants.visbufferHandle], pixel).r;
	if (visBuffer == visbufferClearValue) // Nothing was drawn onto this pixel.
		return;

	uint drawIndex;
	uint primitiveId;
	unpackVisBuffer(visBuffer, drawIndex, primitiveId);

	restrict const meshlet_draw_t draw = pushConstants.drawBuffer.draws[drawIndex];
	restrict primitive_t primitive = pushConstants.primitiveBuffer.primitives[draw.primitive_index];
	restrict const material_t material = pushConstants.materialBuffer.materials[primitive.material_index];

	vec4 resolved = srgb_oetf(material.albedo_factor);

	imageStore(writeonly_image2d_rgba8_heap[pushConstants.outputImageHandle], pixel, resolved);
}
