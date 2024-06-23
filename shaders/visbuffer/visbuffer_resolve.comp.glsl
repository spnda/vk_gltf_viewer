#version 460
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_nonuniform_qualifier : require

#include "visbuffer.h.glsl"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

#include "resource_table.h.glsl"

layout(push_constant, scalar) uniform readonly PushConstants {
	VisbufferResolvePushConstants pushConstants;
};

#define MAX_COLORS 10
vec3 meshletcolors[MAX_COLORS] = {
	vec3(1,0,0),
	vec3(0,1,0),
	vec3(0,0,1),
	vec3(1,1,0),
	vec3(1,0,1),
	vec3(0,1,1),
	vec3(1,0.5,0),
	vec3(0.5,1,0),
	vec3(0,0.5,1),
	vec3(1,1,1)
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

	vec4 resolved = vec4(meshletcolors[drawIndex % MAX_COLORS], 1.f);

	imageStore(writeonly_image2d_rgba8_heap[pushConstants.outputImageHandle], pixel, resolved);
}
