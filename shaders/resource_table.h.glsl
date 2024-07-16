#ifndef RESOURCE_TABLE_GLSL_H
#define RESOURCE_TABLE_GLSL_H

#if defined(__METAL_VERSION__)
#include <metal_texture>
#endif

#include "common.h.glsl"
GLSL_NAMESPACE_BEGIN

GLSL_CONSTANT uint sampledImageBinding = 0;
GLSL_CONSTANT uint storageImageBinding = 1;

#if defined(__METAL_VERSION__)
struct SampledTextureEntry {
    metal::texture2d<float> tex;
    metal::sampler sampler;
};

struct ResourceTableBuffer {
    device SampledTextureEntry* sampled_textures_heap;
    device uint64_t* storage_image_heap;
};

using ResourceTableHandle = metal::uint;
#elif defined(__cplusplus)
using ResourceTableHandle = std::uint32_t;
#else
layout(set = 0, binding = sampledImageBinding) uniform sampler2D sampled_textures_heap[];

layout(set = 0, binding = storageImageBinding, r32ui) uniform readonly  uimage2D readonly_uimage2d_r32ui_heap[];
layout(set = 0, binding = storageImageBinding, r32f ) uniform readonly  image2D  readonly_image2d_r32f_heap[];
layout(set = 0, binding = storageImageBinding, r32f ) uniform writeonly image2D  writeonly_image2d_r32f_heap[];
layout(set = 0, binding = storageImageBinding, rgba8) uniform writeonly image2D  writeonly_image2d_rgba8_heap[];

#define ResourceTableHandle uint
#endif

GLSL_CONSTANT ResourceTableHandle invalidHandle = ~0U;

GLSL_NAMESPACE_END
#endif
