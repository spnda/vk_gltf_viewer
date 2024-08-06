#ifndef SHADERS_RESOURCE_TABLE_H
#define SHADERS_RESOURCE_TABLE_H

#if defined(__METAL_VERSION__)
#include <metal_texture>
#endif

#include "common.h"
SHADER_NAMESPACE_BEGIN

SHADER_CONSTANT uint32_t sampledImageBinding = 0;
SHADER_CONSTANT uint32_t storageImageBinding = 1;

#if defined(SHADER_METAL)
struct SampledTextureEntry {
    metal::texture2d<float> tex;
    metal::sampler sampler;
};

struct ResourceTableBuffer {
    device SampledTextureEntry* sampled_textures_heap;
    device uint64_t* storage_image_heap;
};

using ResourceTableHandle = metal::uint32_t;
#elif defined(SHADER_CPP)
using ResourceTableHandle = std::uint32_t;
#else
layout(set = 0, binding = sampledImageBinding) uniform sampler2D sampled_textures_heap[];

layout(set = 0, binding = storageImageBinding, r32ui) uniform readonly  uimage2D readonly_uimage2d_r32ui_heap[];
layout(set = 0, binding = storageImageBinding, r32f ) uniform readonly  image2D  readonly_image2d_r32f_heap[];
layout(set = 0, binding = storageImageBinding, r32f ) uniform writeonly image2D  writeonly_image2d_r32f_heap[];
layout(set = 0, binding = storageImageBinding, rgba8) uniform writeonly image2D  writeonly_image2d_rgba8_heap[];

#define ResourceTableHandle uint
#endif

SHADER_CONSTANT ResourceTableHandle invalidHandle = ~0U;

SHADER_NAMESPACE_END
#endif
