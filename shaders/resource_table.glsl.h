#ifndef RESOURCE_TABLE_GLSL_H
#define RESOURCE_TABLE_GLSL_H

#include "common.glsl.h"
GLSL_NAMESPACE_BEGIN

GLSL_CONSTANT uint sampledImageBinding = 0;
GLSL_CONSTANT uint storageImageBinding = 1;

#if !defined(__cplusplus)
layout(set = 0, binding = sampledImageBinding) uniform sampler2D sampled_textures_heap[];

layout(set = 0, binding = storageImageBinding, r32ui) uniform readonly  uimage2D readonly_uimage2d_r32ui_heap[];
layout(set = 0, binding = storageImageBinding, rgba8) uniform writeonly image2D  writeonly_image2d_rgba8_heap[];

#define ResourceTableHandle uint
#else
using ResourceTableHandle = std::uint32_t;
#endif

GLSL_CONSTANT ResourceTableHandle invalidHandle = ~0U;

GLSL_NAMESPACE_END
#endif
