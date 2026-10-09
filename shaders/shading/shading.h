#ifndef SHADING_H
#define SHADING_H

#include "common.h"
#include "color.h"

SHADER_NAMESPACE_BEGIN

struct shading_constants {
	color_space_e internal_color_space;
	color_space_e output_color_space;
	// Peak output relative to SDR reference white (1.0), i.e. the EDR headroom. Always 1 for SDR outputs.
	float display_headroom = 1.f;
};

struct exposure_constants {
	color_space_e internal_color_space;
	packed_u16vec2 image_size;
	float min_log_luminance = -10.f;
	float log_luminance_range = 20.f;
	float one_over_log_luminance_range = 1.f / log_luminance_range;
	float time_delta;
	float tau = 1.1f;
};

#if !defined(SHADER_GLSL)
SHADER_CONSTANT uint16_t histogram_group_size = 16; // Always square WGs
SHADER_CONSTANT uint16_t num_histogram_bins = histogram_group_size * histogram_group_size; // 256
#endif

#if defined(SHADER_METAL)
struct gbuffer_tile_data {
	packed_half4 color [[raster_order_group(0)]];
	mtl::rgba8unorm<half4> albedo [[raster_order_group(0)]];
#if GBUFFER_NORMAL_ENCODING == 1
	mtl::rg16snorm<float2> normal [[raster_order_group(0)]];
#else
	mtl::rgba16snorm<float4> normal [[raster_order_group(0)]];
#endif
	packed_half2 metallic_roughness [[raster_order_group(0)]];
};
#endif

SHADER_NAMESPACE_END
#endif
