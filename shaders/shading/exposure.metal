#include <metal_stdlib>

#include "shading.h"

namespace mtl = metal;

float compute_luminance(float3 color, shaders::color_space_e color_space) {
	switch (color_space) {
		using enum shaders::color_space_e;
		case srgb_linear:
		case bt709_linear:
			return mtl::dot(color, float3(0.2126f, 0.7152f, 0.0722f));
		case bt2020_linear:
			// The BT.2020 spec lists this calculation for Y' and not Y... does that matter?
			return mtl::dot(color, float3(0.2627f, 0.6780f, 0.0593f));
		case display_p3_linear:
			return mtl::dot(color, float3(0.2289746f, 0.6917385f, 0.0792869f));
		case cie_1931:
			return color.y;
		default:
			return 1.f;
	}
}

uint get_histogram_index(float3 color, constant shaders::exposure_constants& constants) {
	auto lum = compute_luminance(color, constants.internal_color_space);
	if (lum < 2E-10) return 0;

	auto log_lum = mtl::saturate((mtl::log2(lum) - constants.min_log_luminance) * constants.one_over_log_luminance_range);
	return static_cast<uint>(log_lum * 254.0 + 1.0);
}

[[kernel]] void compute_luminance_histogram(
		mtl::imageblock<shaders::gbuffer_tile_data> tile_data [[alias_implicit_imageblock]],
		constant const shaders::exposure_constants& constants [[buffer(0)]],
		device mtl::atomic<uint32_t>* histogram [[buffer(1)]],
		threadgroup mtl::atomic<uint32_t>* shared_histogram [[threadgroup(0)]],
		//ushort2 gid [[threadgroup_position_in_grid]],
		//ushort2 group_count [[threadgroups_per_grid]],
		ushort2 local_tid [[thread_position_in_threadgroup]],
		ushort2 tid [[thread_position_in_grid]]) {
	const auto local_index = local_tid.y * shaders::histogram_group_size + local_tid.x;

	threadgroup_imageblock auto* data = tile_data.data(local_tid);

	atomic_store_explicit(&shared_histogram[local_index], 0, mtl::memory_order_relaxed);

	threadgroup_barrier(mtl::mem_flags::mem_threadgroup);

	// Compute the histogram bin index for the given HDR color, and increment that local bin by 1.
	// We only write to threadgroup memory first since that should be more efficient and quicker,
	// since we write to device memory less often and therefore also reduce atomic ops across groups.
	const auto color = float4(data->color).rgb;
	const auto histogram_idx = get_histogram_index(color, constants);
	if (tid.x < constants.image_size.x && tid.y < constants.image_size.y)
		atomic_fetch_add_explicit(&shared_histogram[histogram_idx], 1, mtl::memory_order_relaxed);

	threadgroup_barrier(mtl::mem_flags::mem_threadgroup);

	if (local_index < shaders::num_histogram_bins) {
		const auto val = atomic_load_explicit(&shared_histogram[local_index], mtl::memory_order_relaxed);
		atomic_fetch_add_explicit(&histogram[local_index], val, mtl::memory_order_relaxed);
	}
}

[[kernel]] void compute_average_luminance(
		constant const shaders::exposure_constants& constants [[buffer(0)]],
		device float* luminance [[buffer(1)]],
		device const uint32_t* histogram [[buffer(2)]],
		ushort2 local_tid [[thread_position_in_threadgroup]],
		ushort2 tid [[thread_position_in_grid]]) {
	const auto local_index = local_tid.y * shaders::histogram_group_size + local_tid.x;

	threadgroup mtl::array<uint32_t, shaders::num_histogram_bins> shared_histogram;

	const auto bin_entry_count = histogram[local_index];
	shared_histogram[local_index] = bin_entry_count * local_index;

	threadgroup_barrier(mtl::mem_flags::mem_threadgroup);

	// Merge sum that stores the final sum in the first element of the array
	for (auto idx = (shaders::num_histogram_bins >> 1); idx > 0; idx >>= 1) {
		if (local_index < idx) {
			shared_histogram[local_index] += shared_histogram[local_index + idx];
		}

		threadgroup_barrier(mtl::mem_flags::mem_threadgroup);
	}

	if (local_index == 0) {
		const auto pixel_count = constants.image_size.x * constants.image_size.y;
		const auto weighted_log_avg = (static_cast<float>(shared_histogram[0]) / mtl::max((float)pixel_count - bin_entry_count, 1.0)) - 1.0;
		const auto weighted_avg_lum = mtl::exp2(((weighted_log_avg / 254.0) * constants.log_luminance_range) + constants.min_log_luminance);
		const auto alpha = mtl::saturate(1 - mtl::exp(-constants.time_delta * constants.tau));
		*luminance = mtl::mix(*luminance, weighted_avg_lum, alpha);
	}
}
