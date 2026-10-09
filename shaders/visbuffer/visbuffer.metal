#include <metal_stdlib>

#include "visbuffer.h"
#include "mesh_common.h"
#include "culling.h"
#include "color.h"

namespace mtl = metal;

struct meshlet_vertex {
	float4 position [[position]];
	float4 prev_position;
	float4 color;
	float2 uv;
};

struct meshlet_primitive {
	uint draw_id;
	uint material_id;
	uint id [[primitive_id]];
	bool culled [[primitive_culled]];
};

using meshlet_t = mtl::mesh<meshlet_vertex, meshlet_primitive, shaders::maxVertices, shaders::maxPrimitives, mtl::topology::triangle>;

struct object_payload {
	uint base_index;
	mtl::array<uint8_t, shaders::maxMeshlets> indices;
};

/// Object shader that handles up to shaders::maxMeshlets meshlets per threadgroup.
/// This uses frustum culling and a modified index array to have a fully GPU-driven
/// pipeline and cull as many meshlets as possible.
[[object]] void visbuffer_object(
		object_data object_payload& payload [[payload]],
		mtl::mesh_grid_properties outGrid,
		constant const ulong& meshlet_draw_count [[buffer(0)]],
		device const shaders::camera_t& camera [[buffer(1)]],
		device const shaders::meshlet_draw_t* draws [[buffer(2)]],
		device const shaders::primitive_transform_t* transforms [[buffer(3)]],
		device const shaders::primitive_t* primitives [[buffer(4)]],
		uint gid [[threadgroup_position_in_grid]],
		uint thread_index [[thread_position_in_threadgroup]],
		uint threadgroup_size [[threads_per_threadgroup]]) {
	auto meshlet_count = uint(mtl::min(ulong(shaders::maxMeshlets), meshlet_draw_count - (gid * shaders::maxMeshlets)));
	auto base_id = payload.base_index = gid * shaders::maxMeshlets;

	// We use an atomic counter for how many meshlets are visible instead of SIMD intrinsics,
	// since those caused a GPU hang in some cases, for some reason. Might be useful to investigate
	// that in the future, since I doubt this atomic is absolutely free.
	threadgroup mtl::atomic<uint> visible_meshlets;

	const auto meshlet_loops = (meshlet_count + threadgroup_size - 1) / threadgroup_size;
	for (uint i = 0; i < meshlet_loops; ++i) {
		auto tidx = thread_index + i * threadgroup_size;
		auto idx = mtl::min(tidx, meshlet_count - 1U);

		device const auto& draw = draws[base_id + idx];
		device const auto& transform = transforms[draw.transform_index];
		device const auto& primitive = primitives[draw.primitive_index];
		device const auto& meshlet = primitive.meshlet_buffer[draw.meshlet_index];

		// We automatically mark the meshlet as culled if we are a thread that is overprocessing,
		// this prevents any buffer overruns in the payload index array later.
		bool visible = tidx == idx;

		// Frustum culling
		const auto world_aabb_center = (transform.matrix * float4(meshlet.aabb_center, 1.0f)).xyz;
		const auto world_aabb_extent = shaders::getWorldSpaceAabbExtent(meshlet.aabb_extents.xyz, transform.matrix);
		visible = visible && shaders::isAabbInFrustum(world_aabb_center, world_aabb_extent, camera.frustum);

		if (visible) {
			payload.indices[mtl::atomic_fetch_add_explicit(&visible_meshlets, 1, mtl::memory_order_relaxed)]
				= uint8_t(idx);
		}
	}

	threadgroup_barrier(mtl::mem_flags::mem_none);
	if (thread_index == 0) {
		outGrid.set_threadgroups_per_grid(
			uint3(mtl::atomic_load_explicit(&visible_meshlets, mtl::memory_order_relaxed), 1, 1));
	}
}

/// Mesh shader that handles a single meshlet/draw per threadgroup.
/// Since a meshlet can have a variable amount of primitives and vertices,
/// we use fancy loops to properly process them, at the cost of potentially
/// processing the last element multiple times.
[[mesh]] void visbuffer_mesh(
		meshlet_t meshlet_out,
		object_data const object_payload& payload [[payload]],
		device const shaders::meshlet_draw_t* draws [[buffer(0)]],
		device const shaders::primitive_transform_t* transforms [[buffer(1)]],
		device const shaders::primitive_t* primitives [[buffer(2)]],
		device const shaders::camera_t& camera [[buffer(3)]],
		device const shaders::material_t* materials [[buffer(4)]],
		uint payload_index [[threadgroup_position_in_grid]],
		uint tid [[thread_position_in_threadgroup]],
		uint threadgroup_size [[threads_per_threadgroup]]) {
	const auto draw_id = payload.base_index + payload.indices[payload_index];

	device auto& draw = draws[draw_id];

	device auto& primitive = primitives[draw.primitive_index];
	device auto& meshlet = primitive.meshlet_buffer[draw.meshlet_index];
	device auto& material = materials[primitive.material_index];

	if (tid == 0) {
		meshlet_out.set_primitive_count(meshlet.triangleCount);
	}

	device auto& transform = transforms[draw.transform_index];
	const auto mvp = camera.view_projection * transform.matrix;
	const auto prev_mvp = camera.prev_view_projection * transform.matrix;

	threadgroup mtl::array<float3, shaders::maxVertices> clip_vertices;

	const auto vertex_loops = (meshlet.vertexCount + threadgroup_size - 1) / threadgroup_size;
	for (uint i = 0; i < vertex_loops; ++i) {
		auto vidx = tid + i * threadgroup_size;
		vidx = mtl::min(vidx, meshlet.vertexCount - 1U);

		const auto vtx_idx = primitive.vertex_index_buffer[meshlet.vertexOffset + vidx];
		device const auto& vtx_pos = primitive.position_buffer[vtx_idx];

		// If the material's alpha mode is not opaque, meaning that we need to check for alpha values,
		// we'll read and interpolate the vertex color and UV values.
		float4 color = float4(0.0f);
		float2 uv = float2(0.0f);
		if (material.alpha_mode != shaders::alpha_mode_e::opaque) {
			color = primitive.vertex_buffer[vtx_idx].color;
			uv = primitive.uv_buffers[material.albedo.uv_set][vtx_idx];
		}

		auto pos = mvp * float4(vtx_pos, 1.f);
		clip_vertices[vidx] = pos.xyw;
		meshlet_out.set_vertex(vidx, meshlet_vertex {
			.position = pos,
			.prev_position = prev_mvp * float4(vtx_pos, 1.f),
			.color = color,
			.uv = uv,
		});
	}

	threadgroup_barrier(mtl::mem_flags::mem_threadgroup);

	const auto transform_det = mtl::determinant(transform.matrix);
	const auto primitive_loops = (meshlet.triangleCount + threadgroup_size - 1) / threadgroup_size;
	for (uint i = 0; i < primitive_loops; ++i) {
		auto pidx = tid + i * threadgroup_size;
		pidx = mtl::min(pidx, meshlet.triangleCount - 1U);

		auto j = pidx * 3;
		auto idx0 = primitive.primitive_index_buffer[meshlet.triangleOffset + j + 0];
		auto idx1 = primitive.primitive_index_buffer[meshlet.triangleOffset + j + 1];
		auto idx2 = primitive.primitive_index_buffer[meshlet.triangleOffset + j + 2];

		meshlet_out.set_index(j + 0, idx0);
		meshlet_out.set_index(j + 1, idx1);
		meshlet_out.set_index(j + 2, idx2);

		if (!material.double_sided) {
			const auto v0 = clip_vertices[idx0];
			const auto v1 = clip_vertices[idx1];
			const auto v2 = clip_vertices[idx2];
			const auto det = determinant(mtl::float3x3(v0, v1, v2));

			bool culled = transform_det < 0.0f
				? det > 0.0f // Front face culling with Y+ as up.
				: det < 0.0f; // Back face culling with Y+ as up.

			meshlet_out.set_primitive(pidx, meshlet_primitive {
				.draw_id = draw_id,
				.material_id = primitive.material_index,
				.id = pidx,
				.culled = culled
			});
		} else {
			meshlet_out.set_primitive(pidx, meshlet_primitive {
				.draw_id = draw_id,
				.material_id = primitive.material_index,
				.id = pidx,
				.culled = false, // Material is double sided, meaning we can't cull.
			});
		}
	}
}

struct fragment_in {
	meshlet_vertex vert;
	meshlet_primitive prim;
};

// I'm sorry about this, but this actually seems to work fine.
struct rgb16unorm {
	mtl::r16unorm<float> r;
	mtl::r16unorm<float> g;
	mtl::r16unorm<float> b;

	rgb16unorm() thread = default;
	rgb16unorm(float3 value) thread : r(value.r), g(value.g), b(value.b) {}

	explicit operator float3() thread {
		return float3(float(r), float(g), float(b));
	}
	explicit operator float3() threadgroup_imageblock {
		return float3(float(r), float(g), float(b));
	}
};

/// The imageblock struct used for the visbuffer fragment shader and the gbuffer generation
/// tile shader. The idea is to make use of the benefits of a visbuffer to reduce fragment shader
/// cost for each invocation and avoiding helper lanes for derivatives, and then recalculating the
/// derivatives using the barycentrics in a tile shader to manually interpolate vertex attributes
/// and generate the various GBuffer textures.
struct visbuffer_tile_data {
	mtl::rgba8unorm<float4> albedo [[raster_order_group(0)]];
#if GBUFFER_NORMAL_ENCODING == 1
	mtl::rg16snorm<float2> normal [[raster_order_group(0)]];
#else
	mtl::rgba16snorm<float4> normal [[raster_order_group(0)]];
#endif
	packed_half2 metallic_roughness [[raster_order_group(0)]];
	packed_half2 motion_vectors [[raster_order_group(0)]];

	uint visbuffer [[raster_order_group(0)]];

	// Since barycentrics will usually only be in the [0, 1] range except when MSAA is used, we can
	// safely use a unorm type with 16-bits per value to half our memory consumption for virtually no
	// loss.
	rgb16unorm barycentrics [[raster_order_group(0)]];
};

struct visbuffer_frag_out {
	visbuffer_tile_data tile_data [[imageblock_data, alias_implicit_imageblock]];
};

/// We assume here that accessing the barycentrics is free, since they needed to be calculated
/// anyway for generating the exact fragment position (I think).
[[fragment]] visbuffer_frag_out visbuffer_frag(
		fragment_in in [[stage_in]],
		device const shaders::material_t* materials [[buffer(0)]],
		const shaders::ResourceTable resourceTable [[buffer(1)]],
		float3 barycentrics [[barycentric_coord]]) {
	device const auto& material = materials[in.prim.material_id];
	if (material.alpha_mode != shaders::alpha_mode_e::opaque) {
		auto color = in.vert.color;

		if (material.albedo.index != shaders::invalid_handle) {
			device auto& albedo_tex = resourceTable[material.albedo.index];

			color *= shaders::srgb_eotf(albedo_tex.tex.sample(
				albedo_tex.sampler, transform_uv(material.albedo, in.vert.uv)));
		}

		if (color.a < material.alpha_cutoff)
			mtl::discard_fragment();
	}

	constexpr auto mv_scale = float2(0.5f, -0.5f);
	constexpr auto mv_offset = 0.5f;
	auto uv = (in.vert.position.xy / in.vert.position.w) * mv_scale + mv_offset;
	auto prev_uv = (in.vert.prev_position.xy / in.vert.prev_position.w) * mv_scale + mv_offset;
	return {
		.tile_data {
			.motion_vectors = half2(prev_uv - uv),
			.visbuffer = uint32_t(shaders::visbuffer_data(in.prim.draw_id, in.prim.id)),
			.barycentrics = barycentrics,
		},
	};
}

uint3 get_vertex_indices(
		device const shaders::primitive_t& primitive,
		device const shaders::meshlet_t& meshlet,
		uint32_t primitive_id) {
	uchar3 indices(
		primitive.primitive_index_buffer[meshlet.triangleOffset + primitive_id * 3 + 0],
		primitive.primitive_index_buffer[meshlet.triangleOffset + primitive_id * 3 + 1],
		primitive.primitive_index_buffer[meshlet.triangleOffset + primitive_id * 3 + 2]
	);

	return uint3(
		primitive.vertex_index_buffer[meshlet.vertexOffset + indices.x],
		primitive.vertex_index_buffer[meshlet.vertexOffset + indices.y],
		primitive.vertex_index_buffer[meshlet.vertexOffset + indices.z]
	);
}

/// See http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
/// for more details on how this works and the links to the relevant papers.
/// This version is slightly different, since it takes a precomputed barycentrics vector,
/// and only computes the derivatives.
/// TODO: Explore replacing these calculations with half precision floats, since the precision
///       issue shouldn't be noticeable here and recent Apple GPUs can run those ops at 2x
mtl::gradient3d calculate_gradient(
		float4 v0, float4 v1, float4 v2,
		float3 barycentrics,
		float2 pixel,
		float2 size) {
	pixel.y = -pixel.y; // flip because +Y is up.

	auto invW = 1.f / float3(v0.w, v1.w, v2.w);

	auto ndc0 = v0.xy * invW.x;
	auto ndc1 = v1.xy * invW.y;
	auto ndc2 = v2.xy * invW.z;

	auto invDet = 1.f / mtl::determinant(mtl::float2x2(ndc2 - ndc1, ndc0 - ndc1));
	auto ddx = float3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
	auto ddy = float3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;
	auto ddxSum = mtl::dot(ddx, float3(1.f));
	auto ddySum = mtl::dot(ddy, float3(1.f));

	auto deltaVec = pixel - ndc0;
	auto interpInvW = invW.x + deltaVec.x * ddxSum + deltaVec.y * ddySum;

	ddx *= 2.f / size.x;
	ddy *= 2.f / size.y;
	ddxSum *= 2.f / size.x;
	ddySum *= 2.f / size.y;

	//ddy *= -1.f;
	//ddySum *= -1.f;

	auto interpW_ddx = 1.f / (interpInvW + ddxSum);
	auto interpW_ddy = 1.f / (interpInvW + ddySum);

	return mtl::gradient3d(
		interpW_ddx * (barycentrics * interpInvW + ddx) - barycentrics,
		interpW_ddy * (barycentrics * interpInvW + ddy) - barycentrics);
}

template <typename T, size_t N>
mtl::vec<T, N> interpolate(float3 barycentrics, mtl::vec<T, N> v0, mtl::vec<T, N> v1, mtl::vec<T, N> v2) {
	return barycentrics.x * v0
		+ barycentrics.y * v1
		+ barycentrics.z * v2;
}

float3 interpolate_with_deriv(float3 barycentrics, mtl::gradient3d gradient, float3 v) {
	return float3(
		mtl::dot(v, barycentrics),
		mtl::dot(v, gradient.dPdx),
		mtl::dot(v, gradient.dPdy));
}

struct interpolated_value {
	float2 value;
	mtl::gradient2d grad;
	METAL_FUNC constexpr interpolated_value(float2 value, mtl::gradient2d grad) thread : value(value), grad(grad) {}
};

/// Interpolates the value from the given three vertices and interpolated it accordingly, while
/// also computing its derivatives.
interpolated_value interpolate_with_gradient(float3 barycentrics, mtl::gradient3d inp_grad, float2 v0, float2 v1, float2 v2) {
	const auto i0 = interpolate_with_deriv(barycentrics, inp_grad, float3(v0.x, v1.x, v2.x));
	const auto i1 = interpolate_with_deriv(barycentrics, inp_grad, float3(v0.y, v1.y, v2.y));
	return interpolated_value(float2(i0.x, i1.x), mtl::gradient2d(float2(i0.y, i1.y), float2(i0.z, i1.z)));
}

/// This transposes and downcasts the given matrix at once.
mtl::float3x3 as_transposed_3x3(mtl::float4x4 matrix) {
	// Each vector we pass in is a single column.
	return mtl::float3x3(
		mtl::float3(matrix[0][0], matrix[1][0], matrix[2][0]),
		mtl::float3(matrix[0][1], matrix[1][1], matrix[2][1]),
		mtl::float3(matrix[0][2], matrix[1][2], matrix[2][2])
	);
}

/// This kernel/tile shader uses the visbuffer data and barycentrics from the previous fragment
/// shader to generate a GBuffer.
[[kernel]] void gbuffer_generation(
		mtl::imageblock<visbuffer_tile_data> tile_data [[alias_implicit_imageblock]],
		device const shaders::meshlet_draw_t* draws [[buffer(0)]],
		device const shaders::primitive_transform_t* transforms [[buffer(1)]],
		device const shaders::primitive_t* primitives [[buffer(2)]],
		device const shaders::camera_t& camera [[buffer(3)]],
		device const shaders::material_t* materials [[buffer(4)]],
		const shaders::ResourceTable resourceTable [[buffer(5)]],
		ushort2 local_tid [[thread_position_in_threadgroup]],
		ushort2 tid [[thread_position_in_grid]],
		ushort2 grid_size [[threads_per_grid]]) {
	threadgroup_imageblock visbuffer_tile_data* data = tile_data.data(local_tid);

	auto visbuffer = shaders::visbuffer_data(data->visbuffer);
	if (!visbuffer.is_valid())
		return;

    decltype(auto) draw = draws[visbuffer.draw_index];
	//device const auto& draw = draws[visbuffer.draw_index];
	device const auto& transform = transforms[draw.transform_index];
	device const auto& primitive = primitives[draw.primitive_index];
	device const auto& material = materials[primitive.material_index];

	device const auto& meshlet = primitive.meshlet_buffer[draw.meshlet_index];
	auto indices = get_vertex_indices(primitive, meshlet, visbuffer.primitive_id);

	device const auto& pos0 = primitive.position_buffer[indices.x];
	device const auto& pos1 = primitive.position_buffer[indices.y];
	device const auto& pos2 = primitive.position_buffer[indices.z];

	const auto mvp = camera.view_projection * transform.matrix;
	float2 size(grid_size.x, grid_size.y); // TODO: Is this actually correct?
	float2 pixel = (float2(tid) / size) * 2.f - 1.f;
	auto gradient = calculate_gradient(
		mvp * float4(pos0, 1.f),
		mvp * float4(pos1, 1.f),
		mvp * float4(pos2, 1.f),
		float3(data->barycentrics), pixel, size);

	device const auto& vtx0 = primitive.vertex_buffer[indices.x];
	device const auto& vtx1 = primitive.vertex_buffer[indices.y];
	device const auto& vtx2 = primitive.vertex_buffer[indices.z];

	const auto color = interpolate(float3(data->barycentrics),
		float4(vtx0.color), float4(vtx1.color), float4(vtx2.color));
	data->albedo = color * float4(material.albedo_factor);

	if (material.albedo.index != shaders::invalid_handle) {
		device auto& albedo_tex = resourceTable[material.albedo.index];

		device auto* albedo_uv_buffer = primitive.uv_buffers[material.albedo.uv_set];
		const auto uv = interpolate_with_gradient(float3(data->barycentrics), gradient,
			albedo_uv_buffer[indices.x], albedo_uv_buffer[indices.y], albedo_uv_buffer[indices.z]);

		// Metal's pixel types don't support *= for some reason...
		data->albedo = data->albedo * shaders::srgb_eotf(albedo_tex.tex.sample(
			albedo_tex.sampler, transform_uv(material.albedo, uv.value), uv.grad));
	}

	// This currently only supports opaque or masked alpha modes, so the alpha value needs to always be 1 here.
	data->albedo = float4(float4(data->albedo).xyz, 1.f);

	// Sample metallic roughness
	data->metallic_roughness = half2(material.roughness_factor, material.metallic_factor);
	if (material.metallic_roughness.index != shaders::invalid_handle) {
		device auto& mr_tex = resourceTable[material.metallic_roughness.index];
		device auto* mr_uv_buffer = primitive.uv_buffers[material.metallic_roughness.uv_set];
		const auto uv = interpolate_with_gradient(float3(data->barycentrics), gradient,
			mr_uv_buffer[indices.x], mr_uv_buffer[indices.y], mr_uv_buffer[indices.z]);

		// Its green channel contains roughness values and its blue channel contains metalness values.
		data->metallic_roughness *= half2(mr_tex.tex.sample(
			mr_tex.sampler, transform_uv(material.metallic_roughness, uv.value), uv.grad).gb);
	}

	const auto transposed_inverse = as_transposed_3x3(transform.inverse_matrix);
#if VERTEX_BUFFER_NORMAL_ENCODING == 1
	auto normal = transposed_inverse * interpolate(float3(data->barycentrics),
		shaders::normal_decode(float2(vtx0.normal)),
		shaders::normal_decode(float2(vtx1.normal)),
		shaders::normal_decode(float2(vtx2.normal)));
#else
	auto normal = transposed_inverse * interpolate(float3(data->barycentrics),
		float3(vtx0.normal),
		float3(vtx1.normal),
		float3(vtx2.normal));
#endif

	if (material.normal.index != shaders::invalid_handle) {
#if VERTEX_BUFFER_TANGENT_ENCODING == 1
		const auto tangent = interpolate(float3(data->barycentrics),
			shaders::tangent_decode(vtx0.tangent),
			shaders::tangent_decode(vtx1.tangent),
			shaders::tangent_decode(vtx2.tangent));
#else
		const auto tangent = interpolate(float3(data->barycentrics),
			float4(vtx0.tangent), float4(vtx1.tangent), float4(vtx2.tangent));
#endif

		device auto& tex = resourceTable[material.normal.index];
		device auto* uv_buffer = primitive.uv_buffers[material.normal.uv_set];
		const auto uv = interpolate_with_gradient(float3(data->barycentrics), gradient,
			uv_buffer[indices.x], uv_buffer[indices.y], uv_buffer[indices.z]);

		// The texture binding for normal textures MAY additionally contain a scalar scale value that linearly scales X and Y components of the normal vector.
		// Normal vectors MUST be normalized before being used in lighting equations. When scaling is used, vector normalization happens after scaling.
		auto sampled_normal = tex.tex.sample(tex.sampler, transform_uv(material.normal, uv.value), uv.grad).xyz;
		sampled_normal = mtl::normalize((sampled_normal * 2.0 - 1.0) * float3(material.normal_scale, material.normal_scale, 1.f));

		auto T = mtl::normalize(transposed_inverse * tangent.xyz);
		T = mtl::normalize(T - mtl::dot(T, normal) * normal);
		auto B = mtl::cross(normal, T) * tangent.w;
		auto TBN = mtl::float3x3(T, B, normal);

		normal = mtl::normalize(TBN * sampled_normal);
	}

#if GBUFFER_NORMAL_ENCODING == 1
	data->normal = shaders::normal_encode(mtl::normalize(normal));
#else
	data->normal = float4(mtl::normalize(float3(normal)), 1.f);
#endif
}
