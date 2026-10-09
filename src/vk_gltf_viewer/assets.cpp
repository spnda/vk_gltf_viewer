#include <ranges>
#include <utility>
#include <mutex>

#include <fmt/std.h>

#include <mesh_common.h>

#include <util.hpp>
#include <vk_gltf_viewer/assets.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <meshoptimizer.h>

#include <mikktspace.h>

#define WUFFS_CONFIG__MODULE__AUX__IMAGE // For C++ image API
#include "wuffs-v0.4.c"

#define DDS_USE_STD_FILESYSTEM
#include <dds.hpp>

namespace fs = std::filesystem;
namespace fg = fastgltf;

#if defined(_MSC_VER) && !defined(__clang__)
std::size_t operator""UZ(unsigned long long int x) noexcept {
	return std::size_t(x);
}
#endif

struct buffer_load_task_t : ExceptionTaskSet {
	fg::Asset& asset;
	fs::path folder;

	explicit buffer_load_task_t(fg::Asset& asset, fs::path folder) : asset(asset), folder(std::move(folder)) {
		m_SetSize = fg::max(1UZ, asset.buffers.size());
	}
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void buffer_load_task_t::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.buffers.empty())
		return;

	for (auto i = range.start; i < range.end; ++i) {
		auto& buffer = asset.buffers[i];

		// The buffer data is already in CPU memory, we don't need to do anything.
		if (std::holds_alternative<fg::sources::Vector>(buffer.data) || std::holds_alternative<fg::sources::Array>(buffer.data) || std::holds_alternative<fg::sources::ByteView>(buffer.data)) {
			continue;
		}

		if (std::holds_alternative<fg::sources::Fallback>(buffer.data))
			continue; // Ignore these.

		// We only support loading from local URIs.
		assert(std::holds_alternative<fg::sources::URI>(buffer.data));

		const auto& uri = std::get<fg::sources::URI>(buffer.data);
		auto filePath = folder / uri.uri.fspath();
		std::ifstream file(filePath, std::ios::binary);
		if (!file) {
			throw std::runtime_error(fmt::format("Failed to open buffer: {}", filePath));
		}

		fg::StaticVector<std::byte> data(buffer.byteLength);
		file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size_bytes()));
		fg::sources::Array arraySource {
			std::move(data),
		};
		buffer.data = std::move(arraySource);
	}
}

/** Replacement buffer data adapter for fastgltf which supports decompressing with EXT_meshopt_compression */
struct compressed_buffer_data_adapter_t : enki::ITaskSet {
	fg::Asset& asset;

	std::vector<std::optional<fastgltf::StaticVector<std::byte>>> decompressed_buffers;

	enki::Dependency buffer_load_dependency;

	explicit compressed_buffer_data_adapter_t(fg::Asset& asset) : asset(asset) {
		m_SetSize = asset.bufferViews.size();
		m_MinRange = fastgltf::min(24U, m_SetSize);
		decompressed_buffers.resize(m_SetSize);
	}

	/** Get the data pointer of a loaded (possibly compressed) buffer */
	[[nodiscard]] static auto getData(const fastgltf::Buffer& buffer, std::size_t byteOffset, std::size_t byteLength) {
		ZoneScoped;
		using namespace fastgltf;
		return std::visit(visitor {
			[](auto&) -> span<const std::byte> {
				assert(false && "Tried accessing a buffer with no data, likely because no buffers were loaded. Perhaps you forgot to specify the LoadExternalBuffers option?");
				return {};
			},
			[](const sources::Fallback& fallback) -> span<const std::byte> {
				assert(false && "Tried accessing data of a fallback buffer.");
				return {};
			},
			[&](const sources::Array& array) -> span<const std::byte> {
				return span(array.bytes.data(), array.bytes.size_bytes());
			},
			[&](const sources::Vector& vec) -> span<const std::byte> {
				return span(vec.bytes.data(), vec.bytes.size());
			},
			[&](const sources::ByteView& bv) -> span<const std::byte> {
				return bv.bytes;
			},
		}, buffer.data).subspan(byteOffset, byteLength);
	}

	/** Decompress all buffer views and store them in this adapter */
	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override {
		ZoneScoped;

		for (auto i = range.start; i < range.end; ++i) {
			auto& buffer_view = asset.bufferViews[i];
			if (!buffer_view.meshoptCompression) {
				continue;
			}

			// This is a compressed buffer view.
			// For the original implementation, see https://github.com/jkuhlmann/cgltf/pull/129#issue-739550034
			auto& mc = *buffer_view.meshoptCompression;
			fastgltf::StaticVector<std::byte> result(mc.count * mc.byteStride);

			// Get the data span from the compressed buffer.
			auto data = getData(asset.buffers[mc.bufferIndex], mc.byteOffset, mc.byteLength);

			int rc = -1;
			switch (mc.mode) {
				using enum fg::MeshoptCompressionMode;
				case Attributes: {
					rc = meshopt_decodeVertexBuffer(result.data(), mc.count, mc.byteStride,
													reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
				case Triangles: {
					rc = meshopt_decodeIndexBuffer(result.data(), mc.count, mc.byteStride,
											  reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
				case Indices: {
					rc = meshopt_decodeIndexSequence(result.data(), mc.count, mc.byteStride,
												reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
			}

			//if (rc != 0)
			//	  return false;

			switch (mc.filter) {
				using enum fg::MeshoptCompressionFilter;
				case None:
					break;
				case Octahedral: {
					meshopt_decodeFilterOct(result.data(), mc.count, mc.byteStride);
					break;
				}
				case Quaternion: {
					meshopt_decodeFilterQuat(result.data(), mc.count, mc.byteStride);
					break;
				}
				case Exponential: {
					meshopt_decodeFilterExp(result.data(), mc.count, mc.byteStride);
					break;
				}
			}

			decompressed_buffers[i] = std::move(result);
		}
	}

	auto operator()([[maybe_unused]] const fastgltf::Asset& _, std::size_t buffer_view_idx) const {
		ZoneScoped;
		using namespace fastgltf;

		auto& bufferView = asset.bufferViews[buffer_view_idx];
		if (bufferView.meshoptCompression) {
			assert(decompressed_buffers.size() == asset.bufferViews.size());

			assert(decompressed_buffers[buffer_view_idx].has_value());
			return span(decompressed_buffers[buffer_view_idx]->data(), decompressed_buffers[buffer_view_idx]->size_bytes());
		}

		return getData(asset.buffers[bufferView.bufferIndex], bufferView.byteOffset, bufferView.byteLength);
	}
};

struct image_load_task_t : ExceptionTaskSet {
	const fg::Asset& asset;
	std::shared_ptr<graphics::renderer> renderer;

	std::vector<std::shared_ptr<graphics::image_t>> images;

	explicit image_load_task_t(const fg::Asset& asset, std::shared_ptr<graphics::renderer> renderer) : asset(asset), renderer(std::move(renderer)) {
		images.resize(asset.images.size());
		m_SetSize = fg::max(1UZ, asset.images.size());
	}

	std::shared_ptr<graphics::image_t> load_default(std::span<const std::byte> image_bytes);

	static fg::MimeType detect_mime_type(std::span<const std::byte> data);
	std::shared_ptr<graphics::image_t> load(std::span<const std::byte> image_bytes, fg::MimeType mime_type);
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

std::shared_ptr<graphics::image_t> image_load_task_t::load_default(std::span<const std::byte> image_bytes) {
	ZoneScoped;

	wuffs_aux::DecodeImageCallbacks callbacks;
	wuffs_aux::sync_io::MemoryInput input(reinterpret_cast<const char*>(image_bytes.data()), image_bytes.size_bytes());
	auto decode_result = wuffs_aux::DecodeImage(callbacks, input);
	if (!decode_result.error_message.empty()) {
		throw std::runtime_error(decode_result.error_message);
	}
	if (!decode_result.pixbuf.pixcfg.pixel_format().is_interleaved()) {
		throw std::runtime_error("Cannot load non-interleaved PNG/JPEG images.");
	}

	std::uint32_t width = decode_result.pixbuf.pixcfg.width(), height = decode_result.pixbuf.pixcfg.height();

	auto image_data = std::span(
		reinterpret_cast<std::byte*>(decode_result.pixbuf.plane(0).ptr),
		decode_result.pixbuf.pixcfg.pixbuf_len());
	return renderer->create_shared_image(image_data, glm::u32vec2(width, height));
}

/** Sometimes, the mimeType field is unspecified. Here, we try and detect the mimeType if it's initially None */
fg::MimeType image_load_task_t::detect_mime_type(std::span<const std::byte> data) {
	ZoneScoped;

	// The glTF did not provide a mime type. Try to detect the image format using the header magic.
	// See https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#images for the table of patterns
	if (data[0] == std::byte{0xFF} && data[1] == std::byte{0xD8} && data[2] == std::byte{0xFF}) {
		return fg::MimeType::JPEG;
	}

	static constexpr auto png_magic = std::to_array<std::uint8_t>({
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A
	});
	if (std::memcmp(data.data(), png_magic.data(), png_magic.size()) == 0) {
		return fg::MimeType::PNG;
	}

	if (*reinterpret_cast<const std::uint32_t*>(data.data()) == dds::DdsMagicNumber::DDS) {
		return fg::MimeType::DDS;
	}

	if (data[0] == std::byte{0xAB} && data[1] == std::byte{'K'} && data[2] == std::byte{'T'} && data[3] == std::byte{'X'}) {
		return fg::MimeType::KTX2;
	}

	throw std::runtime_error(
		fmt::format("Failed to detect image mime type while loading. Header magic: {0:x}", *reinterpret_cast<const std::uint32_t*>(data.data())));
}

std::shared_ptr<graphics::image_t> image_load_task_t::load(std::span<const std::byte> image_bytes, fg::MimeType mime_type) {
	ZoneScoped;
	if (mime_type == fg::MimeType::None) {
		mime_type = detect_mime_type(image_bytes);
	}

	switch (mime_type) {
		using fg::MimeType;
		case MimeType::PNG:
		case MimeType::JPEG: {
			return load_default(image_bytes);
		}
		default: {
			throw std::runtime_error(fmt::format("Unsupported mime type for loading images: {}", fastgltf::to_underlying(mime_type)));
		}
	}
}

void image_load_task_t::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.images.empty())
		return;

	for (auto i = range.start; i < range.end; ++i) {
		const auto& image = asset.images[i];

		std::visit(fg::visitor {
			[]([[maybe_unused]] auto& arg) {
				throw std::runtime_error("Got an unexpected image data source. Can't load image.");
			},
			[&](const fg::sources::Array& array) {
				images[i] = load(std::span(array.bytes.data(), array.bytes.size()), array.mimeType);
			},
			[&](const fg::sources::BufferView& bufferView) {
				const auto& view = asset.bufferViews[bufferView.bufferViewIndex];
				const auto& buffer = asset.buffers[view.bufferIndex];
				return std::visit(fg::visitor{
					[]([[maybe_unused]] auto& arg) {
						throw std::runtime_error("Got an unexpected image data source. Can't load image.");
					},
					[&](const fg::sources::Array& array) {
						images[i] = load(std::span(array.bytes.data() + view.byteOffset, array.bytes.size()), bufferView.mimeType);
					}
				}, buffer.data);
			}
		}, image.data);
	}
}

/**
 * The TextureCreateTask is responsible for creating samplers and bundling them together into textures.
 * We use a set size of 1 here, since this is a process that is pretty much free and the thread scheduling
 * overhead would be too big in most cases.
 */
struct texture_create_task_t : ExceptionTaskSet {
	const fg::Asset& asset;
	std::shared_ptr<graphics::renderer> renderer;

	std::vector<std::shared_ptr<graphics::texture_t>> textures;

	enki::Dependency image_load_dependency;

	explicit texture_create_task_t(const fg::Asset& asset, std::shared_ptr<graphics::renderer> renderer) : asset(asset), renderer(std::move(renderer)) {
		textures.resize(asset.textures.size());
		m_SetSize = 1;
	}

	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void texture_create_task_t::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	std::vector<std::shared_ptr<graphics::sampler_t>> samplers(asset.samplers.size());

	for (std::size_t i = 0; auto& sampler : asset.samplers)
		samplers[i++] = renderer->create_shared_sampler(sampler);

	auto* image_task = dynamic_cast<const image_load_task_t*>(image_load_dependency.GetDependencyTask());
	for (std::size_t i = 0; auto& texture : asset.textures) {
		auto sampler = texture.samplerIndex.has_value() ? samplers[texture.samplerIndex.value()] : renderer->get_default_sampler();
		textures[i++] = renderer->create_shared_texture(image_task->images[*texture.imageIndex], sampler);
	}
}

struct material_load_task_t : enki::ITaskSet {
	const fg::Asset& asset;
	std::shared_ptr<graphics::renderer> renderer;
	std::vector<graphics::material_index> materials;

	std::mutex material_mutex; // TODO: Should the renderer have this mutex instead?

	enki::Dependency texture_dependency;

	explicit material_load_task_t(const fg::Asset& asset, std::shared_ptr<graphics::renderer> renderer) noexcept : asset(asset), renderer(std::move(renderer)) {
		materials.resize(asset.materials.size());
		m_SetSize = fg::max(1UZ, asset.materials.size());
		m_MinRange = fg::min(16U, m_SetSize);
	}

	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void material_load_task_t::ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.materials.empty())
		return;

	auto* textureTask = dynamic_cast<const texture_create_task_t*>(texture_dependency.GetDependencyTask());
	for (auto i = range.start; i < range.end; ++i) {
		auto& gltf_material = asset.materials[i];
		auto& pbr = gltf_material.pbrData;

		shaders::material_t material {
			.albedo_factor = glm::make_vec4(pbr.baseColorFactor.data()),
			.metallic_factor = pbr.metallicFactor,
			.roughness_factor = pbr.roughnessFactor,
			.alpha_mode = [&]() {
				switch (gltf_material.alphaMode) {
					using enum fastgltf::AlphaMode;
					case Opaque: return shaders::alpha_mode_e::opaque;
					case Mask: return shaders::alpha_mode_e::mask;
					case Blend: return shaders::alpha_mode_e::blend;
					default: std::unreachable();
				}
			}(),
			.alpha_cutoff = gltf_material.alphaCutoff,
			.double_sided = gltf_material.doubleSided,
			.ior = gltf_material.ior,
		};

		if (pbr.baseColorTexture) {
			auto& tex = material.albedo;
			tex.index = textureTask->textures[pbr.baseColorTexture->textureIndex]->get_handle();
			tex.uv_set = static_cast<std::uint32_t>(pbr.baseColorTexture->texCoordIndex);
			if (auto& transform = pbr.baseColorTexture->transform; transform) {
				tex.uv_offset = glm::make_vec2(transform->uvOffset.data());
				tex.uv_scale = glm::make_vec2(transform->uvScale.data());
				tex.uv_rotation = transform->rotation;
			}
		}

		if (pbr.metallicRoughnessTexture) {
			auto& tex = material.metallic_roughness;
			tex.index = textureTask->textures[pbr.metallicRoughnessTexture->textureIndex]->get_handle();
			tex.uv_set = static_cast<std::uint32_t>(pbr.metallicRoughnessTexture->texCoordIndex);
			if (auto& transform = pbr.metallicRoughnessTexture->transform; transform) {
				tex.uv_offset = glm::make_vec2(transform->uvOffset.data());
				tex.uv_scale = glm::make_vec2(transform->uvScale.data());
				tex.uv_rotation = transform->rotation;
			}
		}

		if (gltf_material.normalTexture) {
			material.normal_scale = gltf_material.normalTexture->scale;
			auto& tex = material.normal;
			tex.index = textureTask->textures[gltf_material.normalTexture->textureIndex]->get_handle();
			tex.uv_set = static_cast<std::uint32_t>(gltf_material.normalTexture->texCoordIndex);
			if (auto& transform = gltf_material.normalTexture->transform; transform) {
				tex.uv_offset = glm::make_vec2(transform->uvOffset.data());
				tex.uv_scale = glm::make_vec2(transform->uvScale.data());
				tex.uv_rotation = transform->rotation;
			}
		} else {
			material.normal_scale = 1.f;
		}

		std::lock_guard lock(material_mutex);
		materials[i] = renderer->create_material(material);
	}
}

class tangent_calculation {
	static int get_vertex_index(const SMikkTSpaceContext *context, int iFace, int iVert) {
		auto face_size = get_num_vertices_of_face(context, iFace);
		return (iFace * face_size) + iVert;
	}

	static int get_num_faces(const SMikkTSpaceContext *context) {
		auto& data = *static_cast<tangent_calculation*>(context->m_pUserData);
		return static_cast<int>(data.positions.size() / 3);
	}

	static int get_num_vertices_of_face(const SMikkTSpaceContext *context, int iFace) {
		return 3;
	}

	static void get_position(const SMikkTSpaceContext *context, float outpos[], int iFace, int iVert) {
		auto& data = *static_cast<tangent_calculation*>(context->m_pUserData);
		auto index = get_vertex_index(context, iFace, iVert);
		auto& vtx = data.positions[index];
		outpos[0] = vtx.x;
		outpos[1] = vtx.y;
		outpos[2] = vtx.z;
	}

	static void get_normal(const SMikkTSpaceContext *context, float outnormal[], int iFace, int iVert) {
		auto& data = *static_cast<tangent_calculation*>(context->m_pUserData);
		auto index = get_vertex_index(context, iFace, iVert);
#if VERTEX_BUFFER_NORMAL_ENCODING == 1
		auto vtx = shaders::normal_decode(glm::unpackSnorm2x16(data.vertices[index].normal));
#else
		auto& vtx = data.vertices[index].normal;
#endif
		outnormal[0] = vtx.x;
		outnormal[1] = vtx.y;
		outnormal[2] = vtx.z;
	}

	static void get_tex_coords(const SMikkTSpaceContext *context, float outuv[], int iFace, int iVert) {
		auto& data = *static_cast<tangent_calculation*>(context->m_pUserData);
		auto index = get_vertex_index(context, iFace, iVert);
		// TODO: Support multiple UV sets somehow? the gltf spec allows any number of uv sets to be used,
		// but the material specifies which one to use. Therefore, it'd be hard to precompute tangents here.
		auto& vtx = data.uvs[index];
		outuv[0] = vtx.x;
		outuv[1] = vtx.y;
	}

	static void set_tspace_basic(const SMikkTSpaceContext *context,
								 const float tangentu[],
								 float fSign, int iFace, int iVert) {
		auto& data = *static_cast<tangent_calculation*>(context->m_pUserData);
		auto index = get_vertex_index(context, iFace, iVert);
		auto& vtx = data.vertices[index];
#if VERTEX_BUFFER_TANGENT_ENCODING == 1
		vtx.tangent = shaders::tangent_encode(glm::vec4(glm::make_vec3(tangentu), -fSign));
#else
		vtx.tangent = glm::vec4(glm::make_vec3(tangentu), -fSign);
#endif
	}

	SMikkTSpaceInterface interface {
		.m_getNumFaces = get_num_faces,
		.m_getNumVerticesOfFace = get_num_vertices_of_face,
		.m_getPosition = get_position,
		.m_getNormal = get_normal,
		.m_getTexCoord = get_tex_coords,
		.m_setTSpaceBasic = set_tspace_basic,
	};
	SMikkTSpaceContext context {
		.m_pInterface = &interface,
	};

	std::span<glm::fvec3> positions;
	std::span<shaders::vertex_t> vertices;
	std::span<glm::fvec2> uvs;

public:
	explicit tangent_calculation(std::span<glm::fvec3> positions, std::span<shaders::vertex_t> vertices, std::span<glm::fvec2> uvs)
		: positions(positions), vertices(vertices), uvs(uvs) {
		context.m_pUserData = this;
	}

	void operator()() const {
		genTangSpaceDefault(&context);
	}
};

/**
 * Processes all glTF primitives into meshlets
 */
struct primitive_processing_task_t : ExceptionTaskSet {
	const fg::Asset& asset;
	const compressed_buffer_data_adapter_t& adapter;
	std::shared_ptr<graphics::renderer> renderer;

	std::mutex mesh_mutex;
	std::vector<mesh_primitive_indices> meshes;
	std::vector<std::shared_ptr<graphics::mesh_t>> primitives;

	enki::Dependency buffer_decompress_dependency;
	enki::Dependency material_dependency;

	explicit primitive_processing_task_t(const fg::Asset& _asset, const compressed_buffer_data_adapter_t& _adapter, std::shared_ptr<graphics::renderer> renderer) noexcept
			: asset(_asset), adapter(_adapter), renderer(std::move(renderer)) {
		ZoneScoped;
		m_SetSize = fg::max(1UZ, asset.meshes.size());
		meshes.resize(asset.meshes.size());
		primitives.reserve(meshes.size()); // Is definitely not enough, but we'll just live with it.
	}

	void process_primitive(std::uint64_t primitive_idx, const fg::Primitive& primitive);
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

glm::vec3 get_accessor_min_max(const decltype(fg::Accessor::min)& values) {
	return std::visit(fg::visitor {
		[](const auto& arg) {
			return glm::vec3();
		},
		[&](const FASTGLTF_STD_PMR_NS::vector<double>& values) {
			assert(values.size() == 3);
			return glm::fvec3(values[0], values[1], values[2]);
		},
		[&](const FASTGLTF_STD_PMR_NS::vector<std::int64_t>& values) {
			assert(values.size() == 3);
			return glm::fvec3(values[0], values[1], values[2]);
		},
	}, values);
}

static constexpr std::size_t baseline_stream_count = 2;

std::pair<std::size_t, std::array<meshopt_Stream, baseline_stream_count + shaders::max_uv_sets>> generate_geometry_stream(
	std::span<const glm::fvec3> positions, std::span<const shaders::vertex_t> vertices, std::array<std::span<const glm::fvec2>, shaders::max_uv_sets> uvs) {
	assert(positions.size() == vertices.size());

	std::size_t stream_count = baseline_stream_count;
	std::array<meshopt_Stream, baseline_stream_count + shaders::max_uv_sets> streams {{
		{ positions.data(), sizeof(glm::fvec3), sizeof(glm::fvec3) },
		{ vertices.data(), sizeof(shaders::vertex_t), sizeof(shaders::vertex_t) },
	}};
	for (auto& uv_buffer : uvs) {
		if (uv_buffer.empty())
			continue;

		assert(uv_buffer.size() == positions.size());
		streams[stream_count++] = { uv_buffer.data(), sizeof(glm::fvec2), sizeof(glm::fvec2) };
	}
	return std::make_pair(stream_count, streams);
}

void primitive_processing_task_t::process_primitive(std::uint64_t primitive_idx, const fg::Primitive& gltfPrimitive) {
	ZoneScoped;
	assert(gltfPrimitive.indicesAccessor.has_value());
	auto& idx_accessor = asset.accessors[gltfPrimitive.indicesAccessor.value()];

	std::vector indices(std::from_range, fastgltf::iterateAccessor<graphics::index_t>(asset, idx_accessor, adapter));

	// The code says this is possible, the spec says otherwise.
	auto* position_it = gltfPrimitive.findAttribute("POSITION");
	assert(position_it != gltfPrimitive.attributes.cend());
	auto& pos_accessor = asset.accessors[position_it->accessorIndex];

	glm::fvec3 aabb_center, aabb_extents;
	{
		auto min = get_accessor_min_max(pos_accessor.min);
		auto max = get_accessor_min_max(pos_accessor.max);
		aabb_center = (min + max) / 2.f;
		aabb_extents = max - aabb_center;
	}

	// We always re-generate vertex indices, and to save some reallocation time we preallocate the maximum
	// number here already.
	std::vector positions(std::from_range, fastgltf::iterateAccessor<glm::fvec3>(asset, pos_accessor, adapter));

	std::vector<shaders::vertex_t> vertices;
	vertices.resize(pos_accessor.count, shaders::vertex_t {
		.color = glm::u8vec4(255),
	});

	bool generated_normals = false;
	if (auto* normalAttribute = gltfPrimitive.findAttribute("NORMAL"); normalAttribute != gltfPrimitive.attributes.end()) {
		fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[normalAttribute->accessorIndex],
			[&](glm::vec3 val, std::size_t idx) {
#if VERTEX_BUFFER_NORMAL_ENCODING == 1
			vertices[idx].normal = glm::packSnorm2x16(shaders::normal_encode(glm::normalize(val)));
#else
			vertices[idx].normal = glm::normalize(val);
#endif
		}, adapter);
	} else {
		// Generate basic smooth vertex normals. As we quantize the normals, we have to store them first to normalize them afterwards.
		fastgltf::StaticVector normals(pos_accessor.count, glm::vec3(0.f));
		for (std::uint32_t idx = 0; idx < idx_accessor.count / 3; ++idx) {
			auto i = indices[idx * 3];
			auto i1 = indices[idx * 3 + 1];
			auto i2 = indices[idx * 3 + 2];

			auto v1 = positions[i1] - positions[i];
			auto v2 = positions[i2] - positions[i];
			auto val = glm::normalize(glm::cross(v1, v2));

			normals[i] += val;
			normals[i1] += val;
			normals[i2] += val;
		}
		for (std::size_t i = 0; i < normals.size(); ++i) {
#if VERTEX_BUFFER_NORMAL_ENCODING == 1
			vertices[i].normal = glm::packSnorm2x16(shaders::normal_encode(glm::normalize(normals[i])));
#else
			vertices[i].normal = glm::normalize(normals[i]);
#endif
		}
		generated_normals = true;
	}

	if (auto* colorAttribute = gltfPrimitive.findAttribute("COLOR_0"); colorAttribute != gltfPrimitive.attributes.end()) {
		// The glTF spec allows VEC3 and VEC4 for COLOR_n, with VEC3 data having to be extended with 1.0f for the fourth component.
		auto& colorAccessor = asset.accessors[colorAttribute->accessorIndex];
		if (colorAccessor.type == fastgltf::AccessorType::Vec4) {
			fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, colorAccessor, [&](glm::vec4 val, std::size_t idx) {
				val = glm::clamp(val, 0.0f, 1.0f);
				vertices[idx].color = glm::u8vec4(val * 255.f);
			}, adapter);
		} else if (colorAccessor.type == fastgltf::AccessorType::Vec3) {
			fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, colorAccessor, [&](glm::vec3 val, std::size_t idx) {
				val = glm::clamp(val, 0.0f, 1.0f);
				vertices[idx].color = glm::u8vec4(glm::vec4(val, 1.0f) * 255.f);
			}, adapter);
		}
	}

	std::array<std::vector<glm::fvec2>, shaders::max_uv_sets> uvs;

	for (const auto& [attribute, accessor_index] : gltfPrimitive.attributes) {
		// TODO: Also use this loop to load the other attributes, saving some iteration time?
		if (!attribute.starts_with("TEXCOORD_")) {
			continue;
		}

		auto strlen = []<typename char_t, std::size_t N>(const char_t (&)[N]) consteval {
			return N - 1;
		};

		// The spec requires indices to be just single positive decimal integers.
		static_assert(shaders::max_uv_sets <= 9, "The following code assumes single digit UV sets only.");
		auto idx = attribute[strlen("TEXCOORD_")] - '0';
		assert(idx < uvs.size() && idx >= 0);

		auto& accessor = asset.accessors[accessor_index];
		uvs[idx].reserve(accessor.count);
		for (auto uv : fastgltf::iterateAccessor<glm::fvec2>(asset, accessor)) {
			uvs[idx].emplace_back(uv);
		}
	}

	if (uvs[0].empty()) {
		// Initialize the first UV set to zeroes as a fallback
		uvs[0].resize(pos_accessor.count, glm::fvec2(0.f));
	}

	// the glTF spec requires us to re-generate tangents if we had to generate our own normals
	bool generated_tangents = false;
	if (auto* tangentAttribute = gltfPrimitive.findAttribute("TANGENT"); !generated_normals && tangentAttribute != gltfPrimitive.attributes.end()) {
		fastgltf::iterateAccessorWithIndex<glm::fvec4>(asset, asset.accessors[tangentAttribute->accessorIndex],
			[&](glm::fvec4 val, std::size_t idx) {
#if VERTEX_BUFFER_TANGENT_ENCODING == 1
			vertices[idx].tangent = shaders::tangent_encode(val);
#else
			vertices[idx].tangent = glm::vec4(glm::make_vec3(tangentu), -fSign);
#endif
		}, adapter);
		generated_tangents = true;
	}

	// De-index the geometry and use meshopt to generate a new index buffer, which possibly/hopefully is more optimal.
	{
		auto unindexed_vertex_count = idx_accessor.count;
		std::vector<glm::fvec3> unindexed_positions(unindexed_vertex_count);
		std::vector<shaders::vertex_t> unindexed_vertices(unindexed_vertex_count);
		std::array<std::vector<glm::fvec2>, shaders::max_uv_sets> unindexed_uvs;

		for (std::size_t i = 0; i < unindexed_vertex_count; ++i) {
			unindexed_positions[i] = positions[indices[i]];
			unindexed_vertices[i] = vertices[indices[i]];
		}

		for (auto [uv, unindexed_uv] : std::views::zip(uvs, unindexed_uvs))
			if (!uv.empty()) {
				unindexed_uv.resize(unindexed_vertex_count);
				for (std::size_t i = 0; i < unindexed_vertex_count; ++i)
					unindexed_uv[i] = uv[indices[i]];
			}

		if (!generated_tangents) {
			// Generate tangents using mikktspace. This requires unindexed geometry.
			// TODO: Using UV0 here is wrong, but this should still work most of the time and iirc the
			// sample viewer does so too. I have no idea how one would do it with respect to the material's
			// normal texture's texcoord index.
			tangent_calculation(unindexed_positions, unindexed_vertices, unindexed_uvs[0])();
		}

		const auto [stream_count, streams] = generate_geometry_stream(
			unindexed_positions, unindexed_vertices,
			transform_array(unindexed_uvs, [](const auto& buffer) { return std::span<const glm::fvec2>(buffer); }));

		// Generate the remap data. index count is equal to position count here, since it is unindexed
		std::size_t index_count = unindexed_positions.size();
		std::vector<unsigned int> remap(index_count);
		std::size_t vertex_count = meshopt_generateVertexRemapMulti(remap.data(), nullptr, index_count,
			index_count, streams.data(), stream_count);

		positions.resize(vertex_count);
		vertices.resize(vertex_count);

		// Finally, remap the index buffers and vertex buffers into their original buffers.
		meshopt_remapIndexBuffer(indices.data(), nullptr, index_count, remap.data());
		meshopt_remapVertexBuffer(positions.data(), unindexed_positions.data(), unindexed_vertex_count, sizeof(glm::fvec3), remap.data());
		meshopt_remapVertexBuffer(vertices.data(), unindexed_vertices.data(), unindexed_vertex_count, sizeof(shaders::vertex_t), remap.data());
		for (auto [uv, unindexed_uv] : std::views::zip(uvs, unindexed_uvs))
			if (!unindexed_uv.empty()) {
				uv.resize(vertex_count);
				meshopt_remapVertexBuffer(uv.data(), unindexed_uv.data(), unindexed_vertex_count, sizeof(glm::fvec2), remap.data());
			}
	}

	// Finally, apply some optimisation functions from meshopt
	meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), positions.size());

	auto* materialTask = dynamic_cast<const material_load_task_t*>(material_dependency.GetDependencyTask());
	auto materialIndex = gltfPrimitive.materialIndex.has_value()
		? materialTask->materials[*gltfPrimitive.materialIndex]
		: renderer->get_default_material_index();

	assert(0 < std::count_if(uvs.begin(), uvs.end(), [](auto& uv) { return !uv.empty(); }));
	auto mesh = renderer->create_shared_mesh(
			positions, vertices,
			transform_array(uvs, [](auto& buffer) { return std::span<const glm::fvec2>(buffer); }),
			indices,
			aabb_center, aabb_extents,
			materialIndex);

	{
		std::lock_guard lock(mesh_mutex);
		primitives[primitive_idx] = std::move(mesh);
	}
}

void primitive_processing_task_t::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.meshes.empty())
		return;

	for (auto i = range.start; i < range.end; ++i) {
		auto& gltf_mesh = asset.meshes[i];

		// Create the mesh/primitive CPU structures. We don't care about the order of the primitive buffer structures,
		// and therefore just append the primitives to the end of the vector. The mesh vector still needs to match the
		// order in the glTF asset though.
		{
			std::lock_guard lock(mesh_mutex);
			auto& mesh = meshes[i];
			mesh.primitive_indices.resize(gltf_mesh.primitives.size());

			for (std::size_t j = 0; j < gltf_mesh.primitives.size(); ++j) {
				mesh.primitive_indices[j] = primitives.size();
				primitives.emplace_back();
			}
		}

		for (std::size_t j = 0; auto& gltfPrimitive : gltf_mesh.primitives) {
			process_primitive(meshes[i].primitive_indices[j++], gltfPrimitive);
		}
	}
}

asset_load_task::asset_load_task(std::shared_ptr<graphics::renderer> renderer, fs::path path) : renderer(std::move(renderer)), asset_path(std::move(path)) {
	m_SetSize = 1;
}

std::shared_ptr<fastgltf::Asset> asset_load_task::loadGltf() {
	ZoneScoped;
	auto file = fg::MappedGltfFile::FromPath(asset_path);
	if (!bool(file)) {
		throw std::runtime_error("Failed to open glTF file");
	}

	static constexpr auto gltf_options = fg::Options::GenerateMeshIndices
		| fg::Options::DecomposeNodeMatrices
		| fg::Options::LoadExternalImages;

	static constexpr auto gltf_extensions = fg::Extensions::EXT_meshopt_compression
		| fg::Extensions::KHR_mesh_quantization
		| fg::Extensions::KHR_texture_transform
		| fg::Extensions::KHR_texture_basisu
		| fg::Extensions::MSFT_texture_dds
		| fg::Extensions::KHR_lights_punctual;

	fg::Parser parser(gltf_extensions);
	parser.setUserPointer(this);

	// Load the glTF
	auto loaded_asset = parser.loadGltf(file.get(), asset_path.parent_path(), gltf_options);
	if (!loaded_asset) {
		throw std::runtime_error(fmt::format("Failed to load glTF: {}", fg::getErrorMessage(loaded_asset.error())));
	}
	return std::make_shared<fg::Asset>(std::move(loaded_asset.get()));
}

void asset_load_task::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	asset = loadGltf();

	buffer_load_task_t buffer_load_task(*asset, asset_path.parent_path());

	compressed_buffer_data_adapter_t decompress_task(*asset);
	decompress_task.SetDependency(decompress_task.buffer_load_dependency, &buffer_load_task);

	image_load_task_t image_task(*asset, renderer);

	texture_create_task_t texture_task(*asset, renderer);
	texture_task.SetDependency(texture_task.image_load_dependency, &image_task);

	material_load_task_t material_task(*asset, renderer);
	material_task.SetDependency(material_task.texture_dependency, &texture_task);

	primitive_processing_task_t primitive_task(*asset, decompress_task, renderer);
	primitive_task.SetDependency(primitive_task.buffer_decompress_dependency, &decompress_task);
	primitive_task.SetDependency(primitive_task.material_dependency, &material_task);

	task_scheduler.AddTaskSetToPipe(&buffer_load_task);
	task_scheduler.AddTaskSetToPipe(&image_task);

	task_scheduler.WaitforTask(&decompress_task);

	animations.resize(asset->animations.size());
	for (std::size_t i = 0; i < asset->animations.size(); ++i) {
		auto& gltf_animation = asset->animations[i];
		auto& animation = animations[i];

		animation.channels.reserve(gltf_animation.channels.size());
		for (auto& channel : gltf_animation.channels) {
			animation.channels.emplace_back(channel);
		}

		animation.samplers.reserve(gltf_animation.samplers.size());
		for (auto& gltfSampler : gltf_animation.samplers) {
			auto& sampler = animation.samplers.emplace_back(*asset, gltfSampler);

			auto& inputAccessor = asset->accessors[gltfSampler.inputAccessor];
			sampler.input.resize(inputAccessor.count);
			fastgltf::copyFromAccessor<float>(*asset, inputAccessor, sampler.input.data(), decompress_task);

			auto& outputAccessor = asset->accessors[gltfSampler.outputAccessor];
			sampler.values.resize(outputAccessor.count * sampler.component_count);
			fastgltf::copyComponentsFromAccessor<float>(*asset, outputAccessor, sampler.values.data(), decompress_task);
		}
	}

	buffer_load_task.WaitForExceptionTask();
	primitive_task.WaitForExceptionTask();
	image_task.WaitForExceptionTask();
	task_scheduler.WaitforTask(&material_task);

	meshes = std::move(primitive_task.meshes);
	primitives = std::move(primitive_task.primitives);
	textures = std::move(texture_task.textures);
}
