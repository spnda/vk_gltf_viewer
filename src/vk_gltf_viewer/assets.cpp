#include <utility>
#include <mutex>

#include <fmt/std.h>

#include <mesh_common.h>

#include <vk_gltf_viewer/assets.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <meshoptimizer.h>

namespace fs = std::filesystem;
namespace fg = fastgltf;

#if defined(_MSC_VER) && !defined(__clang__)
std::size_t operator""UZ(unsigned long long int x) noexcept {
	return std::size_t(x);
}
#endif

struct BufferLoadTask : ExceptionTaskSet {
	fg::Asset& asset;
	fs::path folder;

	explicit BufferLoadTask(fg::Asset& asset, fs::path folder) : asset(asset), folder(std::move(folder)) {
		m_SetSize = fg::max(1UZ, asset.buffers.size());
	}
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void BufferLoadTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
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
struct CompressedBufferDataAdapter : enki::ITaskSet {
	fg::Asset& asset;

	std::vector<std::optional<fastgltf::StaticVector<std::byte>>> decompressedBuffers;

	enki::Dependency bufferLoadDependency;

	explicit CompressedBufferDataAdapter(fg::Asset& asset) : asset(asset) {
		m_SetSize = asset.bufferViews.size();
		m_MinRange = fastgltf::min(24U, m_SetSize);
		decompressedBuffers.resize(m_SetSize);
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
				return span(reinterpret_cast<const std::byte*>(array.bytes.data()), array.bytes.size_bytes());
			},
			[&](const sources::Vector& vec) -> span<const std::byte> {
				return span(reinterpret_cast<const std::byte*>(vec.bytes.data()), vec.bytes.size());
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
			auto& bufferView = asset.bufferViews[i];
			if (!bufferView.meshoptCompression) {
				continue;
			}

			// This is a compressed buffer view.
			// For the original implementation, see https://github.com/jkuhlmann/cgltf/pull/129#issue-739550034
			auto& mc = *bufferView.meshoptCompression;
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

			decompressedBuffers[i] = std::move(result);
		}
	}

	auto operator()([[maybe_unused]] const fastgltf::Asset& _, std::size_t bufferViewIdx) const {
		ZoneScoped;
		using namespace fastgltf;

		auto& bufferView = asset.bufferViews[bufferViewIdx];
		if (bufferView.meshoptCompression) {
			assert(decompressedBuffers.size() == asset.bufferViews.size());

			assert(decompressedBuffers[bufferViewIdx].has_value());
			return span(decompressedBuffers[bufferViewIdx]->data(), decompressedBuffers[bufferViewIdx]->size_bytes());
		}

		return getData(asset.buffers[bufferView.bufferIndex], bufferView.byteOffset, bufferView.byteLength);
	}
};

/**
 * Loads all glTF images into Vulkan images
 */
struct ImageLoadTask : ExceptionTaskSet {
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void ImageLoadTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	for (auto i = range.start; i < range.end; ++i) {
	}
}

struct MaterialLoadTask : enki::ITaskSet {
	const fg::Asset& asset;
	std::shared_ptr<graphics::Renderer> renderer;
	std::vector<graphics::MaterialIndex> materials;

	enki::Dependency imageLoadDependency;

	explicit MaterialLoadTask(const fg::Asset& asset, ImageLoadTask& imageLoadTask, std::shared_ptr<graphics::Renderer> renderer) noexcept : asset(asset), renderer(std::move(renderer)) {
		materials.resize(asset.materials.size() + 1);
		m_SetSize = fg::max(1UZ, asset.materials.size());
		m_MinRange = fg::min(16U, m_SetSize);
	}

	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void MaterialLoadTask::ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.materials.empty())
		return;

	for (auto i = range.start; i < range.end; ++i) {
		auto& gltfMaterial = asset.materials[i];
		auto& pbr = gltfMaterial.pbrData;

		materials[i] = renderer->createMaterial(shaders::Material {
			.albedoFactor = glm::make_vec4(pbr.baseColorFactor.data()),
			.alphaCutoff = gltfMaterial.alphaCutoff,
			.doubleSided = gltfMaterial.doubleSided,
		});
	}
}

/**
 * Processes all glTF primitives into meshlets
 */
struct PrimitiveProcessingTask : ExceptionTaskSet {
	const fg::Asset& asset;
	const CompressedBufferDataAdapter& adapter;
	std::shared_ptr<graphics::Renderer> renderer;

	std::mutex meshMutex;
	std::vector<Mesh> meshes;
	std::vector<std::shared_ptr<graphics::Mesh>> primitives;

	enki::Dependency bufferDecompressDependency;
	enki::Dependency materialDependency;

	explicit PrimitiveProcessingTask(const fg::Asset& _asset, const CompressedBufferDataAdapter& _adapter, std::shared_ptr<graphics::Renderer> renderer) noexcept
			: asset(_asset), adapter(_adapter), renderer(std::move(renderer)) {
		ZoneScoped;
		m_SetSize = fg::max(1UZ, asset.meshes.size());
		meshes.resize(asset.meshes.size());
		primitives.reserve(meshes.size()); // Is definitely not enough, but we'll just live with it.
	}

	void processPrimitive(std::uint64_t primitiveIdx, const fg::Primitive& primitive);
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

glm::vec3 getAccessorMinMax(const decltype(fg::Accessor::min)& values) {
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

void PrimitiveProcessingTask::processPrimitive(std::uint64_t primitiveIdx, const fg::Primitive& gltfPrimitive) {
	ZoneScoped;
	// The code says this is possible, the spec says otherwise.
	auto* positionIt = gltfPrimitive.findAttribute("POSITION");
	assert(positionIt != gltfPrimitive.attributes.cend());

	auto& posAccessor = asset.accessors[positionIt->accessorIndex];

	glm::fvec3 aabbCenter, aabbExtents;
	{
		auto primitiveMin = getAccessorMinMax(posAccessor.min);
		auto primitiveMax = getAccessorMinMax(posAccessor.max);
		aabbCenter = (primitiveMin + primitiveMax) / 2.f;
		aabbExtents = primitiveMax - aabbCenter;
	}

	// Load the vertices. TODO: Directly copy into the staging buffer, instead of into a vector first.
	std::vector<shaders::Vertex> vertices; vertices.reserve(posAccessor.count);
	fastgltf::iterateAccessor<glm::vec3>(asset, posAccessor, [&](glm::vec3 val) {
		auto& vtx = vertices.emplace_back();
		vtx.position = val;
	}, adapter);

	assert(gltfPrimitive.indicesAccessor.has_value());
	auto& idxAccessor = asset.accessors[gltfPrimitive.indicesAccessor.value()];

	std::vector<graphics::index_t> indices(idxAccessor.count);
	fastgltf::copyFromAccessor<graphics::index_t>(
			asset, idxAccessor, indices.data(), adapter);

	auto* materialTask = dynamic_cast<const MaterialLoadTask*>(materialDependency.GetDependencyTask());
	auto materialIndex = gltfPrimitive.materialIndex.has_value()
		? materialTask->materials[*gltfPrimitive.materialIndex]
		: renderer->getDefaultMaterialIndex();

	auto mesh = renderer->createSharedMesh(
			vertices, indices, aabbCenter, aabbExtents,
			materialIndex);

	{
		std::lock_guard lock(meshMutex);
		primitives[primitiveIdx] = std::move(mesh);
	}
}

void PrimitiveProcessingTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	if (asset.meshes.empty())
		return;

	for (auto i = range.start; i < range.end; ++i) {
		auto& gltfMesh = asset.meshes[i];

		// Create the mesh/primitive CPU structures. We don't care about the order of the primitive buffer structures,
		// and therefore just append the primitives to the end of the vector. The mesh vector still needs to match the
		// order in the glTF asset though.
		{
			std::lock_guard lock(meshMutex);
			auto& mesh = meshes[i];
			mesh.primitiveIndices.resize(gltfMesh.primitives.size());

			for (std::size_t j = 0; j < gltfMesh.primitives.size(); ++j) {
				mesh.primitiveIndices[j] = primitives.size();
				primitives.emplace_back();
			}
		}

		for (std::size_t j = 0; auto& gltfPrimitive : gltfMesh.primitives) {
			processPrimitive(meshes[i].primitiveIndices[j++], gltfPrimitive);
		}
	}
}

AssetLoadTask::AssetLoadTask(std::shared_ptr<graphics::Renderer> renderer, fs::path path) : renderer(std::move(renderer)), assetPath(std::move(path)) {
	m_SetSize = 1;
}

std::shared_ptr<fastgltf::Asset> AssetLoadTask::loadGltf() {
	ZoneScoped;
	auto file = fg::MappedGltfFile::FromPath(assetPath);
	if (!bool(file)) {
		throw std::runtime_error("Failed to open glTF file");
	}

	static constexpr auto gltfOptions = fg::Options::GenerateMeshIndices
		| fg::Options::DecomposeNodeMatrices;

	static constexpr auto gltfExtensions = fg::Extensions::EXT_meshopt_compression
		| fg::Extensions::KHR_mesh_quantization
		| fg::Extensions::KHR_texture_transform
		| fg::Extensions::KHR_texture_basisu
		| fg::Extensions::MSFT_texture_dds
		| fg::Extensions::KHR_lights_punctual;

	fg::Parser parser(gltfExtensions);
	parser.setUserPointer(this);

	// Load the glTF
	auto loadedAsset = parser.loadGltf(file.get(), assetPath.parent_path(), gltfOptions);
	if (!loadedAsset) {
		throw std::runtime_error(fmt::format("Failed to load glTF: {}", fg::getErrorMessage(loadedAsset.error())));
	}
	return std::make_shared<fg::Asset>(std::move(loadedAsset.get()));
}

void AssetLoadTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	asset = loadGltf();

	BufferLoadTask bufferLoadTask(*asset, assetPath.parent_path());

	CompressedBufferDataAdapter bufferDecompressTask(*asset);
	bufferDecompressTask.SetDependency(bufferDecompressTask.bufferLoadDependency, &bufferLoadTask);

	ImageLoadTask imageLoadTask;

	MaterialLoadTask materialLoadTask(*asset, imageLoadTask, renderer);
	materialLoadTask.SetDependency(materialLoadTask.imageLoadDependency, &imageLoadTask);

	PrimitiveProcessingTask primitiveTask(*asset, bufferDecompressTask, renderer);
	primitiveTask.SetDependency(primitiveTask.bufferDecompressDependency, &bufferDecompressTask);
	primitiveTask.SetDependency(primitiveTask.materialDependency, &materialLoadTask);

	taskScheduler.AddTaskSetToPipe(&bufferLoadTask);
	taskScheduler.AddTaskSetToPipe(&imageLoadTask);

	taskScheduler.WaitforTask(&bufferDecompressTask);
	animations.resize(asset->animations.size());
	for (std::size_t i = 0; i < asset->animations.size(); ++i) {
		auto& gltfAnimation = asset->animations[i];
		auto& animation = animations[i];

		animation.channels.reserve(gltfAnimation.channels.size());
		for (auto& channel : gltfAnimation.channels) {
			animation.channels.emplace_back(channel);
		}

		animation.samplers.reserve(gltfAnimation.samplers.size());
		for (auto& gltfSampler : gltfAnimation.samplers) {
			auto& sampler = animation.samplers.emplace_back(*asset, gltfSampler);

			auto& inputAccessor = asset->accessors[gltfSampler.inputAccessor];
			sampler.input.resize(inputAccessor.count);
			fastgltf::copyFromAccessor<float>(*asset, inputAccessor, sampler.input.data(), bufferDecompressTask);

			auto& outputAccessor = asset->accessors[gltfSampler.outputAccessor];
			sampler.values.resize(outputAccessor.count * sampler.componentCount);
			fastgltf::copyComponentsFromAccessor<float>(*asset, outputAccessor, sampler.values.data(), bufferDecompressTask);
		}
	}

	taskScheduler.WaitforTask(&primitiveTask);
	taskScheduler.WaitforTask(&imageLoadTask);
	taskScheduler.WaitforTask(&materialLoadTask);

	meshes = std::move(primitiveTask.meshes);
	primitives = std::move(primitiveTask.primitives);
}
