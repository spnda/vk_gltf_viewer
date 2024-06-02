#include <utility>
#include <mutex>

#include <fmt/std.h>

#include <mesh_common.glsl.h>

#include <vk_gltf_viewer/assets.hpp>

#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <meshoptimizer.h>

namespace fs = std::filesystem;
namespace fg = fastgltf;

struct BufferLoadTask : ExceptionTaskSet {
	fg::Asset& asset;
	fs::path folder;

	explicit BufferLoadTask(fg::Asset& asset, fs::path folder) : asset(asset), folder(std::move(folder)) {
		m_SetSize = asset.buffers.size();
	}
	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};

void BufferLoadTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
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

		fg::StaticVector<std::uint8_t> data(buffer.byteLength);
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
				case fg::MeshoptCompressionMode::Attributes: {
					rc = meshopt_decodeVertexBuffer(result.data(), mc.count, mc.byteStride,
													reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
				case fg::MeshoptCompressionMode::Triangles: {
					rc = meshopt_decodeIndexBuffer(result.data(), mc.count, mc.byteStride,
											  reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
				case fg::MeshoptCompressionMode::Indices: {
					rc = meshopt_decodeIndexSequence(result.data(), mc.count, mc.byteStride,
												reinterpret_cast<const unsigned char*>(data.data()), mc.byteLength);
					break;
				}
			}

			//if (rc != 0)
			//	  return false;

			switch (mc.filter) {
				case fg::MeshoptCompressionFilter::None:
					break;
				case fg::MeshoptCompressionFilter::Octahedral: {
					meshopt_decodeFilterOct(result.data(), mc.count, mc.byteStride);
					break;
				}
				case fg::MeshoptCompressionFilter::Quaternion: {
					meshopt_decodeFilterQuat(result.data(), mc.count, mc.byteStride);
					break;
				}
				case fg::MeshoptCompressionFilter::Exponential: {
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
 * Processes all glTF primitives into meshlets
 */
struct PrimitiveProcessingTask : ExceptionTaskSet {
	const fg::Asset& asset;
	const CompressedBufferDataAdapter& adapter;
	std::reference_wrapper<Device> device;

	std::mutex meshMutex;
	std::vector<Mesh> meshes;
	std::vector<std::pair<PrimitiveBuffers, glsl::Primitive>> primitives;

	enki::Dependency bufferDecompressDependency;

	explicit PrimitiveProcessingTask(const fg::Asset& _asset, const CompressedBufferDataAdapter& _adapter, Device& _device) noexcept
			: asset(_asset), adapter(_adapter), device(_device) {
		ZoneScoped;
		m_SetSize = asset.meshes.size();
		meshes.resize(asset.meshes.size());
		primitives.reserve(meshes.size()); // Is definitely not enough, but we'll just live with it.
	}

	void createMeshBuffers(PrimitiveBuffers& buffers, std::size_t vertexIndexCount, std::size_t primitiveIndexCount,
						   std::size_t vertexCount, std::size_t meshletCount);
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

void PrimitiveProcessingTask::createMeshBuffers(
	PrimitiveBuffers& buffers, std::size_t vertexIndexCount, std::size_t primitiveIndexCount,
	std::size_t vertexCount, std::size_t meshletCount) {

	ZoneScoped;
	constexpr VmaAllocationCreateInfo allocationCreateInfo {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	{
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(std::uint32_t) * vertexIndexCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
					 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		buffers.vertexIndexBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferCreateInfo, &allocationCreateInfo);
	}

	{
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(std::uint8_t) * primitiveIndexCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
					 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		buffers.primitiveIndexBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferCreateInfo, &allocationCreateInfo);
	}

	{
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(glsl::Vertex) * vertexCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
					 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		buffers.vertexBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferCreateInfo, &allocationCreateInfo);
	}

	{
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(glsl::Meshlet) * meshletCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
					 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		buffers.meshletBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferCreateInfo, &allocationCreateInfo);
	}

}

void PrimitiveProcessingTask::processPrimitive(std::uint64_t primitiveIdx, const fg::Primitive& gltfPrimitive) {
	ZoneScoped;
	glsl::Primitive primitive;
	if (gltfPrimitive.materialIndex.has_value()) {
		primitive.materialIndex = static_cast<std::uint32_t>(gltfPrimitive.materialIndex.value());
	}

	// The code says this is possible, the spec says otherwise.
	auto* positionIt = gltfPrimitive.findAttribute("POSITION");
	assert(positionIt != gltfPrimitive.attributes.cend());

	auto& posAccessor = asset.accessors[positionIt->second];

	{
		auto primitiveMin = getAccessorMinMax(posAccessor.min);
		auto primitiveMax = getAccessorMinMax(posAccessor.max);
		primitive.aabbCenter = (primitiveMin + primitiveMax) / 2.f;
		primitive.aabbExtents = primitiveMax - primitive.aabbCenter;
	}

	// Load the vertices. TODO: Directly copy into the staging buffer, instead of into a vector first.
	std::vector<glsl::Vertex> vertices; vertices.reserve(posAccessor.count);
	fastgltf::iterateAccessor<glm::vec3>(asset, posAccessor, [&](glm::vec3 val) {
		auto& vtx = vertices.emplace_back();
		vtx.position = val;
	}, adapter);

	assert(gltfPrimitive.indicesAccessor.has_value());
	auto& idxAccessor = asset.accessors[gltfPrimitive.indicesAccessor.value()];

	std::vector<std::uint32_t> indices(idxAccessor.count);
	fastgltf::copyFromAccessor<std::uint32_t>(asset, idxAccessor, indices.data(), adapter);

	// Generate the meshlets
	constexpr float coneWeight = 0.0f; // We leave this as 0 because we're not using cluster cone culling.
	constexpr auto maxPrimitives = fastgltf::alignDown(glsl::maxPrimitives, 4U); // meshopt requires the primitive count to be aligned to 4.
	std::size_t maxMeshlets = meshopt_buildMeshletsBound(idxAccessor.count, glsl::maxVertices, maxPrimitives);
	std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
	std::vector<std::uint32_t> meshletVertices(maxMeshlets * glsl::maxVertices);
	std::vector<std::uint8_t> meshletTriangles(maxMeshlets * maxPrimitives * 3);

	{
		primitive.meshletCount = meshopt_buildMeshlets(
			meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
			indices.data(), indices.size(),
			&vertices[0].position.x, vertices.size(), sizeof(decltype(vertices)::value_type),
			glsl::maxVertices, maxPrimitives, coneWeight);

		const auto& lastMeshlet = meshlets[primitive.meshletCount - 1];
		meshletVertices.resize(lastMeshlet.vertex_count + lastMeshlet.vertex_offset);
		meshletTriangles.resize(((lastMeshlet.triangle_count * 3 + 3) & ~3) + lastMeshlet.triangle_offset);
		meshlets.resize(primitive.meshletCount);
	}

	std::vector<glsl::Meshlet> optimisedMeshlets; optimisedMeshlets.reserve(primitive.meshletCount);
	for (auto& meshlet : meshlets) {
		meshopt_optimizeMeshlet(&meshletVertices[meshlet.vertex_offset], &meshletTriangles[meshlet.triangle_offset],
								meshlet.triangle_count, meshlet.vertex_count);

		// Compute meshlet bounds
		auto& initialVertex = vertices[meshletVertices[meshlet.vertex_offset]];
		auto min = glm::vec3(initialVertex.position), max = glm::vec3(initialVertex.position);

		for (std::size_t i = 1; i < meshlet.vertex_count; ++i) {
			std::uint32_t vertexIndex = meshletVertices[meshlet.vertex_offset + i];
			auto& vertex = vertices[vertexIndex];

			// The glm::min and glm::max functions are all component-wise.
			min = glm::min(min, vertex.position);
			max = glm::max(max, vertex.position);
		}

		// We can convert the count variables to a uint8_t since glsl::maxVertices and glsl::maxPrimitives both fit in 8-bits.
		assert(meshlet.vertex_count <= std::numeric_limits<std::uint8_t>::max());
		assert(meshlet.triangle_count <= std::numeric_limits<std::uint8_t>::max());
		auto center = (min + max) * 0.5f;
		optimisedMeshlets.emplace_back(glsl::Meshlet {
			.vertexOffset = meshlet.vertex_offset,
			.triangleOffset = meshlet.triangle_offset,
			.vertexCount = static_cast<std::uint8_t>(meshlet.vertex_count),
			.triangleCount = static_cast<std::uint8_t>(meshlet.triangle_count),
			.aabbExtents = max - center,
			.aabbCenter = center,
		});
	}

	PrimitiveBuffers buffers;
	buffers.meshletCount = primitive.meshletCount;
	createMeshBuffers(buffers, meshletVertices.size(), meshletTriangles.size(), vertices.size(), optimisedMeshlets.size());

	primitive.vertexIndexBuffer = buffers.vertexIndexBuffer->getDeviceAddress();
	primitive.primitiveIndexBuffer = buffers.primitiveIndexBuffer->getDeviceAddress();
	primitive.vertexBuffer = buffers.vertexBuffer->getDeviceAddress();
	primitive.meshletBuffer = buffers.meshletBuffer->getDeviceAddress();

	// Create staging allocations and upload all buffers
	{
		PrimitiveBuffers stagingBuffers;
		stagingBuffers.vertexIndexBuffer = device.get().createHostStagingBuffer(buffers.vertexIndexBuffer->getBufferSize());
		stagingBuffers.primitiveIndexBuffer = device.get().createHostStagingBuffer(buffers.primitiveIndexBuffer->getBufferSize());
		stagingBuffers.vertexBuffer = device.get().createHostStagingBuffer(buffers.vertexBuffer->getBufferSize());
		stagingBuffers.meshletBuffer = device.get().createHostStagingBuffer(buffers.meshletBuffer->getBufferSize());

		{
			ScopedMap map(*stagingBuffers.vertexIndexBuffer);
			std::memcpy(map.get(), meshletVertices.data(), stagingBuffers.vertexIndexBuffer->getBufferSize());
		}
		{
			ScopedMap map(*stagingBuffers.primitiveIndexBuffer);
			std::memcpy(map.get(), meshletTriangles.data(), stagingBuffers.primitiveIndexBuffer->getBufferSize());
		}
		{
			ScopedMap map(*stagingBuffers.vertexBuffer);
			std::memcpy(map.get(), vertices.data(), stagingBuffers.vertexBuffer->getBufferSize());
		}
		{
			ScopedMap map(*stagingBuffers.meshletBuffer);
			std::memcpy(map.get(), optimisedMeshlets.data(), stagingBuffers.meshletBuffer->getBufferSize());
		}

		device.get().immediateSubmit(device.get().getNextTransferQueueHandle(),
									 device.get().uploadCommandPools[taskScheduler.GetThreadNum()],
									 [&](auto cmd) {
			const VkBufferCopy vertexIndexRegion { .size = buffers.vertexIndexBuffer->getBufferSize(), };
			vkCmdCopyBuffer(cmd, stagingBuffers.vertexIndexBuffer->getHandle(), buffers.vertexIndexBuffer->getHandle(), 1, &vertexIndexRegion);

			const VkBufferCopy primitiveIndexRegion { .size = buffers.primitiveIndexBuffer->getBufferSize(), };
			vkCmdCopyBuffer(cmd, stagingBuffers.primitiveIndexBuffer->getHandle(), buffers.primitiveIndexBuffer->getHandle(), 1, &primitiveIndexRegion);

			const VkBufferCopy vertexRegion { .size = buffers.vertexBuffer->getBufferSize(), };
			vkCmdCopyBuffer(cmd, stagingBuffers.vertexBuffer->getHandle(), buffers.vertexBuffer->getHandle(), 1, &vertexRegion);

			const VkBufferCopy meshletRegion { .size = buffers.meshletBuffer->getBufferSize(), };
			vkCmdCopyBuffer(cmd, stagingBuffers.meshletBuffer->getHandle(), buffers.meshletBuffer->getHandle(), 1, &meshletRegion);
		});
	}

	{
		std::lock_guard lock(meshMutex);
		primitives[primitiveIdx] = std::make_pair(std::move(buffers), primitive);
	}
}

void PrimitiveProcessingTask::ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) {
	ZoneScoped;
	for (auto i = range.start; i < range.end; ++i) {
		auto& gltfMesh = asset.meshes[i];

		// Create the mesh/primitive CPU structures. We don't care about the order of the primitive buffer structures,
		// and therefore just append the primitives to the end of the vector. The mesh vector still needs to match the
		// order in the glTF asset though.
		{
			std::lock_guard lock(meshMutex);
			auto& mesh = meshes[i];
			mesh.primitiveIndices.resize(gltfMesh.primitives.size());

			for (std::size_t j = 0; auto& gltfPrimitive : gltfMesh.primitives) {
				mesh.primitiveIndices[j++] = primitives.size();
				primitives.emplace_back();
			}
		}

		for (std::size_t j = 0; auto& gltfPrimitive : gltfMesh.primitives) {
			processPrimitive(meshes[i].primitiveIndices[j++], gltfPrimitive);
		}
	}
}

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

AssetLoadTask::AssetLoadTask(Device& device, fs::path path) : device(device), assetPath(std::move(path)) {
	m_SetSize = 1;
}

std::shared_ptr<fastgltf::Asset> AssetLoadTask::loadGltf() {
	ZoneScoped;
	fg::GltfFileStream file(assetPath);
	if (!file.isOpen()) {
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
	auto loadedAsset = parser.loadGltf(file, assetPath.parent_path(), gltfOptions);
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

	PrimitiveProcessingTask primitiveTask(*asset, bufferDecompressTask, device.get());
	primitiveTask.SetDependency(primitiveTask.bufferDecompressDependency, &bufferDecompressTask);

	taskScheduler.AddTaskSetToPipe(&bufferLoadTask);

	ImageLoadTask imageLoadTask;
	taskScheduler.AddTaskSetToPipe(&imageLoadTask);

	taskScheduler.WaitforTask(&primitiveTask);

	meshes = std::move(primitiveTask.meshes);
	primitives = std::move(primitiveTask.primitives);

	animations.resize(asset->animations.size());
	for (std::size_t i = 0; i < asset->animations.size(); ++i) {
		auto& gltfAnimation = asset->animations[i];
		auto& animation = animations[i];


		animation.channels.reserve(gltfAnimation.channels.size());
		for (auto& channel : gltfAnimation.channels) {
			animation.channels.emplace_back(channel);
		}

		animation.samplers.reserve(gltfAnimation.samplers.size());
		for (auto& sampler : gltfAnimation.samplers) {
			animation.samplers.emplace_back(asset, sampler);
		}
	}

	taskScheduler.WaitforTask(&imageLoadTask);
}
