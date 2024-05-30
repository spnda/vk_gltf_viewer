#pragma once

#include <span>
#include <filesystem>

#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/buffer.hpp>

#include <fastgltf/core.hpp>

/** The buffer handles corresponding to the buffers in each glsl::Primitive. */
struct PrimitiveBuffers {
	std::unique_ptr<ScopedBuffer> vertexIndexBuffer;
	std::unique_ptr<ScopedBuffer> primitiveIndexBuffer;
	std::unique_ptr<ScopedBuffer> vertexBuffer;
	std::unique_ptr<ScopedBuffer> meshletBuffer;

	std::uint32_t meshletCount;
};

struct Mesh {
	std::vector<std::uint64_t> primitiveIndices;
};

struct Asset {
	fastgltf::Asset asset;

	std::vector<Mesh> meshes;
	std::vector<PrimitiveBuffers> primitiveBuffers;
	std::unique_ptr<ScopedBuffer> primitiveBuffer;
};

class AssetLoadTask : public ExceptionTaskSet {
	std::reference_wrapper<Device> device;
	std::filesystem::path assetPath;

	std::unique_ptr<Asset> loadedAsset;

public:
	explicit AssetLoadTask(Device& device, std::filesystem::path path);

	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;

	std::unique_ptr<Asset> getLoadedAsset() noexcept {
		return std::move(loadedAsset);
	}
};
