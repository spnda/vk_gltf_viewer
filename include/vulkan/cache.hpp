#include <fstream>
#include <filesystem>

#include <TaskScheduler.h>
#include <tracy/Tracy.hpp>

#include <fastgltf/types.hpp>

#include <vulkan/vk.hpp>

namespace fs = std::filesystem;

namespace vk {
	/**
	 * enkiTS task that asynchronously loads a file into a VkPipelineCache object.
	 */
	class PipelineCacheLoadTask : public enki::ITaskSet {
		VkDevice device;
		VkPipelineCache* cache;
		fs::path cachePath;

		VkResult result = VK_SUCCESS;

		VkResult createCache(std::size_t size, void* data) {
			ZoneScoped;
			const VkPipelineCacheCreateInfo cacheInfo = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
				.initialDataSize = size,
				.pInitialData = data,
			};
			return vkCreatePipelineCache(device, &cacheInfo, vk::allocationCallbacks.get(), cache);
		}

	public:
		explicit PipelineCacheLoadTask(VkDevice device, VkPipelineCache* cache, fs::path cachePath) :
			device(device), cache(cache), cachePath(std::move(cachePath)) {}

		VkResult getResult() const { return result; }

		void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override {
			ZoneScoped;
			std::ifstream cacheFile(cachePath, std::ios::binary | std::ios::ate);
			if (!cacheFile.is_open() || cacheFile.fail()) {
				result = createCache(0, nullptr);
				return;
			}

			fastgltf::StaticVector<std::byte> bytes(cacheFile.tellg());
			cacheFile.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
			result = createCache(bytes.size(), bytes.data());
		}
	};

	/**
	 * enkiTS task that saves data from a VkPipelineCache to a file.
	 */
	class PipelineCacheSaveTask : public enki::ITaskSet {
		VkDevice device;
		VkPipelineCache* cache;
		fs::path cachePath;

		bool success = false;

	public:
		explicit PipelineCacheSaveTask(VkDevice device, VkPipelineCache* cache, fs::path cachePath) :
			device(device), cache(cache), cachePath(std::move(cachePath)) {}

		bool didSucceed() {
			return success;
		}

		void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override {
			ZoneScoped;
			if (*cache == VK_NULL_HANDLE) {
				return;
			}

			if (!fs::exists(cachePath.parent_path())) {
				fs::create_directory(cachePath.parent_path());
			}

			std::ofstream cacheFile(cachePath, std::ios::binary);
			if (!cacheFile.is_open() || cacheFile.fail()) {
				success = false;
				return;
			}

			std::size_t size = 0;
			auto result = vkGetPipelineCacheData(device, *cache, &size, nullptr);
			if (result != VK_SUCCESS) {
				success = false;
				return;
			}

			fastgltf::StaticVector<std::byte> bytes(size);
			result = vkGetPipelineCacheData(device, *cache, &size, bytes.data());
			if (result != VK_SUCCESS) {
				success = false;
				return;
			}

			cacheFile.write(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		}
	};
} // namespace vk