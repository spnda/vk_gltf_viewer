#pragma once

#include <mutex>
#include <utility>
#include <vector>

#include <vulkan/vk.hpp>

#if defined(VKV_METAL)
#include <Foundation/NSSharedPtr.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLBuffer.hpp>
#endif

#include <resource_table.h.glsl>

struct Device;

namespace graphics {
	class ResourceTable {
	protected:
		/** Each bit of each integer represents a boolean whether that array element is free or not */
		std::vector<std::uint64_t> sampledImageBitmap;
		std::vector<std::uint64_t> storageImageBitmap;
		std::mutex bitmapMutex;

		glsl::ResourceTableHandle findFirstFreeHandle(std::vector<std::uint64_t> &bitmap);

		void freeHandle(std::vector<std::uint64_t> &bitmap, glsl::ResourceTableHandle handle);

	public:
		explicit ResourceTable() = default;
		virtual ~ResourceTable() noexcept = default;

		void removeStorageImageHandle(glsl::ResourceTableHandle handle) noexcept;
		void removeSampledImageHandle(glsl::ResourceTableHandle handle) noexcept;
	};

	namespace vulkan {
		class VkResourceTable : public ResourceTable {
			std::reference_wrapper<Device> device;

			VkDescriptorPool pool = VK_NULL_HANDLE;
			VkDescriptorSetLayout layout = VK_NULL_HANDLE;
			VkDescriptorSet set = VK_NULL_HANDLE;

		public:
			explicit VkResourceTable(Device& device);
			~VkResourceTable() noexcept override;

			[[nodiscard]] auto getLayout() const noexcept -> const VkDescriptorSetLayout& {
				return layout;
			}

			[[nodiscard]] auto getSet() const noexcept -> const VkDescriptorSet& {
				return set;
			}

			[[nodiscard]] glsl::ResourceTableHandle allocateStorageImage(VkImageView view, VkImageLayout imageLayout) noexcept;
			[[nodiscard]] glsl::ResourceTableHandle allocateSampledImage(VkImageView view, VkImageLayout imageLayout, VkSampler sampler) noexcept;
		};
	}

#if defined(VKV_METAL)
	namespace metal {
		class MtlResourceTable : public ResourceTable {
			struct SampledTextureEntry {
				MTL::ResourceID tex;
				MTL::ResourceID sampler;
			};

			NS::SharedPtr<MTL::Device> device;

		public:
			MTL::Buffer* sampledImageBuffer = nullptr;
			MTL::Buffer* storageImageBuffer = nullptr;

		public:
			explicit MtlResourceTable(NS::SharedPtr<MTL::Device> device);
			~MtlResourceTable() noexcept override;

			[[nodiscard]] glsl::ResourceTableHandle allocateStorageImage(MTL::Texture* texture) noexcept;
			[[nodiscard]] glsl::ResourceTableHandle allocateSampledImage(MTL::Texture* texture, MTL::SamplerState* sampler) noexcept;
		};
	}
#endif
} // namespace graphics
