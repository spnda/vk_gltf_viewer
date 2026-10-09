#pragma once

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vulkan/vk.hpp>

#if defined(VKV_METAL)
#include <Foundation/NSSharedPtr.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLCommandEncoder.hpp>
#endif

#include <resource_table.h>

struct Device;

namespace graphics {
	class resource_table {
	protected:
		/** Each bit of each integer represents a boolean whether that array element is free or not */
		std::vector<std::uint64_t> sampled_image_bitmap;
		std::mutex bitmap_mutex;

		shaders::resource_table_handle_t find_first_free_handle(std::vector<std::uint64_t> &bitmap);

		void free_handle(std::vector<std::uint64_t> &bitmap, shaders::resource_table_handle_t handle);

	public:
		explicit resource_table() = default;
		virtual ~resource_table() noexcept = default;

		virtual void remove_sampled_image_handle(shaders::resource_table_handle_t handle) noexcept;
	};

	namespace vulkan {
		class vk_resource_table : public resource_table {
			std::reference_wrapper<Device> device;

			std::vector<std::uint64_t> storage_image_bitmap;

			VkDescriptorPool pool = VK_NULL_HANDLE;
			VkDescriptorSetLayout layout = VK_NULL_HANDLE;
			VkDescriptorSet set = VK_NULL_HANDLE;

		public:
			explicit vk_resource_table(Device& device);
			~vk_resource_table() noexcept override;

			[[nodiscard]] auto get_layout() const noexcept -> const VkDescriptorSetLayout& {
				return layout;
			}

			[[nodiscard]] auto get_set() const noexcept -> const VkDescriptorSet& {
				return set;
			}

			[[nodiscard]] shaders::resource_table_handle_t allocate_storage_image(VkImageView view, VkImageLayout imageLayout) noexcept;
			[[nodiscard]] shaders::resource_table_handle_t allocate_sampled_image(VkImageView view, VkImageLayout imageLayout, VkSampler sampler) noexcept;

			void remove_storage_image_handle(shaders::resource_table_handle_t handle) noexcept;
		};
	}

#if defined(VKV_METAL)
	namespace metal {
		class mtl_resource_table : public resource_table {
			struct sampled_texture_entry {
				MTL::ResourceID tex;
				MTL::ResourceID sampler;
			};

			NS::SharedPtr<MTL::Device> device;

			/** List of resources to make resident */
			std::unordered_map<shaders::resource_table_handle_t, NS::SharedPtr<MTL::Resource>> resources;
			std::mutex resource_mutex;

		public:
			MTL::Buffer* sampled_image_buffer = nullptr;

		public:
			explicit mtl_resource_table(NS::SharedPtr<MTL::Device> device);
			~mtl_resource_table() noexcept override;

			[[nodiscard]] shaders::resource_table_handle_t allocate_sampled_image(NS::SharedPtr<MTL::Texture> texture, NS::SharedPtr<MTL::SamplerState> sampler) noexcept;

			void remove_sampled_image_handle(shaders::resource_table_handle_t handle) noexcept override;

			template <typename T>
			void encode_usage(T* encoder) {
				ZoneScoped;
				std::lock_guard lock(resource_mutex);
				for (auto& [_, resource] : resources) {
					assert(resource.get() != nullptr);
					encoder->useResource(resource.get(), MTL::ResourceUsageRead);
				}
			}
		};
	}
#endif
} // namespace graphics
