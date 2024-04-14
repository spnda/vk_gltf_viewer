#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <utility>

#include <TaskScheduler.h>
#include <fmt/format.h>
#include <imgui_impl_glfw.h>
#include <tracy/Tracy.hpp>

#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/util.hpp>
#include <vk_gltf_viewer/viewer.hpp>
#include <vk_gltf_viewer/imgui_renderer.hpp>

#include <vulkan/vk.hpp>
#include <vulkan/debug_utils.hpp>
#include <vulkan/fmt.hpp>
#include <vulkan/pipeline_builder.hpp>
#include <vulkan/cache.hpp>

namespace fs = std::filesystem;

namespace imgui {
	const auto pipelineCacheFile = std::filesystem::current_path() / "cache/imgui.cache";

	class ShaderLoadTask : public enki::ITaskSet {
		Renderer* renderer;

	public:
		explicit ShaderLoadTask(Renderer* renderer) : renderer(renderer) {}

		void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) override {
			ZoneScoped;
			vk::loadShaderModule("ui.frag.glsl.spv", renderer->viewer->device, &renderer->fragmentShader);
			vk::loadShaderModule("ui.vert.glsl.spv", renderer->viewer->device, &renderer->vertexShader);
		}
	};
} // namespace imgui

void imgui::Renderer::createFontAtlas() {
	ZoneScoped;
	auto* fonts = ImGui::GetIO().Fonts;
	fonts->Build();

	if (fontAtlas != VK_NULL_HANDLE) {
		// destroy should only destroy the texture data.
		vmaDestroyImage(viewer->allocator, fontAtlas, fontAtlasAllocation);
		fontAtlas = VK_NULL_HANDLE;
		fontAtlasAllocation = VK_NULL_HANDLE;
	}

	auto& io = ImGui::GetIO();
	unsigned char* pixels = nullptr;
	int width = 0;
	int height = 0;
	io.Fonts->GetTexDataAsAlpha8(&pixels, &width, &height);
	fontAtlasExtent = { static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height) };

	const VkImageCreateInfo imageCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R8_UNORM,
		.extent = {
			.width = fontAtlasExtent.x,
			.height = fontAtlasExtent.y,
			.depth = 1,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo allocationCreateInfo = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	auto result = vmaCreateImage(viewer->allocator, &imageCreateInfo, &allocationCreateInfo,
				   &fontAtlas, &fontAtlasAllocation, VK_NULL_HANDLE);
	vk::checkResult(result, "Failed to create ImGui font atlas: {}");
	vk::setDebugUtilsName(viewer->device, fontAtlas, "ImGui font atlas");
	vk::setAllocationName(viewer->allocator, fontAtlasAllocation, "ImGui font atlas allocation");

	const VkImageViewCreateInfo imageViewCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = fontAtlas,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = VK_FORMAT_R8_UNORM,
		.components = {
			.r = VK_COMPONENT_SWIZZLE_ONE,
			.g = VK_COMPONENT_SWIZZLE_ONE,
			.b = VK_COMPONENT_SWIZZLE_ONE,
			.a = VK_COMPONENT_SWIZZLE_R,
		},
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		}
	};
	vkCreateImageView(viewer->device, &imageViewCreateInfo, VK_NULL_HANDLE, &fontAtlasView);
	vk::setDebugUtilsName(viewer->device, fontAtlasView, "ImGui font atlas view");
	io.Fonts->SetTexID(static_cast<ImTextureID>(fontAtlasView));

	auto data = std::span<const std::byte> { reinterpret_cast<std::byte*>(pixels), width * height * sizeof(std::byte) };
	VkBuffer stagingBuffer;
	VmaAllocation stagingAllocation;
	result = viewer->createHostStagingBuffer(data.size_bytes(), &stagingBuffer, &stagingAllocation);
	vk::checkResult(result, "Failed to create host staging buffer: {}");

	{
		vk::ScopedMap map(viewer->allocator, stagingAllocation);
		std::memcpy(map.get(), data.data(), data.size_bytes());
	}

	viewer->immediateSubmit(viewer->getNextTransferQueueHandle(), viewer->uploadCommandPools[taskScheduler.GetThreadNum()], [&](VkCommandBuffer cmd) {
		{
			vk::ScopedMap map(viewer->allocator, stagingAllocation);
			std::memcpy(map.get(), data.data(), data.size_bytes());
		}

		// Transition the image to TRANSFER_DST_OPTIMAL
		VkImageMemoryBarrier2 imageBarrier {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
			.srcAccessMask = VK_ACCESS_2_NONE,
			.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = fontAtlas,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.layerCount = 1,
			},
		};
		const VkDependencyInfo dependencyInfo {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &imageBarrier,
		};
		vkCmdPipelineBarrier2(cmd, &dependencyInfo);

		// Copy the image
		const VkBufferImageCopy copy {
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.layerCount = 1,
			},
			.imageOffset = {
				.x = 0,
				.y = 0,
				.z = 0,
			},
			.imageExtent = imageCreateInfo.extent,
		};
		vkCmdCopyBufferToImage(cmd, stagingBuffer, fontAtlas,
							   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

		// Transition the image into the destinationLayout
		imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
		imageBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
		imageBarrier.dstAccessMask = VK_ACCESS_2_NONE;
		imageBarrier.oldLayout = imageBarrier.newLayout;
		imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		vkCmdPipelineBarrier2(cmd, &dependencyInfo);
	}, [allocator = viewer->allocator, buffer = stagingBuffer, allocation = stagingAllocation]() {
		vmaDestroyBuffer(allocator, buffer, allocation);
	});
}

void imgui::Renderer::destroy() {
	ZoneScoped;
	vk::PipelineCacheSaveTask cacheSaveTask(viewer->device, &pipelineCache, pipelineCacheFile);

	if (volkGetLoadedDevice() != nullptr) {
		taskScheduler.AddTaskSetToPipe(&cacheSaveTask);

		for (auto& buf : buffers) {
			vmaDestroyBuffer(viewer->allocator, buf.vertexBuffer, buf.vertexAllocation);
			vmaDestroyBuffer(viewer->allocator, buf.indexBuffer, buf.indexAllocation);
		}

		vmaDestroyBuffer(viewer->allocator, fontAtlasStagingBuffer, fontAtlasStagingAllocation);
		vkDestroySampler(viewer->device, fontAtlasSampler, nullptr);
		vkDestroyImageView(viewer->device, fontAtlasView, nullptr);
		vmaDestroyImage(viewer->allocator, fontAtlas, fontAtlasAllocation);

		vkResetDescriptorPool(viewer->device, descriptorPool, 0);
		vkDestroyDescriptorPool(viewer->device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(viewer->device, descriptorLayout, nullptr);

		vkDestroyPipeline(viewer->device, pipeline, nullptr);
		vkDestroyPipelineLayout(viewer->device, pipelineLayout, nullptr);

		vkDestroyShaderModule(viewer->device, fragmentShader, nullptr);
		vkDestroyShaderModule(viewer->device, vertexShader, nullptr);
	}

	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	if (volkGetLoadedDevice() != nullptr) {
		taskScheduler.WaitforTask(&cacheSaveTask);
		vkDestroyPipelineCache(viewer->device, pipelineCache, nullptr);
	}
}

VkResult imgui::Renderer::createGeometryBuffers(std::size_t index, VkDeviceSize vertexSize, VkDeviceSize indexSize) {
	ZoneScoped;
	// We will allocate at least space for 10.000 vertices, which is already more than most UIs will use.
	constexpr VkDeviceSize minimumVertexCount = 10'000;
	constexpr VkDeviceSize increaseFactor = 10;
	auto& current = buffers[index];

	constexpr auto bufferUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	const VmaAllocationCreateInfo allocationCreateInfo = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	VkResult result = VK_SUCCESS;
	if (current.vertexBufferSize < vertexSize) {
		vmaDestroyBuffer(viewer->allocator, current.vertexBuffer, current.vertexAllocation);

		current.vertexBufferSize = util::max(sizeof(ImDrawVert) * minimumVertexCount, vertexSize * increaseFactor);
		const VkBufferCreateInfo vertexBufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = current.vertexBufferSize,
			.usage = bufferUsage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		};
		result = vmaCreateBuffer(viewer->allocator, &vertexBufferCreateInfo, &allocationCreateInfo, &current.vertexBuffer,
								 &current.vertexAllocation, VK_NULL_HANDLE);
		if (result != VK_SUCCESS) {
			return result;
		}
		vk::setDebugUtilsName(viewer->device, current.vertexBuffer, fmt::format("ImGui Vertex Buffer {}", index));

		// TODO: Check for BDA availability
		const VkBufferDeviceAddressInfo bdaInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = current.vertexBuffer,
		};
		current.vertexBufferAddress = vkGetBufferDeviceAddress(viewer->device, &bdaInfo);
	}

	if (current.indexBufferSize < indexSize) {
		vmaDestroyBuffer(viewer->allocator, current.indexBuffer, current.indexAllocation);

		current.indexBufferSize = util::max(sizeof(ImDrawIdx) * minimumVertexCount, indexSize * increaseFactor);

		const VkBufferCreateInfo indexBufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = current.indexBufferSize,
			.usage = bufferUsage | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		};
		result = vmaCreateBuffer(viewer->allocator, &indexBufferCreateInfo, &allocationCreateInfo, &current.indexBuffer, &current.indexAllocation,
								 VK_NULL_HANDLE);
		if (result != VK_SUCCESS) {
			return result;
		}
		vk::setDebugUtilsName(viewer->device, current.indexBuffer, fmt::format("ImGui Index Buffer {}", index));
	}

	return result;
}

void imgui::Renderer::addTextureToDescriptorSet(ImTextureID textureId) {
	ZoneScoped;
	if (imageDescriptorIndices.contains(textureId)) {
		return;
	}

	// Set a descriptor index for the given texture
	auto idx = imageDescriptorIndices[textureId] = imageDescriptorIndices.size();

	// Update the descriptor set
	const VkDescriptorImageInfo textureInfo {
		.imageView = static_cast<VkImageView>(textureId),
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	const VkWriteDescriptorSet write {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 0,
		.dstArrayElement = idx,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &textureInfo,
	};
	vkUpdateDescriptorSets(viewer->device, 1, &write, 0, nullptr);
}

void imgui::Renderer::draw(VkCommandBuffer commandBuffer, VkImageView swapchainImageView, glm::u32vec2 framebufferSize, std::size_t currentFrame) {
	ZoneScoped;
	// Reset the indices. This effectively clears the descriptor set, and we overwrite it later throughout the draw process.
	// TODO: Is resetting every frame the most efficient way? This implies a full update of the descriptor set.
	imageDescriptorIndices.clear();

	auto* drawData = ImGui::GetDrawData();

	// Copy all vertex and index buffers into the proper buffers. Because of Vulkan, we cannot copy
	// buffers while within a render pass.
	if (drawData->TotalVtxCount <= 0) {
		return;
	}

	auto& frameBuffers = buffers[currentFrame];
	auto commandLists = std::span(drawData->CmdLists.Data, drawData->CmdLists.Size);

	const std::size_t vertexBufferSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
	const std::size_t indexBufferSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

	// We will have to resize the buffers if they're not large enough for all the data.
	if (vertexBufferSize > frameBuffers.vertexBufferSize || indexBufferSize > frameBuffers.indexBufferSize) {
		auto result = createGeometryBuffers(currentFrame, vertexBufferSize, indexBufferSize);
		vk::checkResult(result, "Failed to create ImGui geometry buffers: {}");
	}

	// Copy the vertex and index buffers
	{
		vk::ScopedMap<ImDrawVert> vtxData(viewer->allocator, frameBuffers.vertexAllocation);
		vk::ScopedMap<ImDrawIdx> idxData(viewer->allocator, frameBuffers.indexAllocation);

		auto* vertexDestination = vtxData.get();
		auto* indexDestination = idxData.get();
		for (const auto& list : commandLists) {
			std::memcpy(vertexDestination, list->VtxBuffer.Data, list->VtxBuffer.Size * sizeof(ImDrawVert));
			std::memcpy(indexDestination, list->IdxBuffer.Data, list->IdxBuffer.Size * sizeof(ImDrawIdx));

			// Because the destination pointers have a type of ImDrawXYZ*, it already
			// properly takes the byte size into account.
			vertexDestination += list->VtxBuffer.Size;
			indexDestination += list->IdxBuffer.Size;
		}
	}

	{
		std::array<VkBufferMemoryBarrier2, 2> memoryBarriers = {{
			{
				.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
				.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
				.buffer = frameBuffers.vertexBuffer,
				.offset = 0,
				.size = frameBuffers.vertexBufferSize,
			},
			{
				.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
				.dstAccessMask = VK_ACCESS_2_INDEX_READ_BIT,
				.buffer = frameBuffers.indexBuffer,
				.offset = 0,
				.size = frameBuffers.indexBufferSize,
			}
		}};
		const VkDependencyInfo geometryBufferDependency = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.bufferMemoryBarrierCount = 2,
			.pBufferMemoryBarriers = memoryBarriers.data(),
		};
		vkCmdPipelineBarrier2(commandBuffer, &geometryBufferDependency);
	}

	const VkRenderingAttachmentInfo colorAttachmentInfo = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = swapchainImageView,
		.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.resolveMode = VK_RESOLVE_MODE_NONE,
		.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = { 0.0F, 0.0F, 0.0F, 0.0F },
	};
	const VkRenderingInfo renderingInfo = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { .offset = {}, .extent = { .width = framebufferSize.x, .height = framebufferSize.y } },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachmentInfo,
	};
	vkCmdBeginRendering(commandBuffer, &renderingInfo);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

	{
		const VkViewport viewport = {
			.x = 0.0F,
			.y = 0.0F,
			.width = drawData->DisplaySize.x * drawData->FramebufferScale.x,
			.height = drawData->DisplaySize.y * drawData->FramebufferScale.y,
			.minDepth = 0.0F,
			.maxDepth = 1.0F,
		};
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
	}

	// TODO: Bind descriptor set

	const ImVec2& clipOffset = drawData->DisplayPos;      // (0,0) unless using multi-viewports
	const ImVec2& clipScale = drawData->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

	auto framebufferWidth = static_cast<std::uint32_t>(drawData->DisplaySize.x * drawData->FramebufferScale.x);
	auto framebufferHeight = static_cast<std::uint32_t>(drawData->DisplaySize.y * drawData->FramebufferScale.y);

	// Update the scale and translate floats for the vertex shader.
	pushConstants.scale.x = 2.0F / drawData->DisplaySize.x;
	pushConstants.scale.y = 2.0F / drawData->DisplaySize.y;
	pushConstants.translate.x = -1.0F - drawData->DisplayPos.x * pushConstants.scale.x;
	pushConstants.translate.y = -1.0F - drawData->DisplayPos.y * pushConstants.scale.y;

	std::size_t vertexOffset = 0;
	std::size_t indexOffset = 0;
	for (auto& list : commandLists) {
		auto cmdBuffer = std::span(list->CmdBuffer.Data, list->CmdBuffer.Size);
		for (const auto& cmd : cmdBuffer) {
			if (cmd.ElemCount == 0) { // drawIndexed doesn't accept this
				continue;
			}

			const glm::u32vec2 clipMin = {
				util::max(0U, static_cast<std::uint32_t>((cmd.ClipRect.x - clipOffset.x) * clipScale.x)),
				util::max(0U, static_cast<std::uint32_t>((cmd.ClipRect.y - clipOffset.y) * clipScale.y))
			};
			const glm::u32vec2 clipMax = {
				util::min(framebufferWidth, static_cast<std::uint32_t>((cmd.ClipRect.z - clipOffset.x) * clipScale.x)),
				util::min(framebufferHeight, static_cast<std::uint32_t>((cmd.ClipRect.w - clipOffset.y) * clipScale.y))
			};

			if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y) {
				continue;
			}

			const VkRect2D rect = {
				.offset = {
					.x = static_cast<std::int32_t>(clipMin.x),
					.y = static_cast<std::int32_t>(clipMin.y),
				},
				.extent = {
					.width = clipMax.x - clipMin.x,
					.height = clipMax.y - clipMin.y,
				}
			};
			vkCmdSetScissor(commandBuffer, 0, 1, &rect);

			if (cmd.GetTexID() == nullptr) {
				// If no texture ID was specified, we default to the font atlas.
				pushConstants.imageIndex = imageDescriptorIndices[fontAtlasView];
			} else {
				// Write the texture used for this draw command into the descriptor set.
				addTextureToDescriptorSet(cmd.GetTexID());

				pushConstants.imageIndex = imageDescriptorIndices[cmd.GetTexID()];
			}
			pushConstants.vertexBufferAddress = frameBuffers.vertexBufferAddress + (vertexOffset + cmd.VtxOffset) * sizeof(ImDrawVert);
			vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
							   0, sizeof(glsl::PushConstants), &pushConstants);

			vkCmdBindIndexBuffer(commandBuffer, frameBuffers.indexBuffer,
								 (cmd.IdxOffset + static_cast<std::uint32_t>(indexOffset)) * sizeof(ImDrawIdx),
								 sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);

			vkCmdDrawIndexed(commandBuffer, cmd.ElemCount, 1, 0, 0, 0);
		}

		indexOffset += list->IdxBuffer.Size;
		vertexOffset += list->VtxBuffer.Size;
	}

	vkCmdEndRendering(commandBuffer);
}

VkResult imgui::Renderer::init(Viewer* pViewer) {
	ZoneScoped;
	viewer = pViewer;

	vk::PipelineCacheLoadTask cacheLoadTask(viewer->device, &pipelineCache, pipelineCacheFile);
	taskScheduler.AddTaskSetToPipe(&cacheLoadTask);

	ShaderLoadTask shaderLoadTask(this);
	taskScheduler.AddTaskSetToPipe(&shaderLoadTask);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForVulkan(viewer->window, true);

	auto& io = ImGui::GetIO();
	io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
	io.BackendRendererName = "imgui::ImGuiRenderer";
	io.BackendPlatformName = "Vulkan";

	// Create the sampler. It is static, therefore we will pass it as an immutable sampler to the
	// descriptor layout.
	const VkSamplerCreateInfo samplerInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
	};
	auto samplerResult = vkCreateSampler(viewer->device, &samplerInfo, VK_NULL_HANDLE, &fontAtlasSampler);
	vk::checkResult(samplerResult, "Failed to create ImGui font-atlas sampler");
	vk::setDebugUtilsName(viewer->device, fontAtlasSampler, "ImGui font-atlas sampler");

	// Create the descriptor layout
	std::vector<VkSampler> immutableSamplers(maxBindlessImages, fontAtlasSampler); // TODO: Perhaps we don't need immutable samplers?
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = maxBindlessImages, // Is this a sane max value for displayed images?
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = immutableSamplers.data(),
	};
	const VkDescriptorBindingFlags bindingFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
	const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		.bindingCount = 1,
		.pBindingFlags = &bindingFlags,
	};
	const VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = &bindingFlagsInfo,
		.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		.bindingCount = 1,
		.pBindings = &binding,
	};
	auto descriptorLayoutResult = vkCreateDescriptorSetLayout(viewer->device, &descriptorLayoutInfo, nullptr, &descriptorLayout);
	vk::checkResult(descriptorLayoutResult, "Failed to create ImGui font atlas descriptor set layout: {}");

	// Create the descriptor pool to hold exactly the amount of descriptors the backend supports.
	std::array<VkDescriptorPoolSize, 1> descriptorPoolSizes = {{
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = maxBindlessImages,
		}
	}};
	const VkDescriptorPoolCreateInfo descriptorPoolInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
		.maxSets = 1,
		.poolSizeCount = static_cast<std::uint32_t>(descriptorPoolSizes.size()),
		.pPoolSizes = descriptorPoolSizes.data(),
	};
	auto descriptorPoolResult = vkCreateDescriptorPool(viewer->device, &descriptorPoolInfo, nullptr, &descriptorPool);
	vk::checkResult(descriptorPoolResult, "Failed to create ImGui descriptor pool: {}");

	// Create the single descriptor set
	const VkDescriptorSetAllocateInfo descriptorSetInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptorLayout,
	};
	auto allocateResult = vkAllocateDescriptorSets(viewer->device, &descriptorSetInfo, &descriptorSet);
	vk::checkResult(allocateResult, "Failed to create ImGui frame descriptor sets: {}");

	// Create the pipeline layout
	const VkPushConstantRange pushConstantRange = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(glsl::PushConstants),
	};
	const VkPipelineLayoutCreateInfo layoutCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptorLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	auto result = vkCreatePipelineLayout(viewer->device, &layoutCreateInfo, VK_NULL_HANDLE, &pipelineLayout);
	if (result != VK_SUCCESS) {
		return result;
	}

	// Create the pipeline
	const VkFormat colorAttachmentFormat = viewer->swapchain.image_format;
	const VkPipelineRenderingCreateInfo renderingCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &colorAttachmentFormat,
	};

	const VkPipelineColorBlendAttachmentState blendAttachment = {
		.blendEnable = VK_TRUE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	auto builder = vk::GraphicsPipelineBuilder(viewer->device, 1)
		.setPipelineLayout(0, pipelineLayout)
		.addDynamicState(0, VK_DYNAMIC_STATE_SCISSOR)
		.addDynamicState(0, VK_DYNAMIC_STATE_VIEWPORT)
		.pushPNext(0, &renderingCreateInfo)
		.setBlendAttachment(0, &blendAttachment)
		.setTopology(0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		.setDepthState(0, VK_FALSE, VK_FALSE, VK_COMPARE_OP_GREATER_OR_EQUAL)
		.setRasterState(0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE)
		.setMultisampleCount(0, VK_SAMPLE_COUNT_1_BIT)
		.setScissorCount(0, 1U)
		.setViewportCount(0, 1U);

	taskScheduler.WaitforTask(&shaderLoadTask);
	builder.addShaderStage(0, VK_SHADER_STAGE_VERTEX_BIT, vertexShader, "main")
		.addShaderStage(0, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader, "main");

	taskScheduler.WaitforTask(&cacheLoadTask);
	if (cacheLoadTask.getResult() == VK_SUCCESS) {
		fmt::print("Successfully created ImGui pipeline cache\n");
		builder.setPipelineCache(pipelineCache);
	} else {
		fmt::print(stderr, "Failed to load ImGui pipeline cache: {}\n", cacheLoadTask.getResult());
	}

	return builder.build(&pipeline);
}

VkResult imgui::Renderer::initFrameData(std::uint32_t frameCount) {
	ZoneScoped;
	// Create the index/vertex buffers. As the swapchain implementation might have multiple
	// swapchain images, meaning we have multiple frames in flight, we'll need unique buffers
	// for each frame in flight to avoid any race conditions.
	buffers.resize(frameCount);
	for (auto i = 0U; i < frameCount; ++i) {
		createGeometryBuffers(i, 0, 0);
	}
	return VK_SUCCESS;
}

void imgui::Renderer::newFrame() {
	ZoneScoped;
	ImGui_ImplGlfw_NewFrame();
}
