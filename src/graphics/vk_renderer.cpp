#include <tracy/Tracy.hpp>

#include <meshoptimizer.h>

#include <fastgltf/util.hpp>

#include <graphics/vk_renderer.hpp>

namespace gvk = graphics::vulkan;

void gvk::VkScene::rebuildDrawBuffer(std::size_t frameIndex) {
	ZoneScoped;
}

void gvk::VkScene::updateTransformBuffer(std::size_t frameIndex) {
	ZoneScoped;
}

void gvk::VkScene::addMesh(std::shared_ptr<Mesh> sharedMesh) {
	ZoneScoped;
	meshes.emplace_back(std::move(sharedMesh));

	// TODO: Can we invalidate less often somehow?
	for (auto& drawBuffer : drawBuffers)
		drawBuffer.isMeshletBufferBuilt = false;
}

void gvk::VkScene::updateDrawBuffers(std::size_t frameIndex, float dt) {
	ZoneScoped;

	rebuildDrawBuffer(frameIndex);
	updateTransformBuffer(frameIndex);
}

std::unique_ptr<graphics::Buffer> gvk::VkRenderer::createUniqueBuffer() {
	ZoneScoped;
}

std::shared_ptr<graphics::Buffer> gvk::VkRenderer::createSharedBuffer() {
	ZoneScoped;
}

std::shared_ptr<graphics::Mesh> gvk::VkRenderer::createSharedMesh(std::span<glsl::Vertex> vertexBuffer, std::span<index_t> indexBuffer) {
	ZoneScoped;

	// Generate the meshlets
	constexpr float coneWeight = 0.0f; // We leave this as 0 because we're not using cluster cone culling.
	constexpr auto maxPrimitives = fastgltf::alignDown(glsl::maxPrimitives, 4U); // meshopt requires the primitive count to be aligned to 4.
	std::size_t maxMeshlets = meshopt_buildMeshletsBound(indexBuffer.size(), glsl::maxVertices, maxPrimitives);
	std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
	std::vector<std::uint32_t> meshletVertices(maxMeshlets * glsl::maxVertices);
	std::vector<std::uint8_t> meshletTriangles(maxMeshlets * maxPrimitives * 3);


	// TODO: Take the mesh buffers and generate meshlets...
	return std::make_shared<gvk::VkMesh>();
}

void gvk::VkRenderer::updateResolution(glm::u32vec2 resolution) {
	ZoneScoped;
}

void gvk::VkRenderer::prepareFrame(std::size_t frameIndex) {
	ZoneScoped;
	auto& syncData = frameSyncData[frameIndex];

	// Wait for the last render for this frame index to finish, so that we can use
	// the associated resources again.
	vk::checkResult(syncData.presentFinished->wait(std::numeric_limits<std::uint64_t>::max()), "Failed to wait on the previous frame's fence");
	vk::checkResult(syncData.presentFinished->reset(), "Failed to reset the previous frame's fence");

	// Check if anything can be deleted this frame.
	device->timelineDeletionQueue->check();

	frameCommandPools[frameIndex].commandPool.reset_pool();
}

bool gvk::VkRenderer::draw(std::size_t frameIndex, graphics::Scene& gscene, float dt) {
	ZoneScoped;
	auto& scene = dynamic_cast<VkScene&>(gscene);

	scene.updateDrawBuffers(frameIndex, dt);

	auto& syncData = frameSyncData[frameIndex];
	auto& cmdPool = frameCommandPools[frameIndex];
	auto& cmd = cmdPool.commandBuffer;

	// Acquire the next swapchain image
	std::uint32_t swapchainImageIndex = 0;
	{
		auto result = vkAcquireNextImageKHR(*device, *swapchain, std::numeric_limits<std::uint64_t>::max(),
											syncData.imageAvailable->handle,
											VK_NULL_HANDLE, &swapchainImageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			swapchainNeedsRebuild = true;
			return false;
		}
		vk::checkResult(result, "Failed to acquire swapchain image");
	}

	// Begin the command buffer
	VkCommandBufferBeginInfo beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, // We're only using once, then resetting.
	};
	vkBeginCommandBuffer(cmd, &beginInfo);

	// Draw UI
	{
		TracyVkZone(device->tracyCtx, cmd, "ImGui rendering");
		const VkDebugUtilsLabelEXT label {
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
			.pLabelName = "ImGui rendering",
		};
		vkCmdBeginDebugUtilsLabelEXT(cmd, &label);

		// Insert a barrier to protect against any hazard reads from ImGui textures we might be using as render targets.
		const VkMemoryBarrier2 memoryBarrier {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
		};
		const VkDependencyInfo dependency {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &memoryBarrier,
		};
		vkCmdPipelineBarrier2(cmd, &dependency);

		auto extent = glm::u32vec2(renderResolution.x, renderResolution.y);
		imguiRenderer->draw(cmd, swapchain->imageViews[swapchainImageIndex], extent, frameIndex);

		vkCmdEndDebugUtilsLabelEXT(cmd);
	}

	// Always collect at the end of the main command buffer.
	TracyVkCollect(device->tracyCtx, cmd);

	vkEndCommandBuffer(cmd);

	// Submit
	{
		const VkSemaphoreSubmitInfo waitSemaphoreInfo {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = syncData.imageAvailable->handle,
			.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		};
		const VkCommandBufferSubmitInfo bufferSubmitInfo {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
			.commandBuffer = cmd,
		};
		std::array<VkSemaphoreSubmitInfo, 2> signalSemaphoreInfos = {{
			{
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
				.semaphore = syncData.renderingFinished->handle,
				.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			},
			{
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
				.semaphore = device->timelineDeletionQueue->getSemaphoreHandle(),
				.value = device->timelineDeletionQueue->nextValue(),
				.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			},
		}};
		const VkSubmitInfo2 submitInfo {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.waitSemaphoreInfoCount = 1,
			.pWaitSemaphoreInfos = &waitSemaphoreInfo,
			.commandBufferInfoCount = 1,
			.pCommandBufferInfos = &bufferSubmitInfo,
			.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signalSemaphoreInfos.size()),
			.pSignalSemaphoreInfos = signalSemaphoreInfos.data(),
		};
		vk::checkResult(device->graphicsQueue.submit(submitInfo, *syncData.presentFinished),
						"Failed to submit frame command buffer to queue");

		// Lastly, present the swapchain image as soon as rendering is done
		const VkPresentInfoKHR presentInfo {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &syncData.renderingFinished->handle,
			.swapchainCount = 1,
			.pSwapchains = &swapchain->swapchain.swapchain,
			.pImageIndices = &swapchainImageIndex,
		};
		auto presentResult = device->graphicsQueue.present(presentInfo);
		if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
			swapchainNeedsRebuild = true;
			return false;
		}
		vk::checkResult(presentResult, "Failed to present to queue");
	}

	return true;
}
