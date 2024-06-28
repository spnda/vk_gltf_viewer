#include <vk_gltf_viewer/assets.hpp>

namespace fg = fastgltf;

World::World(Device& _device, std::size_t frameOverlap) noexcept : device(_device) {
	drawBuffers.resize(frameOverlap);
}

/** This remaps the data from the loaded asset, which essentially fuses the assets together to one. */
void World::addAsset(const std::shared_ptr<AssetLoadTask>& task) {
	ZoneScoped;
	auto nodeOffset = nodes.size();
	auto meshOffset = meshes.size();
	auto primitiveOffset = primitiveBuffers.size();
	auto materialOffset = materials.size();

	auto& newAsset = assets.emplace_back(task->asset);

	// Append all the data to the back of our data, and adjust all indices correctly.
	meshes.reserve(meshes.size() + task->meshes.size());
	for (auto& mesh : task->meshes) {
		auto indices = mesh.primitiveIndices;
		for (auto& index : indices)
			index += primitiveOffset;

		meshes.emplace_back(Mesh {
			.primitiveIndices = std::move(indices),
		});
	}

	primitiveBuffers.reserve(primitiveBuffers.size() + task->primitives.size());
	for (auto& [buffers, primitive] : task->primitives)
		primitiveBuffers.emplace_back(std::move(buffers));

	materials.reserve(materials.size() + task->materials.size());
	for (auto& mat : task->materials)
		materials.emplace_back(mat);

	animations.reserve(animations.size() + task->animations.size());
	for (auto& animation : task->animations) {
		auto channels = animation.channels;
		for (auto& channel : channels)
			if (channel.nodeIndex.has_value())
				*channel.nodeIndex += nodeOffset;

		animations.emplace_back(Animation {
			.channels = std::move(channels),
			// The AnimationSampler object contains all data it needs, and therefore doesn't need remapping.
			.samplers = std::move(animation.samplers),
		});
	}

	materials.reserve(materials.size() + task->materials.size());
	materials.insert(materials.end(), task->materials.begin(), task->materials.end());

	nodes.reserve(nodes.size() + newAsset->nodes.size());
	for (auto& node : newAsset->nodes) {
		auto children = node.children;
		for (auto& child : children)
			child += nodeOffset;

		nodes.emplace_back(fastgltf::Node {
			.meshIndex = node.meshIndex.transform([&](std::size_t v) { return meshOffset + v; }),
			//.skinIndex = node.skinIndex.transform([&](std::size_t v) { return skinOffset + v; }),
			//.cameraIndex = node.cameraIndex.transform([&](std::size_t v) { return cameraOffset + v; }),
			//.lightIndex = node.lightIndex.transform([&](std::size_t v) { return lightOffset + v; }),
			.children = std::move(children),
			.transform = node.transform,
			.name = node.name,
		});
	}

	// TODO: Scene management?
	if (scenes.empty()) {
		scenes.emplace_back();
	}
	for (auto& node : task->asset->scenes.front().nodeIndices)
		scenes.front().nodeIndices.emplace_back(node + nodeOffset);

	// Upload the glsl::Primitive vector into the GPU buffer, and copy
	constexpr VmaAllocationCreateInfo allocationCreateInfo {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	std::array<std::uint32_t, 2> queueFamilies {{
		device.get().graphicsQueueFamily,
		device.get().transferQueueFamily,
	}};

	{
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = primitiveBuffers.size() * sizeof(glsl::Primitive),
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
					 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			// We need to use CONCURRENT since we're using the old buffer to copy while it still may be in use
			.sharingMode = VK_SHARING_MODE_CONCURRENT,
			.queueFamilyIndexCount = static_cast<std::uint32_t>(queueFamilies.size()),
			.pQueueFamilyIndices = queueFamilies.data(),
		};
		auto oldPrimitiveBuffer = std::move(primitiveBuffer);
		primitiveBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferCreateInfo, &allocationCreateInfo);
		vk::setDebugUtilsName(device.get(), primitiveBuffer->getHandle(), "Primitive buffer");

		auto primitiveStagingBuffer = device.get().createHostStagingBuffer(
			task->primitives.size() * sizeof(glsl::Primitive));
		{
			ScopedMap<glsl::Primitive> map(*primitiveStagingBuffer);
			for (std::size_t i = 0; auto& primitive: task->primitives) {
				auto& p = map.get()[i++] = primitive.second;
				p.materialIndex += materialOffset;
			}
		}

		device.get().immediateSubmit(device.get().getNextTransferQueueHandle(),
		                             device.get().uploadCommandPools[taskScheduler.GetThreadNum()],
		                             [&](auto cmd) {
			VkDeviceSize dstOffset = 0;
			if (oldPrimitiveBuffer) {
				const VkBufferCopy copyRegion {
					.size = oldPrimitiveBuffer->getBufferSize(),
				};
				dstOffset = copyRegion.size;
				vkCmdCopyBuffer(cmd, oldPrimitiveBuffer->getHandle(), primitiveBuffer->getHandle(), 1, &copyRegion);
			}

			const VkBufferCopy uploadRegion {
				.dstOffset = dstOffset,
				.size = primitiveStagingBuffer->getBufferSize(),
			};
			vkCmdCopyBuffer(cmd, primitiveStagingBuffer->getHandle(), primitiveBuffer->getHandle(), 1, &uploadRegion);
		});

		if (oldPrimitiveBuffer) {
			device.get().timelineDeletionQueue->push([buffer = std::move(oldPrimitiveBuffer)]() mutable {
				buffer.reset();
			});
		}
	}

	{
		const VkBufferCreateInfo bufferCreateInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = materials.size() * sizeof(glsl::Material),
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
					 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			// We need to use CONCURRENT since we're using the old buffer to copy while it still may be in use
			.sharingMode = VK_SHARING_MODE_CONCURRENT,
			.queueFamilyIndexCount = static_cast<std::uint32_t>(queueFamilies.size()),
			.pQueueFamilyIndices = queueFamilies.data(),
		};
		auto oldMaterialBuffer = std::move(materialBuffer);
		materialBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferCreateInfo, &allocationCreateInfo);
		vk::setDebugUtilsName(device.get(), materialBuffer->getHandle(), "Material buffer");

		auto materialStagingBuffer = device.get().createHostStagingBuffer(
			task->materials.size() * sizeof(glsl::Material));
		{
			ScopedMap<glsl::Material> map(*materialStagingBuffer);
			for (std::size_t i = 0; auto& material: task->materials) {
				map.get()[i++] = material;
			}
		}

		device.get().immediateSubmit(device.get().getNextTransferQueueHandle(),
		                             device.get().uploadCommandPools[taskScheduler.GetThreadNum()],
		                             [&](auto cmd) {
			VkDeviceSize dstOffset = 0;
			if (oldMaterialBuffer) {
				const VkBufferCopy copyRegion {
					.size = oldMaterialBuffer->getBufferSize(),
				};
				dstOffset = copyRegion.size;
				vkCmdCopyBuffer(cmd, oldMaterialBuffer->getHandle(), materialBuffer->getHandle(), 1, &copyRegion);
			}

			const VkBufferCopy uploadRegion {
				.dstOffset = dstOffset,
				.size = materialStagingBuffer->getBufferSize(),
			};
			vkCmdCopyBuffer(cmd, materialStagingBuffer->getHandle(), materialBuffer->getHandle(), 1, &uploadRegion);
		});

		if (oldMaterialBuffer) {
			device.get().timelineDeletionQueue->push([buffer = std::move(oldMaterialBuffer)]() mutable {
				buffer.reset();
			});
		}
	}

	// Invalidate old draw buffers
	for (auto& drawBuffer : drawBuffers)
		drawBuffer.isMeshletBufferBuilt = false;

	fmt::print(stderr, "Finished loading asset\n");
}

void World::iterateNode(std::size_t nodeIndex, fg::math::fmat4x4 parent,
						std::function<void(fg::Node&, const fg::math::fmat4x4&)>& callback) {
	ZoneScoped;
	auto& node = nodes[nodeIndex];

	// Compute the animations
	fg::TRS trs = std::get<fg::TRS>(node.transform); // We specify DecomposeNodeMatrices, so it should always be this.
	for (auto& animation : animations) {
		for (auto& channel : animation.channels) {
			if (!channel.nodeIndex.has_value() || *channel.nodeIndex != nodeIndex)
				continue;

			auto& sampler = animation.samplers[channel.samplerIndex];
			switch (channel.path) {
				using enum fg::AnimationPath;
				case Translation: {
					trs.translation = sampler.sample<fg::AnimationPath::Translation>(animationTime);
					break;
				}
				case Scale: {
					trs.scale = sampler.sample<fg::AnimationPath::Scale>(animationTime);
					break;
				}
				case Rotation: {
					trs.rotation = sampler.sample<fg::AnimationPath::Rotation>(animationTime);
					break;
				}
				case Weights:
					break;
			}
		}
	}

	// Compute the matrix with the animated values
	auto matrix = scale(rotate(translate(parent, trs.translation), trs.rotation), trs.scale);

	callback(node, matrix);

	for (auto& child : node.children) {
		iterateNode(child, matrix, callback);
	}
}

void World::rebuildDrawBuffer(std::size_t frameIndex) {
	ZoneScoped;
	auto& drawBuffer = drawBuffers[frameIndex];
	if (scenes.empty() || drawBuffer.isMeshletBufferBuilt)
		return;

	VkDeviceSize currentDrawBufferSize = drawBuffer.meshletDrawBuffer ? drawBuffer.meshletDrawBuffer->getBufferSize() : 0;

	std::vector<glsl::MeshletDraw> draws;
	draws.reserve(currentDrawBufferSize / sizeof(glsl::MeshletDraw));

	auto& scene = scenes[0];
	std::uint32_t transformCount = 0;
	for (auto& node : scene.nodeIndices) {
		// This assumes that the glTF node hierarchy *never* changes and we can just reuse the indices
		// into the transform buffer
		std::function<void(fg::Node&, const fg::math::fmat4x4&)> callback = [&](fg::Node& node, const fg::math::fmat4x4& matrix) {
			ZoneScoped;
			if (!node.meshIndex.has_value()) {
				return;
			}

			auto transformIndex = transformCount++;
			for (auto& primitive : meshes[*node.meshIndex].primitiveIndices) {
				auto& buffers = primitiveBuffers[primitive];
				for (std::uint32_t i = 0; i < buffers.meshletCount; ++i) {
					draws.emplace_back(glsl::MeshletDraw {
						.primitiveIndex = static_cast<std::uint32_t>(primitive),
						.meshletIndex = i,
						.transformIndex = transformIndex,
					});
				}
			}
		};
		iterateNode(node, fg::math::fmat4x4(1.f), callback);
	}

	constexpr auto bufferUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	constexpr VmaAllocationCreateInfo allocationCreateInfo = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	const auto requiredDrawBufferSize = draws.size() * sizeof(glsl::MeshletDraw);
	if (currentDrawBufferSize < requiredDrawBufferSize) {
		const VkBufferCreateInfo bufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = requiredDrawBufferSize,
			.usage = bufferUsage,
		};
		drawBuffer.meshletDrawBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferCreateInfo, &allocationCreateInfo);
		vk::setDebugUtilsName(device.get(), drawBuffer.meshletDrawBuffer->getHandle(), fmt::format("Meshlet draw buffer {}", frameIndex));
	}

	// This copy takes quite a while, often multiple milliseconds, which is why
	// this should be updated as rarely as possible, and especially not every frame.
	if (requiredDrawBufferSize > 0) {
		ZoneScopedN("Draw buffer copy");
		ScopedMap drawMap(*drawBuffer.meshletDrawBuffer);
		std::memcpy(drawMap.get(), draws.data(), requiredDrawBufferSize);
	}

	drawBuffer.isMeshletBufferBuilt = true;
}

void World::updateTransformBuffer(std::size_t frameIndex) {
	ZoneScoped;
	if (scenes.empty())
		return;

	auto& drawBuffer = drawBuffers[frameIndex];
	VkDeviceSize currentTransformBufferSize = drawBuffer.transformBuffer ? drawBuffer.transformBuffer->getBufferSize() : 0;

	std::vector<fastgltf::math::fmat4x4> transforms;
	transforms.reserve(currentTransformBufferSize / sizeof(fastgltf::math::fmat4x4));

	auto& scene = scenes[0];
	for (auto& node : scene.nodeIndices) {
		// This assumes that the glTF node hierarchy *never* changes and we can just reuse the indices
		// into the transform buffer
		std::function<void(fg::Node&, const fg::math::fmat4x4&)> callback = [&](fg::Node& node, const fg::math::fmat4x4& matrix) {
			ZoneScoped;
			if (!node.meshIndex.has_value()) {
				return;
			}

			transforms.emplace_back(matrix);
		};
		iterateNode(node, fg::math::fmat4x4(1.f), callback);
	}

	constexpr auto bufferUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	constexpr VmaAllocationCreateInfo allocationCreateInfo = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	const auto requiredTransformBufferSize = transforms.size() * sizeof(glm::mat4);
	if (currentTransformBufferSize < requiredTransformBufferSize) {
		const VkBufferCreateInfo bufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = requiredTransformBufferSize,
			.usage = bufferUsage,
		};
		drawBuffer.transformBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferCreateInfo, &allocationCreateInfo);
		vk::setDebugUtilsName(device.get(), drawBuffer.transformBuffer->getHandle(), fmt::format("Transform buffer {}", frameIndex));
	}

	if (requiredTransformBufferSize > 0) {
		ZoneScopedN("Transform buffer copy");
		ScopedMap<fastgltf::math::fmat4x4> transformMap(*drawBuffer.transformBuffer);
		// memcpy would technically also work, but fmat4x4 is technically not TriviallyCopyable,
		// fingers crossed that the compiler will optimise this properly.
		std::copy(transforms.begin(), transforms.end(), transformMap.get());
	}
}

void World::updateDrawBuffers(std::size_t frameIndex, float dt) {
	ZoneScoped;
	if (!freezeAnimations) {
		animationTime += dt;
	}

	rebuildDrawBuffer(frameIndex);
	updateTransformBuffer(frameIndex);
}
