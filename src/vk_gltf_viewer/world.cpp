#include <vk_gltf_viewer/assets.hpp>

namespace fg = fastgltf;

World::World(Device& _device, std::size_t frameOverlap) noexcept : device(_device) {
	drawBuffers.resize(frameOverlap);
}

void World::addAsset(const std::shared_ptr<AssetLoadTask>& task) {
	ZoneScoped;
	assert(assets.empty()); // Remove when we support more than one asset.

	// TODO: Add support for rendering multiple assets
	//       This will require a merged primitive buffer, or some adjustment in the MeshletDraw
	//       structure, to identify to which asset a primitive belongs.
	assets.emplace_back(task->asset);

	// Append all the data to the back of our data, and adjust all indices correctly.
	meshes = std::move(task->meshes);
	primitiveBuffers.reserve(task->primitives.size());
	for (auto& [buffers, primitive] : task->primitives)
		primitiveBuffers.emplace_back(std::move(buffers));

	animations = std::move(task->animations);

	// TODO: This needs updating, too
	nodes = assets.back()->nodes;
	scenes = assets.back()->scenes;

	// TODO: We need to append to a new, bigger buffer, and then delete the old one.
	device.get().timelineDeletionQueue->push([buffer = std::move(primitiveBuffer)]() mutable {
		buffer.reset();
	});
	primitiveBuffer = std::move(task->primitiveBuffer);
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
				case fg::AnimationPath::Translation: {
					trs.translation = sampler.sample<fg::AnimationPath::Translation>(animationTime);
					break;
				}
				case fg::AnimationPath::Scale: {
					trs.scale = sampler.sample<fg::AnimationPath::Scale>(animationTime);
					break;
				}
				case fastgltf::AnimationPath::Rotation: {
					trs.rotation = sampler.sample<fg::AnimationPath::Rotation>(animationTime);
					break;
				}
				case fastgltf::AnimationPath::Weights:
					break;
			}
		}
	}

	// Compute the matrix with the animated values
	auto matrix = parent
		* translate(fg::math::fmat4x4(), trs.translation)
		* fg::math::fmat4x4(asMatrix(trs.rotation))
		* scale(fg::math::fmat4x4(), trs.scale);

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
	{
		ZoneScopedN("Draw buffer copy");
		ScopedMap drawMap(*drawBuffer.meshletDrawBuffer);
		std::memcpy(drawMap.get(), draws.data(), drawBuffer.meshletDrawBuffer->getBufferSize());
	}

	drawBuffer.isMeshletBufferBuilt = true;
}

void World::updateTransformBuffer(std::size_t frameIndex) {
	ZoneScoped;
	if (scenes.empty())
		return;

	auto& drawBuffer = drawBuffers[frameIndex];
	VkDeviceSize currentTransformBufferSize = drawBuffer.transformBuffer ? drawBuffer.meshletDrawBuffer->getBufferSize() : 0;

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

	{
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
