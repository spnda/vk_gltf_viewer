#pragma once

#include <span>
#include <filesystem>

#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/buffer.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include <mesh_common.h.glsl>

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

template <fastgltf::AnimationPath path>
struct AnimationSamplerTraits {};

template <> struct AnimationSamplerTraits<fastgltf::AnimationPath::Translation> { using OutputType = fastgltf::math::fvec3; };
template <> struct AnimationSamplerTraits<fastgltf::AnimationPath::Scale> { using OutputType = fastgltf::math::fvec3; };
template <> struct AnimationSamplerTraits<fastgltf::AnimationPath::Rotation> { using OutputType = fastgltf::math::fquat; };

struct AnimationSampler {
	std::vector<float> input;

	fastgltf::AnimationInterpolation interpolation;
	std::size_t componentCount;
	std::size_t outputCount;
	std::vector<float> values;

	explicit AnimationSampler(const fastgltf::Asset& asset, const fastgltf::AnimationSampler& sampler) : interpolation(sampler.interpolation) {
		ZoneScoped;
		auto& outputAccessor = asset.accessors[sampler.outputAccessor];
		outputCount = outputAccessor.count;
		componentCount = fastgltf::getNumComponents(outputAccessor.type);
	}

	template <fastgltf::AnimationPath path>
	auto sample(float time) {
		ZoneScoped;
		using T = typename AnimationSamplerTraits<path>::OutputType;

		time = std::fmod(time, input.back()); // Ugly hack to loop animations

		auto it = std::lower_bound(input.begin(), input.end(), time);
		if (it == input.cbegin()) {
			if (interpolation == fastgltf::AnimationInterpolation::CubicSpline)
				return T::fromPointer(&values[(1) * componentCount]);
			return T::fromPointer(&values[0]);
		}
		if (it == input.cend()) {
			if (interpolation == fastgltf::AnimationInterpolation::CubicSpline)
				return T::fromPointer(&values[(outputCount - 2) * componentCount]);
			return T::fromPointer(&values[(outputCount - 1) * componentCount]);
		}

		auto i = std::distance(input.begin(), it);
		auto t = (time - input[i - 1]) / (input[i] - input[i - 1]);

		switch (interpolation) {
			using enum fastgltf::AnimationInterpolation;
			case Step:
				return T::fromPointer(&values[(i - 1) * componentCount]);
			case Linear: {
				auto vk = T::fromPointer(&values[(i - 1) * componentCount]);
				auto vk1 = T::fromPointer(&values[i * componentCount]);

				if constexpr (path == fastgltf::AnimationPath::Rotation) {
					return fastgltf::math::slerp(vk, vk1, t);
				} else {
					return fastgltf::math::lerp(vk, vk1, t);
				}
			}
			case CubicSpline: {
				auto t2 = std::powf(t, 2);
				auto t3 = std::powf(t, 3);
				auto dt = input[i] - input[i - 1];

				std::array<T, 4> arr {{
					T::fromPointer(&values[(3 * (i - 1) + 1) * componentCount]),
					T::fromPointer(&values[(3 * (i - 1) + 2) * componentCount]),
					T::fromPointer(&values[(3 * (i + 0) + 1) * componentCount]),
					T::fromPointer(&values[(3 * (i + 0) + 0) * componentCount]),
				}};

				auto v = arr[0] * (2 * t3 - 3 * t2 + 1)
					   + arr[1] * (t3 - 2 * t2 + t) * dt
					   + arr[2] * (-2 * t3 + 3 * t2)
					   + arr[3] * (t3 - t2) * dt;

				if constexpr (path == fastgltf::AnimationPath::Rotation) {
					return normalize(v);
				} else {
					return v;
				}
			}
			default:
				std::unreachable();
		}
	}
};

struct Animation {
	std::vector<fastgltf::AnimationChannel> channels;
	std::vector<AnimationSampler> samplers;
};

struct DrawBuffers {
	bool isMeshletBufferBuilt = false;
	std::unique_ptr<ScopedBuffer> meshletDrawBuffer;
	std::unique_ptr<ScopedBuffer> transformBuffer;
};

class AssetLoadTask;

/**
 * A World represents the whole world we render at once. This holds everything from
 * the glTF assets, to the GPU buffers used for drawing.
 */
struct World {
	std::reference_wrapper<Device> device;

	std::vector<std::shared_ptr<fastgltf::Asset>> assets;

	std::vector<Mesh> meshes;
	std::vector<PrimitiveBuffers> primitiveBuffers;
	std::unique_ptr<ScopedBuffer> primitiveBuffer;

	std::vector<Animation> animations;
	float animationTime = 0.f;
	bool freezeAnimations = false;

	std::vector<fastgltf::Node> nodes;
	std::vector<fastgltf::Scene> scenes;

	std::vector<DrawBuffers> drawBuffers;

private:
	void rebuildDrawBuffer(std::size_t frameIndex);
	void updateTransformBuffer(std::size_t frameIndex);

public:
	explicit World(Device& device, std::size_t frameOverlap) noexcept;

	void addAsset(const std::shared_ptr<AssetLoadTask>& task);
	void iterateNode(std::size_t nodeIndex, fastgltf::math::fmat4x4 parent, std::function<void(fastgltf::Node&, const fastgltf::math::fmat4x4&)>& callback);
	void updateDrawBuffers(std::size_t frameIndex, float dt);
};

class AssetLoadTask : public ExceptionTaskSet {
	friend struct World;

	std::reference_wrapper<Device> device;
	std::filesystem::path assetPath;

	std::shared_ptr<fastgltf::Asset> asset;
	std::vector<Mesh> meshes;
	std::vector<std::pair<PrimitiveBuffers, glsl::Primitive>> primitives;
	std::vector<Animation> animations;

	std::shared_ptr<fastgltf::Asset> loadGltf();

public:
	explicit AssetLoadTask(Device& device, std::filesystem::path path);

	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};
