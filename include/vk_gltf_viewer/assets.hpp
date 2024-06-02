#pragma once

#include <span>
#include <filesystem>

#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/buffer.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

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
	fastgltf::AnimationSampler sampler;

	std::vector<float> input;

	explicit AnimationSampler(const fastgltf::Asset& asset, const fastgltf::AnimationSampler& _sampler) : sampler(_sampler) {
		ZoneScoped;
		auto& inputAccessor = asset.accessors[sampler.inputAccessor];
		input.resize(inputAccessor.count);
		fastgltf::copyFromAccessor<float>(asset, inputAccessor, input.data());
	}

	template <fastgltf::AnimationPath path>
	auto sample(const fastgltf::Asset& asset, float time) {
		ZoneScoped;
		using T = typename AnimationSamplerTraits<path>::OutputType;

		time = std::fmod(time, input.back()); // Ugly hack to loop animations

		auto& outputAccessor = asset.accessors[sampler.outputAccessor];

		auto it = std::lower_bound(input.begin(), input.end(), time);
		if (it == input.cbegin()) {
			if (sampler.interpolation == fastgltf::AnimationInterpolation::CubicSpline)
				return fastgltf::getAccessorElement<T>(asset, outputAccessor, 1);
			return fastgltf::getAccessorElement<T>(asset, outputAccessor, 0);
		}
		if (it == input.cend()) {
			if (sampler.interpolation == fastgltf::AnimationInterpolation::CubicSpline)
				return fastgltf::getAccessorElement<T>(asset, outputAccessor, outputAccessor.count - 2);
			return fastgltf::getAccessorElement<T>(asset, outputAccessor, outputAccessor.count - 1);
		}

		auto i = std::distance(input.begin(), it);
		auto t = (time - input[i - 1]) / (input[i] - input[i - 1]);

		switch (sampler.interpolation) {
			case fastgltf::AnimationInterpolation::Step:
				return fastgltf::getAccessorElement<T>(asset, outputAccessor, i - 1);
			case fastgltf::AnimationInterpolation::Linear: {
				auto vk = fastgltf::getAccessorElement<T>(asset, outputAccessor, i - 1);
				auto vk1 = fastgltf::getAccessorElement<T>(asset, outputAccessor, i);

				if constexpr (path == fastgltf::AnimationPath::Rotation) {
					return fastgltf::math::slerp(vk, vk1, t);
				} else {
					return fastgltf::math::lerp(vk, vk1, t);
				}
			}
			case fastgltf::AnimationInterpolation::CubicSpline: {
				auto t2 = std::powf(t, 2);
				auto t3 = std::powf(t, 3);
				auto dt = input[i] - input[i - 1];

				std::array<T, 4> values {{
					fastgltf::getAccessorElement<T>(asset, outputAccessor, 3 * (i - 1) + 1),
					fastgltf::getAccessorElement<T>(asset, outputAccessor, 3 * (i - 1) + 2),
					fastgltf::getAccessorElement<T>(asset, outputAccessor, 3 * (i + 0) + 1),
					fastgltf::getAccessorElement<T>(asset, outputAccessor, 3 * (i + 0) + 0),
				}};

				auto v = values[0] * (2 * t3 - 3 * t2 + 1)
					   + values[1] * (t3 - 2 * t2 + t) * dt
					   + values[2] * (-2 * t3 + 3 * t2)
					   + values[3] * (t3 - t2) * dt;

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
	std::vector<AnimationSampler> samplers;
};

struct Asset {
	fastgltf::Asset asset;

	std::vector<Mesh> meshes;
	std::vector<PrimitiveBuffers> primitiveBuffers;
	std::unique_ptr<ScopedBuffer> primitiveBuffer;

	std::vector<Animation> animations;
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
