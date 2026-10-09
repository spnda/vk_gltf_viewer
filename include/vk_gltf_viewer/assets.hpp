#pragma once

#include <algorithm>
#include <filesystem>
#include <span>

#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/buffer.hpp>
#include <graphics/renderer.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include <mesh_common.h>

/** The buffer handles corresponding to the buffers in each shaders::Primitive. */
struct PrimitiveBuffers {
	std::unique_ptr<ScopedBuffer> vertexIndexBuffer;
	std::unique_ptr<ScopedBuffer> primitiveIndexBuffer;
	std::unique_ptr<ScopedBuffer> vertexBuffer;
	std::unique_ptr<ScopedBuffer> meshletBuffer;

	std::uint32_t meshletCount;
};

struct mesh_primitive_indices {
	std::vector<std::uint64_t> primitive_indices;
};

template <fastgltf::AnimationPath path>
struct animation_sampler_traits {};

template <> struct animation_sampler_traits<fastgltf::AnimationPath::Translation> { using output_type = fastgltf::math::fvec3; };
template <> struct animation_sampler_traits<fastgltf::AnimationPath::Scale> { using output_type = fastgltf::math::fvec3; };
template <> struct animation_sampler_traits<fastgltf::AnimationPath::Rotation> { using output_type = fastgltf::math::fquat; };

struct animation_sampler_t {
	std::vector<float> input;

	fastgltf::AnimationInterpolation interpolation;
	std::size_t component_count;
	std::size_t output_count;
	std::vector<float> values;

	explicit animation_sampler_t(const fastgltf::Asset& asset, const fastgltf::AnimationSampler& sampler) : interpolation(sampler.interpolation) {
		ZoneScoped;
		auto& output_accessor = asset.accessors[sampler.outputAccessor];
		output_count = output_accessor.count;
		component_count = fastgltf::getNumComponents(output_accessor.type);
	}

	template <fastgltf::AnimationPath path>
	auto sample(float time) {
		ZoneScoped;
		using T = typename animation_sampler_traits<path>::output_type;

		time = std::fmod(time, input.back()); // Ugly hack to loop animations

		auto it = std::ranges::lower_bound(input, time);
		if (it == input.cbegin()) {
			if (interpolation == fastgltf::AnimationInterpolation::CubicSpline)
				return T::fromPointer(&values[(1) * component_count]);
			return T::fromPointer(&values[0]);
		}
		if (it == input.cend()) {
			if (interpolation == fastgltf::AnimationInterpolation::CubicSpline)
				return T::fromPointer(&values[(output_count - 2) * component_count]);
			return T::fromPointer(&values[(output_count - 1) * component_count]);
		}

		auto i = std::distance(input.begin(), it);
		auto t = (time - input[i - 1]) / (input[i] - input[i - 1]);

		switch (interpolation) {
			using enum fastgltf::AnimationInterpolation;
			case Step:
				return T::fromPointer(&values[(i - 1) * component_count]);
			case Linear: {
				auto vk = T::fromPointer(&values[(i - 1) * component_count]);
				auto vk1 = T::fromPointer(&values[i * component_count]);

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
					T::fromPointer(&values[(3 * (i - 1) + 1) * component_count]),
					T::fromPointer(&values[(3 * (i - 1) + 2) * component_count]),
					T::fromPointer(&values[(3 * (i + 0) + 1) * component_count]),
					T::fromPointer(&values[(3 * (i + 0) + 0) * component_count]),
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

struct animation_t {
	std::vector<fastgltf::AnimationChannel> channels;
	std::vector<animation_sampler_t> samplers;
};

struct DrawBuffers {
	bool isMeshletBufferBuilt = false;
	std::unique_ptr<ScopedBuffer> meshletDrawBuffer;
	std::unique_ptr<ScopedBuffer> transformBuffer;
};

class asset_load_task;

/**
 * A World represents the whole world we render at once. This holds everything from
 * the glTF assets, to the GPU buffers used for drawing.
 */
struct World {
	std::reference_wrapper<Device> device;

	std::vector<std::shared_ptr<fastgltf::Asset>> assets;

	std::vector<mesh_primitive_indices> meshes;
	std::vector<PrimitiveBuffers> primitiveBuffers;
	std::unique_ptr<ScopedBuffer> primitiveBuffer;

	std::vector<shaders::material_t> materials;
	std::unique_ptr<ScopedBuffer> materialBuffer;

	std::vector<animation_t> animations;
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

	void addAsset(const std::shared_ptr<asset_load_task>& task);
	void iterateNode(std::size_t nodeIndex, fastgltf::math::fmat4x4 parent, std::function<void(fastgltf::Node&, const fastgltf::math::fmat4x4&)>& callback);
	void updateDrawBuffers(std::size_t frameIndex, float dt);
};

class asset_load_task : public ExceptionTaskSet {
	friend struct World;
	friend class application;

	std::shared_ptr<graphics::renderer> renderer;
	std::filesystem::path asset_path;

	std::shared_ptr<fastgltf::Asset> asset;
	std::vector<mesh_primitive_indices> meshes;
	std::vector<std::shared_ptr<graphics::mesh_t>> primitives;
	std::vector<animation_t> animations;
	std::vector<std::shared_ptr<graphics::texture_t>> textures;
	std::vector<shaders::material_t> materials;

	std::shared_ptr<fastgltf::Asset> loadGltf();

public:
	explicit asset_load_task(std::shared_ptr<graphics::renderer> renderer, std::filesystem::path path);

	void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) override;
};
