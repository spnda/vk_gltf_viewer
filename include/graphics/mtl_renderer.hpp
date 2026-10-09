#pragma once

#include <exception>

#include <graphics/renderer.hpp>

#include <Foundation/NSSharedPtr.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLSampler.hpp>
#include <MetalFX/MTLFXTemporalScaler.hpp>
#include <MetalFX/MTLFXSpatialScaler.hpp>
#include <QuartzCore/CAMetalLayer.hpp>

#include <graphics/resource_table.hpp>
#include <graphics/imgui/mtl_renderer.hpp>

namespace graphics::metal {
class meshlet_renderer;

class mtl_buffer : public graphics::buffer_t {
	friend meshlet_renderer;
	MTL::Buffer* buffer = nullptr;

public:
	explicit mtl_buffer() = default;
	~mtl_buffer() noexcept = default;
};

class mtl_sampler final : public sampler_t, public NS::SharedPtr<MTL::SamplerState> {
public:
	explicit mtl_sampler(MTL::SamplerState* sampler)
			: NS::SharedPtr<MTL::SamplerState>(NS::TransferPtr(sampler)) {}
};

class mtl_image final : public image_t, public NS::SharedPtr<MTL::Texture> {
public:
	explicit mtl_image(MTL::Texture* texture)
			: NS::SharedPtr<MTL::Texture>(NS::TransferPtr(texture)) {}
};

class mtl_texture final : public texture_t {
	std::shared_ptr<mtl_resource_table> resource_table;
	std::shared_ptr<mtl_image> image;
	std::shared_ptr<mtl_sampler> sampler;

public:
	explicit mtl_texture(std::shared_ptr<mtl_resource_table> resourceTable, std::shared_ptr<mtl_image> image, std::shared_ptr<mtl_sampler> sampler)
			: texture_t(resourceTable->allocate_sampled_image(*image.get(), *sampler.get())), resource_table(std::move(resourceTable)), image(std::move(image)), sampler(std::move(sampler)) {}
	~mtl_texture() override {
		resource_table->remove_sampled_image_handle(get_handle());
	}
};

class meshlet_mesh : public graphics::mesh_t {
public:
	NS::SharedPtr<MTL::Buffer> vertex_index_buffer;
	NS::SharedPtr<MTL::Buffer> primitive_index_buffer;
	NS::SharedPtr<MTL::Buffer> meshlet_buffer;
	NS::SharedPtr<MTL::Buffer> position_buffer;
	NS::SharedPtr<MTL::Buffer> vertex_buffer;
	std::array<NS::SharedPtr<MTL::Buffer>, shaders::max_uv_sets> uv_buffers;

	glm::fvec3 aabb_extents;
	glm::fvec3 aabb_center;

	std::uint32_t meshlet_count;
	std::uint32_t material_index;

	explicit meshlet_mesh() = default;
	~meshlet_mesh() noexcept = default;
};

struct meshlet_scene_mesh {
	std::shared_ptr<meshlet_mesh> mesh;
	shaders::primitive_t primitive;

	explicit meshlet_scene_mesh(std::shared_ptr<meshlet_mesh> mesh, shaders::primitive_t primitive)
			: mesh(std::move(mesh)), primitive(primitive) {}
	meshlet_scene_mesh(meshlet_scene_mesh&& other) noexcept
			: mesh(std::move(other.mesh)), primitive(other.primitive) {}
};

struct meshlet_draw_buffers {
	NS::SharedPtr<MTL::Buffer> primitive_buffer;
	NS::SharedPtr<MTL::Buffer> meshlet_draw_buffer;
	NS::SharedPtr<MTL::Buffer> transform_buffer;
};

class meshlet_scene : public graphics::scene_t {
public:
	NS::SharedPtr<MTL::Device> device;

	std::vector<meshlet_scene_mesh> meshes;
	std::vector<shaders::primitive_transform_t> transforms;
	std::vector<shaders::meshlet_draw_t> meshlet_draws;

	std::vector<meshlet_draw_buffers> draw_buffers;

	explicit meshlet_scene(NS::SharedPtr<MTL::Device> device, std::size_t frame_overlap) : device(std::move(device)) {
		draw_buffers.resize(frame_overlap);
	}
	~meshlet_scene() noexcept override = default;

	instance_index add_mesh_instance(std::shared_ptr<mesh_t> mesh) override;
	void update_transform(instance_index instance, glm::fmat4x4 transform) override;

	void update_draw_buffers(std::size_t frame_index);
};

CA::MetalLayer* create_metal_layer(GLFWwindow* window);

struct visbuffer_pass {
	NS::SharedPtr<MTL::RenderPipelineState> visbuffer_pipeline;
	NS::SharedPtr<MTL::RenderPipelineState> gbuffer_tile_pipeline;
	NS::SharedPtr<MTL::DepthStencilState> depth_state;

	NS::SharedPtr<MTL::Texture> albedo_texture;
	NS::SharedPtr<MTL::Texture> normal_texture;
	NS::SharedPtr<MTL::Texture> metallic_roughness_texture;
	NS::SharedPtr<MTL::Texture> motion_vector_texture;
	NS::SharedPtr<MTL::Texture> depth_texture;

	void init_pass(meshlet_renderer& renderer);
	void update_resolution(meshlet_renderer& renderer);
};

struct shading_pass {
	NS::SharedPtr<MTL::RenderPipelineState> shading_pass_pipeline;

	NS::SharedPtr<MTL::RenderPipelineState> luminance_histogram_pipeline;
	NS::SharedPtr<MTL::Buffer> luminance_histogram;
	NS::SharedPtr<MTL::ComputePipelineState> average_luminance_pipeline;

	NS::SharedPtr<MTL::RenderPipelineState> tonemap_pipeline;

	void init_pass(meshlet_renderer& renderer);
	void update_resolution(meshlet_renderer& renderer);
};

class meshlet_renderer : public graphics::renderer {
	friend std::shared_ptr<renderer> graphics::renderer::create_renderer(GLFWwindow* window);
	friend visbuffer_pass;
	friend shading_pass;

	/** This comes first so that it wraps around the entire lifetime of the renderer object */
	// NS::SharedPtr<NS::AutoreleasePool> pool;

	NS::SharedPtr<MTL::Device> device;

	CA::MetalLayer* layer;
	NS::SharedPtr<MTL::CommandQueue> command_queue;
	dispatch_semaphore_t draw_semaphore;
	std::exception_ptr command_buffer_exception;

	std::unique_ptr<imgui::imgui_renderer> imgui_renderer;

	float render_scale = 1.f;
	glm::u32vec2 render_resolution;
	scaling_modes_e scaling_mode = scaling_modes_e::none;
	NS::SharedPtr<MTLFX::TemporalScaler> temporal_scaler;
	bool first_frame_temporal = false;

	// TODO: Is it really necessary to have an additional upscaled texture?
	// MetalFX will not allow us to upscale into the drawable directly, since it's not
	// a Private texture... for some reason.
	NS::SharedPtr<MTL::Texture> color_texture;
	NS::SharedPtr<MTL::Texture> color_output_texture;

	std::shared_ptr<mtl_resource_table> resource_table;

	NS::SharedPtr<MTL::Library> global_library;

	std::vector<NS::SharedPtr<MTL::Buffer>> camera_buffers;

	std::vector<shaders::material_t> materials;
	std::vector<NS::SharedPtr<MTL::Buffer>> material_buffers;

	std::shared_ptr<mtl_sampler> default_sampler;

	visbuffer_pass visbuffer_pass;
	shading_pass shading_pass;

	std::array<shaders::color_space_e, 5> internal_color_spaces {{
		shaders::color_space_e::srgb_linear,
		shaders::color_space_e::display_p3_linear,
		shaders::color_space_e::bt709_linear,
		shaders::color_space_e::bt2020_linear,
		shaders::color_space_e::cie_1931,
	}};
	std::array<shaders::color_space_e, 8> output_color_spaces {{
		shaders::color_space_e::srgb_nonlinear,
		shaders::color_space_e::bt709_nonlinear,
		shaders::color_space_e::display_p3_nonlinear,
		shaders::color_space_e::display_p3_pq,
		shaders::color_space_e::display_p3_hlg,
		shaders::color_space_e::bt2020_nonlinear,
		shaders::color_space_e::bt2100_pq,
		shaders::color_space_e::bt2100_hlg,
	}};
	shaders::color_space_e current_internal_color_space = shaders::color_space_e::bt2020_linear;
	shaders::color_space_e current_output_color_space = shaders::color_space_e::display_p3_pq;

	void create_temporal_scaler();
	float get_max_edr_value() const;
	float get_display_headroom() const;

public:
	explicit meshlet_renderer(GLFWwindow* window);
	~meshlet_renderer() noexcept override;

	std::unique_ptr<buffer_t> create_unique_buffer() override;
	std::shared_ptr<buffer_t> create_shared_buffer() override;

	material_index create_material(shaders::material_t material) override;

	std::shared_ptr<mesh_t> create_shared_mesh(
		std::span<const glm::fvec3> positions, std::span<const shaders::vertex_t> vertices,
		std::array<std::span<const glm::fvec2>, shaders::max_uv_sets> uvs, std::span<const index_t> indices,
		glm::fvec3 aabb_center, glm::fvec3 aabb_extents,
		material_index material_index) override;

	std::shared_ptr<scene_t> create_shared_scene() override;

	std::shared_ptr<image_t> create_shared_image(
		std::span<std::byte> imageData, glm::u32vec2 extents) override;

	std::shared_ptr<sampler_t> get_default_sampler() override;
	std::shared_ptr<sampler_t> create_shared_sampler(
		const fastgltf::Sampler& sampler) override;
	std::shared_ptr<texture_t> create_shared_texture(
		std::shared_ptr<image_t> image, std::shared_ptr<sampler_t> sampler) override;

	bool can_render() override {
		return true; // TODO: Detect window being minimized or sth
	}

	void update_resolution(glm::u32vec2 resolution) override;
	auto get_window_resolution() const noexcept -> glm::u32vec2;
	auto get_render_resolution() const noexcept -> glm::u32vec2 override;

	void prepare_frame(std::size_t frame_index) override;
	bool draw(std::size_t frame_index, scene_t& world,
			  const shaders::camera_t& camera, float dt) override;

	std::vector<scaling_modes_e> get_scaling_modes() const override;
	std::vector<scaling_preset_t> get_scaling_presets(scaling_modes_e mode) const override;
	void use_upscaler(scaling_modes_e scaler, scaling_preset_t preset) override;

	std::span<const shaders::color_space_e> get_internal_color_spaces() const override;
	void set_internal_color_space(shaders::color_space_e space) override;
	shaders::color_space_e get_internal_color_space() const override;

	std::span<const shaders::color_space_e> get_output_color_spaces() const override;
	void set_output_color_space(shaders::color_space_e space) override;
	shaders::color_space_e get_output_color_space() const override;
};
}
