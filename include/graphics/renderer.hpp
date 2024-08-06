#pragma once

#include <memory>
#include <span>

#include <GLFW/glfw3.h>

#include <mesh_common.h>
#include <resource_table.h>

namespace graphics {
	using index_t = std::uint32_t; // TODO: Support dynamic index bit width?

	/** The amount of frames we render ahead, regardless of what the windowing system or GPU supports */
	static constexpr std::uint32_t frameOverlap = 3;

	class Buffer {

	};

	class Mesh {

	};

	using MaterialIndex = std::uint32_t;
	using InstanceIndex = std::uint32_t;

	class Scene {
	public:
		Scene() noexcept = default;
		virtual ~Scene() noexcept = default;

		[[nodiscard]] virtual InstanceIndex addMeshInstance(std::shared_ptr<Mesh> mesh) = 0;
		virtual void updateTransform(InstanceIndex instance, glm::fmat4x4 transform) = 0;
	};

	/**
	 * The abstracted renderer interface.
	 */
	class Renderer {
	public:
		Renderer() noexcept = default;
		virtual ~Renderer() noexcept = default;

		[[nodiscard]] static std::shared_ptr<Renderer> createRenderer(GLFWwindow* window);

		[[nodiscard]] virtual std::unique_ptr<Buffer> createUniqueBuffer() = 0;
		[[nodiscard]] virtual std::shared_ptr<Buffer> createSharedBuffer() = 0;

		[[nodiscard]] virtual MaterialIndex getDefaultMaterialIndex() const noexcept {
			return 0;
		}
		[[nodiscard]] virtual MaterialIndex createMaterial(shaders::Material material) = 0;

		[[nodiscard]] virtual std::shared_ptr<Mesh> createSharedMesh(
				std::span<shaders::Vertex> vertexBuffer, std::span<index_t> indexBuffer,
				glm::fvec3 aabbCenter, glm::fvec3 aabbExtents,
				MaterialIndex materialIndex) = 0;

		[[nodiscard]] virtual std::shared_ptr<Scene> createSharedScene() = 0;

		[[nodiscard]] virtual shaders::ResourceTableHandle createSampledTextureHandle() = 0;
		[[nodiscard]] virtual shaders::ResourceTableHandle createStorageTextureHandle() = 0;

		/**
		 * If this returns false, the window might be minimised or being resized, forcing us to pause rendering shortly.
		 * In that case, glfwWaitEvents should be used.
		 */
		[[nodiscard]] virtual bool canRender() = 0;

		virtual void updateResolution(glm::u32vec2 resolution) = 0;
		/**
		 * Returns the resolution at which this renders the scene. Note that this might not be
		 * the same resolution the window has, since the renderer might use some upscaling
		 * technique.
		 */
		[[nodiscard]] virtual glm::u32vec2 getRenderResolution() const noexcept = 0;

		virtual void prepareFrame(std::size_t frameIndex) = 0;
		virtual bool draw(std::size_t frameIndex, Scene& scene,
						  const shaders::Camera& camera, float dt) = 0;
	};
}
