#pragma once

#include <memory>
#include <span>

#include <GLFW/glfw3.h>

#include <mesh_common.h.glsl>
#include <resource_table.h.glsl>

namespace graphics {
	using index_t = std::uint32_t; // TODO: Support dynamic index bit width?

	class Buffer {

	};

	class Mesh {

	};

	using InstanceIndex = std::uint32_t;

	class Scene {
	public:
		Scene() noexcept = default;
		virtual ~Scene() noexcept = default;

		virtual InstanceIndex addMesh(std::shared_ptr<Mesh> mesh) = 0;
		virtual void updateTransform(InstanceIndex instance, glm::fmat4x4 transform) = 0;
	};

	/**
	 * The abstracted renderer interface.
	 */
	class Renderer {
	public:
		Renderer() noexcept = default;
		virtual ~Renderer() noexcept = default;

		static std::shared_ptr<Renderer> createRenderer(GLFWwindow* window);

		virtual std::unique_ptr<Buffer> createUniqueBuffer() = 0;
		virtual std::shared_ptr<Buffer> createSharedBuffer() = 0;

		virtual std::shared_ptr<Mesh> createSharedMesh(std::span<glsl::Vertex> vertexBuffer, std::span<index_t> indexBuffer) = 0;

		virtual std::shared_ptr<Scene> createSharedScene() = 0;

		virtual glsl::ResourceTableHandle createSampledTextureHandle() = 0;
		virtual glsl::ResourceTableHandle createStorageTextureHandle() = 0;

		/**
		 * If this returns false, the window might be minimised or being resized, forcing us to pause rendering shortly.
		 * In that case, glfwWaitEvents should be used.
		 */
		virtual bool canRender() = 0;

		virtual void updateResolution(glm::u32vec2 resolution) = 0;

		virtual void prepareFrame(std::size_t frameIndex) = 0;
		virtual bool draw(std::size_t frameIndex, Scene& scene, float dt) = 0;
	};
}
