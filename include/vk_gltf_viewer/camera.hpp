#pragma once

#include <utility>

#include <vk_gltf_viewer/buffer.hpp>

#include <GLFW/glfw3.h>

#include <mesh_common.h.glsl>

struct Camera : public glsl::Camera {
	glm::vec3 accelerationVector = glm::vec3(0.f);
	glm::vec3 velocity = glm::vec3(0.f);
	glm::vec3 position = glm::vec3(0.f, 0.f, 5.f);

	glm::vec3 direction = glm::vec3(0.f, 0.f, -1.f);
	glm::dvec2 lastCursorPos = glm::dvec2(0.f);
	float yaw = 0.f;
	float pitch = 0.f;

	bool freezeCameraFrustum = false;
	bool freezeCullingMatrix = false;
	float speedMultiplier = 5.f;

	static constexpr auto cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
	static constexpr auto cameraRight = glm::vec3(0.0f, 0.0f, 1.0f);

	explicit Camera();
	~Camera() noexcept;

	/** Updates the camera position and rotation using the last known window inputs */
	void updateCamera(GLFWwindow* window, double deltaTime, glm::u32vec2 framebufferExtent);

	[[nodiscard]] glm::vec3 getPosition() const noexcept {
		return position;
	}
};
