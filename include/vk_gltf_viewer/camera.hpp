#pragma once

#include <utility>

#include <vk_gltf_viewer/buffer.hpp>

#include <GLFW/glfw3.h>

#include <mesh_common.glsl.h>

class Camera {
	std::reference_wrapper<const Device> device;

	std::vector<std::unique_ptr<ScopedBuffer>> cameraBuffers;

	glm::vec3 accelerationVector = glm::vec3(0.f);
	glm::vec3 velocity = glm::vec3(0.f);
	glm::vec3 position = glm::vec3(0.f);

	glm::vec3 direction = glm::vec3(0.f, 0.f, -1.f);
	glm::dvec2 lastCursorPos = glm::dvec2(0.f);
	float yaw = 0.f;
	float pitch = 0.f;

	float speedMultiplier = 5.f;

	static constexpr auto cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
	static constexpr auto cameraRight = glm::vec3(0.0f, 0.0f, 1.0f);

public:
	explicit Camera(const Device& device, std::size_t frameOverlap);
	~Camera() noexcept;

	bool freezeCameraFrustum = false;

	/** Updates the camera position and rotation using the last known window inputs */
	void updateCamera(std::size_t currentFrame, GLFWwindow* window, double deltaTime, VkExtent2D framebufferExtent);

	[[nodiscard]] VkDeviceAddress getCameraDeviceAddress(std::size_t currentFrame) const noexcept {
		return cameraBuffers[currentFrame]->getDeviceAddress();
	}

	[[nodiscard]] ScopedBuffer& getCameraBuffer(std::size_t currentFrame) const noexcept {
		return *cameraBuffers[currentFrame];
	}

	[[nodiscard]] glm::vec3 getPosition() const noexcept {
		return position;
	}

	glm::mat4 viewProjection;
};
