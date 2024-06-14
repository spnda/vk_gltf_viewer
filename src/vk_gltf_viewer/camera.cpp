#include <variant>

#include <vk_gltf_viewer/device.hpp>
#include <vk_gltf_viewer/camera.hpp>

#include <fastgltf/types.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

glm::mat4 getCameraProjectionMatrix(fastgltf::Camera& camera, VkExtent2D framebufferExtent) {
	ZoneScoped;
	// The following projection matrices do not use the math defined by the glTF spec here:
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#projection-matrices
	// The reason is that Vulkan uses a different depth range to OpenGL, which has to be accounted.
	// Therefore, we always use the appropriate _ZO glm functions.
	return std::visit(fastgltf::visitor {
		[&](fastgltf::Camera::Perspective& perspective) {
			assert(framebufferExtent.width != 0 && framebufferExtent.height != 0);
			auto aspectRatio = perspective.aspectRatio.value_or(
				static_cast<float>(framebufferExtent.width) / static_cast<float>(framebufferExtent.height));

			if (perspective.zfar.has_value()) {
				return glm::perspectiveRH_ZO(perspective.yfov, aspectRatio, perspective.znear, *perspective.zfar);
			} else {
				return glm::infinitePerspectiveRH_ZO(perspective.yfov, aspectRatio, perspective.znear);
			}
		},
		[&](fastgltf::Camera::Orthographic& orthographic) {
			return glm::orthoRH_ZO(-orthographic.xmag, orthographic.xmag,
								   -orthographic.ymag, orthographic.ymag,
								   orthographic.znear, orthographic.zfar);
		},
	}, camera.camera);
}

glm::mat4 reverseDepth(glm::mat4 projection) {
	// We use reversed Z, see https://iolite-engine.com/blog_posts/reverse_z_cheatsheet
	// This converts any projection matrix into using reversed Z.
	constexpr glm::mat4 reverseZ {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, -1.f, 0.0f,
		0.0f, 0.0f, 1.0f, 1.0f
	};
	return reverseZ * projection;
}

std::array<glm::vec3, 8> getFrustumCorners(glm::mat4 viewProjection) {
	ZoneScoped;
	const auto inv = glm::inverse(viewProjection);

	// The inner loop is the z coordinate, as we want the first 4 elements to be the near plane,
	// and the last 4 elements the far plane.
	std::array<glm::vec3, 8> corners {};
	std::size_t i = 0;
	for (std::uint32_t z = 0; z < 2; ++z) {
		for (std::uint32_t y = 0; y < 2; ++y) {
			for (std::uint32_t x = 0; x < 2; ++x) {
				const auto pt =
					inv * glm::vec4(2.f * glm::fvec2(x, y) - 1.0f, z, 1.0f);
				corners[i++] = glm::vec3(pt) / pt.w;
			}
		}
	}
	return corners;
}

void generateCameraFrustum(const glm::mat4x4& vp, std::array<glm::vec4, 6>& frustum) {
	ZoneScoped;
	// This plane extraction code is from https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
	auto& p = frustum;
	for (glm::length_t i = 0; i < 4; ++i) { p[0][i] = vp[i][3] + vp[i][0]; }
	for (glm::length_t i = 0; i < 4; ++i) { p[1][i] = vp[i][3] - vp[i][0]; }
	for (glm::length_t i = 0; i < 4; ++i) { p[2][i] = vp[i][3] + vp[i][1]; }
	for (glm::length_t i = 0; i < 4; ++i) { p[3][i] = vp[i][3] - vp[i][1]; }
	for (glm::length_t i = 0; i < 4; ++i) { p[4][i] = vp[i][3] + vp[i][2]; }
	for (glm::length_t i = 0; i < 4; ++i) { p[5][i] = vp[i][3] - vp[i][2]; }
	for (auto& plane : p) {
		plane /= glm::length(glm::vec3(plane));
		plane.w = -plane.w;
	}
}

Camera::Camera(const Device& _device, std::size_t frameOverlap) : device(_device) {
	ZoneScoped;
	cameraBuffers.resize(frameOverlap);

	for (std::size_t i = 0; auto& cameraBuffer : cameraBuffers) {
		const VmaAllocationCreateInfo allocationInfo {
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		};
		const VkBufferCreateInfo bufferInfo {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(glsl::Camera),
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		};
		cameraBuffer = std::make_unique<ScopedBuffer>(device.get(), &bufferInfo, &allocationInfo);
		vk::setDebugUtilsName(device.get(), cameraBuffer->getHandle(), fmt::format("Camera buffer {}", i++));
	}
}

Camera::~Camera() noexcept = default;

void Camera::updateCamera(std::size_t currentFrame, GLFWwindow* window, double deltaTime, glm::u32vec2 framebufferExtent) {
	ScopedMap<glsl::Camera> mappedCamera(*cameraBuffers[currentFrame]);

	// Update the acceleration vector based on keyboard input
	{
		auto& acc = accelerationVector;
		acc = glm::vec3(0.0f);
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			acc += direction;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			acc -= direction;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			acc += glm::normalize(glm::cross(direction, cameraUp));
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			acc -= glm::normalize(glm::cross(direction, cameraUp));
		}
		if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
			acc += cameraUp;
		}
		if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
			acc -= cameraUp;
		}
	}

	// Update the direction vector based on the mouse position
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
		if (state == GLFW_PRESS) {
			auto offset = glm::dvec2(xpos - lastCursorPos.x, lastCursorPos.y - ypos);
			lastCursorPos = {xpos, ypos};
			offset *= 0.1;

			yaw += static_cast<float>(offset.x);
			pitch += static_cast<float>(offset.y);
			pitch = glm::clamp(pitch, -89.0f, 89.0f);

			direction.x = std::cosf(glm::radians(yaw)) * std::cosf(glm::radians(pitch));
			direction.y = std::sinf(glm::radians(pitch));
			direction.z = std::sinf(glm::radians(yaw)) * std::cosf(glm::radians(pitch));
			direction = glm::normalize(direction);
		} else {
			lastCursorPos = {xpos, ypos};
		}
	}

	float mult = speedMultiplier;
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		mult *= 10.0f;
	}

	velocity += (accelerationVector * mult * float(deltaTime)); // v = a * t
	// Lerp the velocity to 0, adding deceleration.
	velocity = velocity + (5.0f * float(deltaTime)) * (-velocity);
	// Add the velocity into the position
	position += velocity * float(deltaTime);

	auto view = glm::lookAtRH(position, position + direction, cameraUp);

	// glm::perspectiveRH_ZO is correct, see https://johannesugb.github.io/gpu-programming/setting-up-a-proper-vulkan-projection-matrix/
	static constexpr auto zNear = 0.1f;
	static constexpr auto zFar = 1000.0f;
	static constexpr auto fov = glm::radians(75.0f);
	const auto aspectRatio = static_cast<float>(framebufferExtent.x) / static_cast<float>(framebufferExtent.y);
	auto projectionMatrix = glm::perspectiveRH_ZO(fov, aspectRatio, zNear, zFar);
	projectionMatrix[1][1] *= -1;

	auto& camera = *mappedCamera.get();
	camera.prevViewProjection = camera.viewProjection;
	camera.prevOcclusionViewProjection = camera.occlusionViewProjection;

	// mappedCamera.get()->projection = reverseDepth(projectionMatrix);
	mappedCamera.get()->viewProjection = reverseDepth(projectionMatrix) * view;
	viewProjection = mappedCamera.get()->viewProjection;
	//mappedCamera.get()->views[0].viewProjection = reverseDepth(projectionMatrix) * mappedCamera.get()->view;

	if (!freezeCullingMatrix)
		mappedCamera.get()->occlusionViewProjection = viewProjection;

	if (!freezeCameraFrustum)
		generateCameraFrustum(mappedCamera.get()->viewProjection, mappedCamera.get()->frustum);
}
