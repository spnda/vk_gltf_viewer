#pragma once

#include <deque>
#include <ranges>
#include <vector>

#include <vulkan/vk.hpp>
#include <vulkan/vma.hpp>
#include <vulkan/debug_utils.hpp>
#include <VkBootstrap.h>

#include <TaskScheduler.h>

#include <tracy/TracyVulkan.hpp>

#include <meshoptimizer.h>

#include <glfw/glfw3.h>

#include <glm/mat4x4.hpp>

#include <fastgltf/types.hpp>

#include <vk_gltf_viewer/imgui_renderer.hpp>
#include <vk_gltf_viewer/fence_pool.hpp>

extern enki::TaskScheduler taskScheduler;

struct FrameSyncData {
    VkSemaphore imageAvailable;
    VkSemaphore renderingFinished;
    VkFence presentFinished;
};

struct FrameCommandPools {
    VkCommandPool pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
};

static constexpr const std::size_t frameOverlap = 2;

class FileLoadTask;

struct PerFrameCameraBuffer {
	VkBuffer handle;
	VmaAllocation allocation;

	VkDescriptorSet cameraSet;
};

struct CameraMovement {
	glm::vec3 accelerationVector = glm::vec3(0.0f);
	glm::vec3 velocity = glm::vec3(0.0f);
	glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::dvec2 lastCursorPosition = glm::dvec2(0.0f);
	glm::vec3 direction = glm::vec3(0.0f, 0.0f, -1.0f);
	float yaw = -90.0f;
	float pitch = 0.0f;
	bool firstMouse = false;

	float speedMultiplier = 0.05f;
};

#include <mesh_common.glsl.h>

struct Primitive {
	std::uint32_t descOffset;
	std::uint32_t vertexIndicesOffset;
	std::uint32_t triangleIndicesOffset;
	std::uint32_t verticesOffset;

	std::size_t meshlet_count;
	std::uint32_t materialIndex;
};

struct Mesh {
	std::vector<Primitive> primitives;
};

struct MeshBuffers {
	VkBuffer descHandle = VK_NULL_HANDLE;
	VmaAllocation descAllocation = VK_NULL_HANDLE;

	VkBuffer vertexIndiciesHandle = VK_NULL_HANDLE;
	VmaAllocation vertexIndiciesAllocation = VK_NULL_HANDLE;

	VkBuffer triangleIndicesHandle = VK_NULL_HANDLE;
	VmaAllocation triangleIndicesAllocation = VK_NULL_HANDLE;

	VkBuffer verticesHandle = VK_NULL_HANDLE;
	VmaAllocation verticesAllocation = VK_NULL_HANDLE;

	std::vector<VkDescriptorSet> descriptors;
};

/** Temporary object used when generating all of the mesh data for a glTF */
struct GlobalMeshData {
	std::vector<glsl::Vertex> globalVertices;
	std::vector<glsl::Meshlet> globalMeshlets;
	std::vector<std::uint32_t> globalMeshletVertices;
	std::vector<std::uint8_t> globalMeshletTriangles;
	std::mutex lock;
};

struct FrameDrawCommandBuffers {
	VkBuffer primitiveDrawHandle;
	VmaAllocation primitiveDrawAllocation;
	VkDeviceSize primitiveDrawBufferSize;

	VkBuffer aabbDrawHandle;
	VmaAllocation aabbDrawAllocation;
	VkDeviceSize aabbDrawBufferSize;

	std::uint32_t drawCount;
};

struct SampledImage {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView imageView = VK_NULL_HANDLE;

	VkExtent2D size = {0, 0};
};

/** Deletion queue, as used by vkguide.dev to ensure proper destruction order of global Vulkan objects */
class DeletionQueue {
	friend struct Viewer;
	std::deque<std::function<void()>> deletors;

public:
	void push(std::function<void()>&& function) {
		deletors.emplace_back(function);
	}

	void flush() {
		ZoneScoped;
		for (auto& func : deletors | std::views::reverse) {
			func();
		}
		deletors.clear();
	}
};

/** DeletionQueue that uses a timeline semaphore to destroy GPU objects when they have actually finished */
struct TimelineDeletionQueue {
	friend struct Viewer;
	VkDevice device;
	VkSemaphore timelineSemaphore;
	std::uint64_t hostValue = 0;

	struct Entry {
		std::uint64_t timelineValue = 0;
		std::function<void()> deletion;
	};
	std::vector<Entry> deletors;

public:
	void create(VkDevice nDevice) {
		ZoneScoped;
		device = nDevice;

		const VkSemaphoreTypeCreateInfo timelineCreateInfo {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
			.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
			.initialValue = 0,
		};

		const VkSemaphoreCreateInfo createInfo {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = &timelineCreateInfo,
			.flags = 0,
		};

		auto result = vkCreateSemaphore(device, &createInfo, nullptr, &timelineSemaphore);
		vk::checkResult(result, "Failed to create timeline semaphore for deletion queue: {}");
		vk::setDebugUtilsName(device, timelineSemaphore, "Deletion queue timeline semaphore");
	}

	void destroy() {
		vkDestroySemaphore(device, timelineSemaphore, nullptr);
		timelineSemaphore = VK_NULL_HANDLE;
	}

	[[nodiscard]] VkSemaphore getSemaphoreHandle() const noexcept {
		return timelineSemaphore;
	}

	void push(std::function<void()>&& function) {
		deletors.emplace_back(hostValue, std::move(function));
	}

	[[nodiscard]] std::uint64_t nextValue() {
		return ++hostValue;
	}

	/** Function to be called at the start of every frame, which deletes objects if they're old enough */
	void check() {
		ZoneScoped;
		std::uint64_t currentValue;
		auto result = vkGetSemaphoreCounterValue(device, timelineSemaphore, &currentValue);
		vk::checkResult(result, "Failed to get timeline semaphore counter value: {}");
		for (auto it = deletors.begin(); it != deletors.end();) {
			auto& [timelineValue, deletion] = *it;
			if (timelineValue < currentValue) {
				deletion();
				it = deletors.erase(it);
			} else {
				++it;
			}
		}
	}

	void flush() {
		ZoneScoped;
		for (auto& entry : deletors) {
			entry.deletion();
		}
		deletors.clear();
	}
};

struct Queue {
	VkQueue handle = VK_NULL_HANDLE;
	std::unique_ptr<std::mutex> lock; // Can't hold the object in a vector otherwise.
};

struct Viewer {
    vkb::Instance instance;
    vkb::Device device;
	VmaAllocator allocator = VK_NULL_HANDLE;
	TracyVkCtx tracyCtx = nullptr;

	FencePool fencePool;

	std::uint32_t graphicsQueueFamily = VK_QUEUE_FAMILY_IGNORED;
    Queue graphicsQueue;
	std::uint32_t transferQueueFamily = VK_QUEUE_FAMILY_IGNORED;
	std::vector<Queue> transferQueues;

	Queue& getNextTransferQueueHandle() {
		static std::atomic<std::size_t> idx = 0;
		return transferQueues[idx++ % transferQueues.size()];
	}

	struct CommandPool {
		VkCommandPool pool;
		// TODO: Find some mechanism to allow using multiple command buffers on one thread.
		//       Perhaps we can use up to N command buffers and fences for multiple submits from a single thread.
		VkCommandBuffer buffer;
	};
	std::vector<CommandPool> uploadCommandPools; //< Command pools for transfer queues
	std::vector<CommandPool> graphicsCommandPools; //< Command pools for the main graphics queue

    GLFWwindow* window = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    vkb::Swapchain swapchain;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    bool swapchainNeedsRebuild = false;

	VkImage depthImage = VK_NULL_HANDLE;
	VmaAllocation depthImageAllocation = VK_NULL_HANDLE;
	VkImageView depthImageView = VK_NULL_HANDLE;

    std::vector<FrameSyncData> frameSyncData;
    std::vector<FrameCommandPools> frameCommandPools;

	std::vector<PerFrameCameraBuffer> cameraBuffers;
	float lastFrame = 0.0f;
	float deltaTime = 0.0f;
	CameraMovement movement;

	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	VkDescriptorSetLayout cameraSetLayout = VK_NULL_HANDLE;

    VkPipelineLayout meshPipelineLayout = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE;

	VkPipeline aabbVisualizingPipeline = VK_NULL_HANDLE;
	bool enableAabbVisualization = false;
	bool freezeCameraFrustum = false;

    fastgltf::Asset asset {};
    std::vector<std::shared_ptr<FileLoadTask>> fileLoadTasks;

	// The mesh data required for rendering the meshlets
	std::vector<FrameDrawCommandBuffers> drawBuffers;
	VkDescriptorSetLayout meshletSetLayout = VK_NULL_HANDLE;
	std::vector<Mesh> meshes;
	MeshBuffers globalMeshBuffers;

	static constexpr std::size_t numDefaultTextures = 1;
	static constexpr std::size_t numDefaultImages = 1;
	static constexpr std::size_t numDefaultMaterials = 1;
	static constexpr std::size_t numDefaultSamplers = 1;

	// Image/material data
	VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
	VkDescriptorSet materialSet = VK_NULL_HANDLE;
	std::vector<VkSampler> samplers;
	std::vector<SampledImage> images;
	std::size_t materialCount = 0;
	VkBuffer materialBuffer = VK_NULL_HANDLE;
	VmaAllocation materialAllocation = VK_NULL_HANDLE;

	// ImGUI / UI objects
	imgui::Renderer imgui;

	// glTF data and options
	std::size_t sceneIndex = 0;
	std::size_t materialVariant = 0;
	fastgltf::Optional<std::size_t> cameraIndex = std::nullopt;
	std::vector<fastgltf::Node*> cameraNodes;

	// Shadow maps
	static constexpr auto shadowResolution = glm::u32vec2(4096.f, 4096.f);
	VkImage shadowMapImage = VK_NULL_HANDLE;
	VmaAllocation shadowMapAllocation = VK_NULL_HANDLE;
	VkImageView shadowMapImageView = VK_NULL_HANDLE;
	VkSampler shadowMapSampler = VK_NULL_HANDLE;
	VkPipelineLayout shadowMapPipelineLayout = VK_NULL_HANDLE;
    VkPipeline shadowMapPipeline = VK_NULL_HANDLE;
	glm::vec3 lightPosition = glm::vec3(-5.0f, 17.5f, 3.75f);

    DeletionQueue deletionQueue;
	TimelineDeletionQueue timelineDeletionQueue;

    Viewer() = default;
    ~Viewer() = default;

    void flushObjects() {
        vkDeviceWaitIdle(device);
		timelineDeletionQueue.flush();
        deletionQueue.flush();
    }

	void loadGltf(const std::filesystem::path& file);

	/** This function creates a DEVICE_LOCAL storage buffer suitable for a transfer destination */
	VkResult createGpuTransferBuffer(std::size_t byteSize, VkBuffer* buffer, VmaAllocation* allocation) const noexcept;
	/** This function creates a VkBuffer on host memory suitable as a staging buffer (transfer source) */
	VkResult createHostStagingBuffer(std::size_t byteSize, VkBuffer* buffer, VmaAllocation* allocation) const noexcept;
	/** Uploads the data to a new storage buffer */
	void uploadBufferToDevice(std::span<const std::byte> bytes, VkBuffer* buffer, VmaAllocation* allocation);
	/** Uploads image data to a image */
	void uploadImageToDevice(std::size_t stagingBufferSize, const std::function<void(VkCommandBuffer, VkBuffer, VmaAllocation)>& commands);

	void uploadMeshlets(GlobalMeshData& globalMeshData);
	/** Takes glTF meshes and uploads them to the GPU */
	void loadGltfMeshes();

	/** Asynchronously loads all gltf images into GPU memory */
	void loadGltfImages();
	void createDefaultImages();
	void loadGltfMaterials();

    void setupVulkanInstance();
    void setupVulkanDevice();

	/** Rebuilds the swapchain after a resize, including other screen targets such as the depth texture */
    void rebuildSwapchain(std::uint32_t width, std::uint32_t height);

	void createDescriptorPool();
	void buildCameraDescriptor();

    /** Builds the Vulkan pipeline used for rendering meshes */
    void buildMeshPipeline();

    void createFrameData();

	/** Functions dedicated to updating GPU buffers at the start of every frame*/
	void updateCameraBuffer(std::size_t currentFrame);
	void updateDrawBuffer(std::size_t currentFrame);

	void drawNode(std::vector<glsl::PrimitiveDraw>& cmd, std::vector<VkDrawIndirectCommand>& aabbCmd, std::size_t nodeIndex, glm::mat4 matrix);
	void drawMesh(std::vector<glsl::PrimitiveDraw>& cmd, std::vector<VkDrawIndirectCommand>& aabbCmd, std::size_t meshIndex, glm::mat4 matrix);

	/** Fills the cameraNodes vector */
	void updateCameraNodes(std::size_t nodeIndex);
	auto getCameraProjectionMatrix(fastgltf::Camera& camera) const -> glm::mat4;

	/** Create UI using ImGui */
	void renderUi();

	/** Creates the shadow map and the necessary pipeline */
	void createShadowMapAndPipeline();
};
