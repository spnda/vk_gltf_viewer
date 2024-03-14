#include <ranges>
#include <vector>

#include <vulkan/vk.hpp>
#include <VkBootstrap.h>

#include <TaskScheduler.h>

#include <glfw/glfw3.h>

#include <glm/mat4x4.hpp>

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

struct Camera {
	glm::mat4 viewProjectionMatrix;
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
};

struct Vertex {
	glm::vec4 position;
	glm::vec4 color;
	glm::vec2 uv;
};

struct Primitive {
	std::uint32_t descOffset;
	std::uint32_t vertexIndicesOffset;
	std::uint32_t triangleIndicesOffset;
	std::uint32_t verticesOffset;

	std::uint32_t materialIndex;
	std::size_t meshlet_count;
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

struct PrimitiveDraw {
	VkDrawMeshTasksIndirectCommandEXT command;

	// The matrix from the glTF is technically per-mesh, but we define it for each primitive.
	// We could optimise this slightly, but there's many models where each mesh has only one primitive.
	glm::mat4x4 modelMatrix;

	// TODO: Switch these to VkDeviceSize/uint64_t
	std::uint32_t descOffset;
	std::uint32_t vertexIndicesOffset;
	std::uint32_t triangleIndicesOffset;
	std::uint32_t verticesOffset;

	std::uint32_t meshletCount;
	std::uint32_t materialIndex;
};

struct FrameDrawCommandBuffers {
	VkBuffer primitiveDrawHandle;
	VmaAllocation primitiveDrawAllocation;
	VkDeviceSize primitiveDrawBufferSize;

	std::uint32_t drawCount;
};

struct Material {
	glm::vec4 albedoFactor;
	std::uint32_t albedoIndex;
	float alphaCutoff;

	glm::vec2 padding;
};

struct SampledImage {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView imageView = VK_NULL_HANDLE;
};

struct Viewer {
    vkb::Instance instance;
    vkb::Device device;
	VmaAllocator allocator = VK_NULL_HANDLE;
	TracyVkCtx tracyCtx = nullptr;

    VkQueue graphicsQueue = VK_NULL_HANDLE;

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

    fastgltf::Asset asset {};
    std::vector<std::shared_ptr<FileLoadTask>> fileLoadTasks;

	// The mesh data required for rendering the meshlets
	std::vector<FrameDrawCommandBuffers> drawBuffers;
	VkDescriptorSetLayout meshletSetLayout = VK_NULL_HANDLE;
	std::vector<Mesh> meshes;
	MeshBuffers globalMeshBuffers;

	// TODO: Differentiate between numDefaultTextures and numDefaultImages?
	static constexpr std::size_t numDefaultTextures = 1;
	static constexpr std::size_t numDefaultMaterials = 1;
	static constexpr std::size_t numDefaultSamplers = 1;

	// Image/material data
	VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
	VkDescriptorSet materialSet = VK_NULL_HANDLE;
	std::vector<VkSampler> samplers;
	std::vector<SampledImage> images;
	VkBuffer materialBuffer = VK_NULL_HANDLE;
	VmaAllocation materialAllocation = VK_NULL_HANDLE;

    // This is the same paradigm as used by vkguide.dev. This makes sure every object
    // is properly destroyed in reverse-order to creation.
    class DeletionQueue {
        friend struct Viewer;
        std::deque<std::function<void()>> deletors;

    public:
        void push(std::function<void()>&& function) {
            deletors.emplace_back(function);
        }

        void flush() {
			for (auto& func : deletors | std::views::reverse) {
				func();
			}
            deletors.clear();
        }
    };
    DeletionQueue deletionQueue;

    Viewer() = default;
    ~Viewer() = default;

    void flushObjects() {
        vkDeviceWaitIdle(device);
        deletionQueue.flush();
    }

	void loadGltf(std::string_view file);

	/** This function uploads a buffer to DEVICE_LOCAL memory on the GPU using a staging buffer. */
	VkResult createGpuTransferBuffer(std::size_t byteSize, VkBuffer* buffer, VmaAllocation* allocation) noexcept;
	void uploadMeshlets(std::vector<meshopt_Meshlet>& meshlets,
						std::vector<unsigned int>& meshletVertices, std::vector<unsigned char>& meshletTriangles,
						std::vector<Vertex>& vertices);
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

	void drawNode(std::vector<PrimitiveDraw>& cmd, std::size_t nodeIndex, glm::mat4 matrix);
	void drawMesh(std::vector<PrimitiveDraw>& cmd, std::size_t meshIndex, glm::mat4 matrix);
};
