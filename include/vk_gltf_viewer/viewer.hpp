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

struct MeshPushConstants {
	glm::mat4 modelMatrix;
};

struct Vertex {
	glm::vec4 position;
};

struct Primitive {
	// TODO: Switch these to VkDeviceSize/uint64_t
	std::uint32_t descOffset;
	std::uint32_t vertexIndicesOffset;
	std::uint32_t triangleIndicesOffset;
	std::uint32_t verticesOffset;

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

	VkDescriptorSet descriptor = VK_NULL_HANDLE;
};

struct Viewer {
    enki::TaskScheduler taskScheduler;

    vkb::Instance instance;
    vkb::Device device;
	VmaAllocator allocator = VK_NULL_HANDLE;

    //using queue_type = std::pair<std::uint32_t, VkQueue>;
    //queue_type graphicsQueue;
    //queue_type transferQueue;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;

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
	VkDescriptorSetLayout meshletSetLayout = VK_NULL_HANDLE;
	std::vector<Mesh> meshes;
	MeshBuffers globalMeshBuffers;

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
            for (auto it = deletors.rbegin(); it != deletors.rend(); ++it) {
                (*it)();
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
	void uploadMeshlets(std::vector<meshopt_Meshlet>& meshlets,
						std::vector<unsigned int>& meshletVertices, std::vector<unsigned char>& meshletTriangles,
						std::vector<Vertex>& vertices);
	/** Takes glTF meshes and uploads them to the GPU */
	void loadGltfMeshes();

    void setupVulkanInstance();
    void setupVulkanDevice();

	/** Rebuilds the swapchain after a resize, including other screen targets such as the depth texture */
    void rebuildSwapchain(std::uint32_t width, std::uint32_t height);

	void createDescriptorPool();
	void buildCameraDescriptor();

    /** Builds the Vulkan pipeline used for rendering meshes */
    void buildMeshPipeline();

    void createFrameData();

	void drawNode(VkCommandBuffer cmd, std::size_t nodeIndex, glm::mat4 matrix);
	void drawMesh(VkCommandBuffer cmd, std::size_t meshIndex, glm::mat4 matrix);
};
