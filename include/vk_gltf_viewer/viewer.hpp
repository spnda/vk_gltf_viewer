#include <vulkan/vk.hpp>
#include <VkBootstrap.h>

#include <TaskScheduler.h>

#include <glfw/glfw3.h>

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

struct Viewer {
    enki::TaskScheduler taskScheduler;

    vkb::Instance instance;
    vkb::Device device;

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

    std::vector<FrameSyncData> frameSyncData;
    std::vector<FrameCommandPools> frameCommandPools;

    VkPipelineLayout meshPipelineLayout = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE;

    fastgltf::Asset asset {};
    std::vector<std::shared_ptr<FileLoadTask>> fileLoadTasks;

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

    void setupVulkanInstance();
    void setupVulkanDevice();
    void rebuildSwapchain(std::uint32_t width, std::uint32_t height);

    /** Builds the Vulkan pipeline used for rendering meshes */
    void buildMeshPipeline();

    void createFrameData();
};
