set(VKV_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(EXISTS "${VKV_DEPS_DIR}/glfw")
    message(STATUS "vk_gltf_viewer: Found glfw")
    # copied from glfw's CMakeLists. We seriously don't want any examples/tests/docs.
    option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
    option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
    option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
    option(GLFW_INSTALL "Generate installation target" OFF)

    add_subdirectory("${VKV_DEPS_DIR}/glfw")

    set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1)
    set_target_properties(update_mappings PROPERTIES EXCLUDE_FROM_ALL 1)

    add_library(glfw::glfw ALIAS glfw)
endif()

if(EXISTS "${VKV_DEPS_DIR}/glm")
    message(STATUS "vk_gltf_viewer: Found glm")
    add_subdirectory("${VKV_DEPS_DIR}/glm")

    if (TARGET glm_static)
        set_target_properties(glm_static PROPERTIES EXCLUDE_FROM_ALL 1)
        add_library(glm::static ALIAS glm_static)
    endif()
    if (TARGET glm_shared)
        set_target_properties(glm_shared PROPERTIES EXCLUDE_FROM_ALL 1)
        add_library(glm::shared ALIAS glm_shared)
    endif()
endif()

if(EXISTS "${VKV_DEPS_DIR}/stb")
    message(STATUS "vk_gltf_viewer: Found stb")
    add_library(stb INTERFACE)
    target_include_directories(stb INTERFACE "${VKV_DEPS_DIR}/stb")
endif()

if (EXISTS "${VKV_DEPS_DIR}/fmt")
    message(STATUS "vk_gltf_viewer: Found fmtlib")
    add_subdirectory("${VKV_DEPS_DIR}/fmt")
endif()

if (EXISTS "${VKV_DEPS_DIR}/vulkan-headers")
    message(STATUS "vk_gltf_viewer: Found Vulkan-Headers")
    add_library(Vulkan-Headers INTERFACE)
    add_library(Vulkan::Headers ALIAS Vulkan-Headers)
    target_include_directories(Vulkan-Headers SYSTEM INTERFACE "${VKV_DEPS_DIR}/vulkan-headers/include")
endif()

if (EXISTS "${VKV_DEPS_DIR}/volk")
    message(STATUS "vk_gltf_viewer: Found volk")
    # volk automatically calls find_package(Vulkan) if VOLK_PULL_IN_VULKAN is ON. VOLK_HEADERS_ONLY makes it only add volk::volk_headers and
    # doesn't try compiling volk::volk statically, which would fail because VOLK_PULL_IN_VULKAN is set to OFF.
    option(VOLK_PULL_IN_VULKAN "" OFF)
    option(VOLK_HEADERS_ONLY "" ON)
    add_subdirectory(${VKV_DEPS_DIR}/volk)
endif()

if (EXISTS "${VKV_DEPS_DIR}/vk-bootstrap")
    message(STATUS "vk_gltf_viewer: Found vkb")
    add_subdirectory(${VKV_DEPS_DIR}/vk-bootstrap)
endif()

if (EXISTS "${VKV_DEPS_DIR}/vulkanmemoryallocator")
    message(STATUS "vk_gltf_viewer: Found VMA")
    add_library(VulkanMemoryAllocator INTERFACE)
    add_library(Vulkan::MemoryAllocator ALIAS VulkanMemoryAllocator)
    target_include_directories(VulkanMemoryAllocator SYSTEM INTERFACE "${VKV_DEPS_DIR}/vulkanmemoryallocator/include")
endif()

if (EXISTS "${VKV_DEPS_DIR}/enkits")
    message(STATUS "vk_gltf_viewer: Found enkiTS")
    option(ENKITS_BUILD_C_INTERFACE "" OFF)
    option(ENKITS_BUILD_EXAMPLES "" OFF)
    add_subdirectory("${VKV_DEPS_DIR}/enkiTS")
    add_library(enkiTS::enkiTS ALIAS enkiTS)
endif()

if (EXISTS "${VKV_DEPS_DIR}/fastgltf")
    message(STATUS "vk_gltf_viewer: Found fastgltf")
    add_subdirectory("${VKV_DEPS_DIR}/fastgltf")
endif()

if (EXISTS "${VKV_DEPS_DIR}/tracy")
    message(STATUS "vk_gltf_viewer: Found tracy")
    add_library(tracy_client "${VKV_DEPS_DIR}/tracy/public/TracyClient.cpp")
    add_library(Tracy::Client ALIAS tracy_client)

    find_package(Threads REQUIRED)
    target_compile_features(tracy_client PUBLIC cxx_std_11)
    target_include_directories(tracy_client SYSTEM PUBLIC "${VKV_DEPS_DIR}/tracy/public")
    target_link_libraries(tracy_client PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()

if (EXISTS "${VKV_DEPS_DIR}/meshoptimizer")
    add_subdirectory("${VKV_DEPS_DIR}/meshoptimizer")
endif()
