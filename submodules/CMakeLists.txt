set(VKV_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if (EXISTS "${VKV_DEPS_DIR}/tracy")
    message(STATUS "vk_gltf_viewer: Found tracy")
    add_library(tracy_client "${VKV_DEPS_DIR}/tracy/public/TracyClient.cpp")
    add_library(Tracy::Client ALIAS tracy_client)

    find_package(Threads REQUIRED)
    target_compile_features(tracy_client PUBLIC cxx_std_11)
    target_include_directories(tracy_client SYSTEM PUBLIC "${VKV_DEPS_DIR}/tracy/public")
    target_link_libraries(tracy_client PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

    # Tracy only cares if the macro is defined, not its value. Use PUBLIC to propagate the value.
    if (TRACY_ENABLE)
        target_compile_definitions(tracy_client PUBLIC "TRACY_ENABLE=1")
    endif()
endif()

if(EXISTS "${VKV_DEPS_DIR}/glfw")
    message(STATUS "vk_gltf_viewer: Found glfw")
    # copied from glfw's CMakeLists. We seriously don't want any examples/tests/docs.
    option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
    option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
    option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
    option(GLFW_INSTALL "Generate installation target" OFF)

    option(GLFW_BUILD_WAYLAND "" OFF)

    add_subdirectory("${VKV_DEPS_DIR}/glfw")

    # We only want and need Vulkan.
    target_compile_definitions(glfw INTERFACE "GLFW_INCLUDE_NONE=1")

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

if(EXISTS "${VKV_DEPS_DIR}/wuffs")
    message(STATUS "vk_gltf_viewer: Found wuffs")
    add_library(wuffs INTERFACE)
    target_include_directories(wuffs INTERFACE "${VKV_DEPS_DIR}/wuffs/release/c")
endif()

if (EXISTS "${VKV_DEPS_DIR}/dds_image")
    message(STATUS "vk_gltf_viewer: Found dds_image")
    add_subdirectory("${VKV_DEPS_DIR}/dds_image")
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

if (EXISTS "${VKV_DEPS_DIR}/vulkan-utils")
    message(STATUS "vk_gltf_viewer: Found Vulkan-Utils")
    add_library(Vulkan-Utils INTERFACE)
    add_library(Vulkan::Utils ALIAS Vulkan-Utils)
    target_include_directories(Vulkan-Utils SYSTEM INTERFACE "${VKV_DEPS_DIR}/vulkan-utils/include")
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
    option(FASTGLTF_COMPILE_AS_CPP20 "" ON)
    add_subdirectory("${VKV_DEPS_DIR}/fastgltf")
endif()

if (EXISTS "${VKV_DEPS_DIR}/meshoptimizer")
    add_subdirectory("${VKV_DEPS_DIR}/meshoptimizer")
endif()

if (EXISTS "${VKV_DEPS_DIR}/imgui")
    message(STATUS "vk_gltf_viewer: Found imgui")

    set(IMGUI_DIRECTORY "${VKV_DEPS_DIR}/imgui")
    if(NOT EXISTS "${IMGUI_DIRECTORY}")
        message(FATAL_ERROR "The imgui submodule does not exist.")
    endif()

    add_library(imgui)
    add_library(imgui::imgui ALIAS imgui)
    target_compile_features(imgui PUBLIC cxx_std_11)
    add_source_directory(TARGET imgui FOLDER "${IMGUI_DIRECTORY}")
    add_source_directory(TARGET imgui FOLDER "${IMGUI_DIRECTORY}/misc/cpp/")

    # This is that we can keep a custom ImGui config header in our local repository, that has accesses
    # to other headers in our project's include directories.
    target_include_directories(imgui SYSTEM PUBLIC "${PROJECT_SOURCE_DIR}/include")
    target_compile_definitions(imgui PUBLIC IMGUI_USER_CONFIG=\"imgui/config.hpp\")

    target_include_directories(imgui SYSTEM PUBLIC "${IMGUI_DIRECTORY}")
    target_include_directories(imgui SYSTEM PUBLIC "${IMGUI_DIRECTORY}/backends")
    target_include_directories(imgui SYSTEM PUBLIC "${IMGUI_DIRECTORY}/misc/cpp")

    target_compile_definitions(imgui PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS=1)

    # We only need the GLFW implementation
    target_sources(imgui PUBLIC "${IMGUI_DIRECTORY}/backends/imgui_impl_glfw.h" "${IMGUI_DIRECTORY}/backends/imgui_impl_glfw.cpp")

    target_link_libraries(imgui PUBLIC glfw::glfw glm::glm)
endif()

if (EXISTS "${VKV_DEPS_DIR}/ktx-software")
    message(STATUS "vk_gltf_viewer: Found ktx-software")

    # KTX-Software is probably the shittiest CMake library I have yet to encounter,
    # and changes these CMake variables
    set(TEMPORARY_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(TEMPORARY_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(TEMPORARY_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    # For some reason using option() doesn't work here?
    set(KTX_FEATURE_TOOLS OFF CACHE BOOL "" FORCE)
    set(KTX_FEATURE_TESTS OFF CACHE BOOL "" FORCE)
    set(KTX_FEATURE_KTX1 ON CACHE BOOL "" FORCE)
    set(KTX_FEATURE_KTX2 ON CACHE BOOL "" FORCE)
    set(KTX_FEATURE_VK_UPLOAD OFF CACHE BOOL "" FORCE)
    set(KTX_FEATURE_GL_UPLOAD OFF CACHE BOOL "" FORCE)
    set(KTX_FEATURE_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
    set(KTX_FEATURE_LOADTEST_APPS "" CACHE BOOL "" FORCE)
    add_subdirectory("${VKV_DEPS_DIR}/ktx-software")
    add_library(ktx_read::ktx_read ALIAS ktx_read)

    # Restore global variables
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${TEMPORARY_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TEMPORARY_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TEMPORARY_RUNTIME_OUTPUT_DIRECTORY})
endif()

if (EXISTS "${VKV_DEPS_DIR}/nv-aftermath" AND NOT APPLE)
    message(STATUS "vk_gltf_viewer: Found NVIDIA Aftermath")

    add_library(nv_aftermath SHARED IMPORTED GLOBAL)
    add_library(nv::aftermath ALIAS nv_aftermath)
    target_include_directories(nv_aftermath INTERFACE "${VKV_DEPS_DIR}/nv-aftermath/include")
    if(WIN32)
        set_target_properties(nv_aftermath PROPERTIES
            IMPORTED_LOCATION "${VKV_DEPS_DIR}/nv-aftermath/lib/x64/GFSDK_Aftermath_Lib.x64.dll"
            IMPORTED_IMPLIB "${VKV_DEPS_DIR}/nv-aftermath/lib/x64/GFSDK_Aftermath_Lib.x64.lib"
            IMPORTED_CONFIGURATIONS "RELEASE"
        )
    elseif(UNIX AND NOT APPLE)
        set_target_properties(nv_aftermath PROPERTIES
            IMPORTED_LOCATION "${VKV_DEPS_DIR}/nv-aftermath/lib/x64/libGFSDK_Aftermath_Lib.x64.so"
            IMPORTED_CONFIGURATIONS "RELEASE"
        )
    endif()
endif()

if(EXISTS "${VKV_DEPS_DIR}/dlss" AND NOT APPLE)
    message(STATUS "vk_gltf_viewer: Found NVIDIA DLSS")

    add_library(nv_dlss SHARED IMPORTED GLOBAL)
    add_library(nv::dlss ALIAS nv_dlss)
    target_include_directories(nv_dlss INTERFACE "${VKV_DEPS_DIR}/dlss/include")
    if(WIN32)
        set_target_properties(nv_dlss PROPERTIES
            IMPORTED_LOCATION "${VKV_DEPS_DIR}/dlss/lib/Windows_x86_64/rel/nvngx_dlss.dll"
            IMPORTED_LOCATION_DEBUG "${VKV_DEPS_DIR}/dlss/lib/Windows_x86_64/dev/nvngx_dlss.dll"
            IMPORTED_IMPLIB "${VKV_DEPS_DIR}/dlss/lib/Windows_x86_64/x86_64/nvsdk_ngx_d.lib"
            IMPORTED_IMPLIB_DEBUG "${VKV_DEPS_DIR}/dlss/lib/Windows_x86_64/x86_64/nvsdk_ngx_d_dbg.lib"
            IMPORTED_CONFIGURATIONS "RELEASE;DEBUG"
        )
    elseif(UNIX AND NOT APPLE)
        set_target_properties(nv_dlss PROPERTIES
            IMPORTED_LOCATION "${VKV_DEPS_DIR}/dlss/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.3.7.10"
            IMPORTED_LOCATION_DEBUG "${VKV_DEPS_DIR}/dlss/lib/Linux_x86_64/dev/libnvidia-ngx-dlss.so.3.7.10"
            IMPORTED_IMPLIB "${VKV_DEPS_DIR}/dlss/lib/Linux_x86_64/libnvsdk_ngx.a"
            IMPORTED_CONFIGURATIONS "RELEASE;DEBUG"
        )
    endif()
endif()

if(EXISTS "${VKV_DEPS_DIR}/metal-cpp" AND APPLE)
    add_library(metal-cpp INTERFACE)
    add_library(metal::cpp ALIAS metal-cpp)
    target_include_directories(metal-cpp INTERFACE "${VKV_DEPS_DIR}/metal-cpp")
    target_link_libraries(metal-cpp INTERFACE
            "-framework Metal"
            "-framework MetalKit"
            "-framework AppKit"
            "-framework Foundation"
            "-framework QuartzCore"
    )
endif()

return(PROPAGATE SHADER_INCLUDE_DIRS)
