cmake_minimum_required(VERSION 3.27)
project(vk_gltf_viewer)

include(cmake/add_source_directory.cmake)
add_subdirectory(submodules)

add_executable(vk_gltf_viewer EXCLUDE_FROM_ALL)
target_compile_features(vk_gltf_viewer PUBLIC cxx_std_20)
target_link_libraries(vk_gltf_viewer PRIVATE fastgltf stb dds_image ktx_read::ktx_read meshoptimizer)
target_link_libraries(vk_gltf_viewer PRIVATE glfw::glfw imgui::imgui enkiTS::enkiTS Tracy::Client fmt::fmt glm::glm)
target_link_libraries(vk_gltf_viewer PRIVATE Vulkan::Headers Vulkan::Utils volk::volk_headers vk-bootstrap::vk-bootstrap Vulkan::MemoryAllocator)

add_source_directory(TARGET vk_gltf_viewer FOLDER "src")
add_source_directory(TARGET vk_gltf_viewer FOLDER "src/vulkan")

add_source_directory(TARGET vk_gltf_viewer FOLDER "include/vk_gltf_viewer")
add_source_directory(TARGET vk_gltf_viewer FOLDER "include/vulkan")
target_include_directories(vk_gltf_viewer PRIVATE "include")

# Add the shaders directory as an include directory
target_include_directories(vk_gltf_viewer PRIVATE "shaders")
file (GLOB_RECURSE SHADER_HEADER_FILES "shaders/*.h" "shaders/**/*.h")
foreach(HEADER_FILE ${SHADER_HEADER_FILES})
    target_sources(vk_gltf_viewer PUBLIC ${HEADER_FILE})
endforeach()

# Try and search for glslangValidator, which we use to compile shaders
find_program(GLSLANG_EXECUTABLE glslangValidator)
if (NOT GLSLANG_EXECUTABLE)
    message(FATAL_ERROR "Failed to find glslangValidator executable required for compiling shaders")
endif()

message(STATUS "vk_gltf_viewer: Found glslangValidator: ${GLSLANG_EXECUTABLE}")

# Get a list of all GLSL shaders in the shaders directory and generate CMake targets for them
# This piece of code is largely copied/refactored from my shader_processor project.
# https://github.com/spnda/shader_processor/blob/a4a2fe2a60549b245503c4c9f6c2a4dfff19eb44/CMakeLists.txt#L9-L71
file(GLOB_RECURSE SHADER_FILES "shaders/*.glsl" "shaders/**/*.glsl")
foreach(SHADER_FILE ${SHADER_FILES})
    message(STATUS "vk_gltf_viewer: Found shader: ${SHADER_FILE}")
    string(SHA1 PATH_HASH ${SHADER_FILE})
    set(SHADER_TARGET_NAME "build_shader_${PATH_HASH}")
    set(SHADER_TIMESTAMP_NAME "${PATH_HASH}.timestamp")
    cmake_path(GET SHADER_FILE PARENT_PATH SHADER_FOLDER)

    # Parse includes and add them as a dependency
    # TODO: Make this recursive?
    file(STRINGS ${SHADER_FILE} FILE_CONTENTS)
    set(FILE_INCLUDES "")
    foreach(LINE IN LISTS FILE_CONTENTS)
        # Search for potential includes
        if (LINE MATCHES "^#include ")
            string(LENGTH ${LINE} LINE_LENGTH)
            MATH(EXPR LINE_LENGTH "${LINE_LENGTH}-10-1") # Remove #include and the last "
            string(SUBSTRING ${LINE} 10 ${LINE_LENGTH} INCLUDED_FILE)
            set(INCLUDED_FILE "${SHADER_FOLDER}/${INCLUDED_FILE}")
            list(APPEND FILE_INCLUDES ${INCLUDED_FILE})
        endif()
    endforeach()

    cmake_path(GET SHADER_FILE FILENAME SHADER_FILENAME)
    add_custom_command(
        OUTPUT ${SHADER_TIMESTAMP_NAME}
        COMMAND ${GLSLANG_EXECUTABLE} $<$<CONFIG:DEBUG>:-g> --target-env vulkan1.3 -o ${SHADER_FILENAME}.spv ${SHADER_FILE}
        COMMAND ${CMAKE_COMMAND} -E touch ${SHADER_TIMESTAMP_NAME}
        DEPENDS ${SHADER_FILE} ${FILE_INCLUDES}
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        # VERBATIM
        COMMENT "Processing ${SHADER_FILE}"
    )
    add_custom_target(${SHADER_TARGET_NAME} DEPENDS ${SHADER_TIMESTAMP_NAME})
    add_dependencies(vk_gltf_viewer ${SHADER_TARGET_NAME})
endforeach()
