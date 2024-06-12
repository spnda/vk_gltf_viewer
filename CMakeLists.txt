cmake_minimum_required(VERSION 3.27)
project(vk_gltf_viewer)

option(ENABLE_NV_AFTERMATH "Enables NV Aftermath functionality" OFF)

set(SHADER_INCLUDE_DIRS "") # So that the submodule script can update the variable

include(cmake/add_source_directory.cmake)
add_subdirectory(submodules)

add_executable(vk_gltf_viewer EXCLUDE_FROM_ALL)
target_compile_features(vk_gltf_viewer PUBLIC cxx_std_23)
target_link_libraries(vk_gltf_viewer PRIVATE fastgltf wuffs dds_image ktx_read::ktx_read meshoptimizer)
target_link_libraries(vk_gltf_viewer PRIVATE glfw::glfw imgui::imgui enkiTS::enkiTS Tracy::Client fmt::fmt glm::glm)
target_link_libraries(vk_gltf_viewer PRIVATE Vulkan::Headers Vulkan::Utils volk::volk_headers vk-bootstrap::vk-bootstrap Vulkan::MemoryAllocator)
if (ENABLE_NV_AFTERMATH AND TARGET nv::aftermath)
    message(STATUS "vk_gltf_viewer: Enabling use of NV Aftermath")
    target_link_libraries(vk_gltf_viewer PRIVATE nv::aftermath)
    target_compile_definitions(vk_gltf_viewer PRIVATE "VKV_NV_AFTERMATH=1")
    add_source_directory(TARGET vk_gltf_viewer FOLDER "src/nvidia")
endif()

add_source_directory(TARGET vk_gltf_viewer FOLDER "src")
add_source_directory(TARGET vk_gltf_viewer FOLDER "src/vk_gltf_viewer")
add_source_directory(TARGET vk_gltf_viewer FOLDER "src/vulkan")

add_source_directory(TARGET vk_gltf_viewer FOLDER "include/vk_gltf_viewer")
add_source_directory(TARGET vk_gltf_viewer FOLDER "include/vulkan")
target_include_directories(vk_gltf_viewer PRIVATE "include")

# Set compile flags. We target AVX2 as a minimum
fastgltf_compiler_flags(vk_gltf_viewer)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(vk_gltf_viewer PRIVATE "-mavx2")
    elseif(MSVC)
        target_compile_options(vk_gltf_viewer PRIVATE "/arch:AVX")
    endif()
endif()

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

list(APPEND SHADER_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders")

# Get a list of all GLSL shaders in the shaders directory and generate CMake targets for them
# This piece of code is largely copied/refactored from my shader_processor project.
# https://github.com/spnda/shader_processor/blob/a4a2fe2a60549b245503c4c9f6c2a4dfff19eb44/CMakeLists.txt#L9-L71
file(GLOB_RECURSE SHADER_FILES "shaders/*.glsl" "shaders/**/*.glsl")
set(SHADER_C_NAMES "") # List holding all compiled shader names
foreach(SHADER_FILE ${SHADER_FILES})
    message(STATUS "vk_gltf_viewer: Found shader: ${SHADER_FILE}")
    string(SHA1 PATH_HASH ${SHADER_FILE})
    set(SHADER_TIMESTAMP_NAME "shaders/${PATH_HASH}.timestamp")
    cmake_path(GET SHADER_FILE PARENT_PATH SHADER_FOLDER)

    # Match glslang's behaviour by first searching the shader's directory, and then
    # searching the root shaders directory and other include directories.
    set(LOCAL_INCLUDE_DIRS "${SHADER_FOLDER};${SHADER_INCLUDE_DIRS}")

    # Recursively parses file includes
    function(parse_includes RETURN_LIST CONTENTS)
        foreach(LINE IN LISTS ${CONTENTS})
            if (LINE MATCHES "^#include ")
                string(LENGTH ${LINE} LINE_LENGTH)
                math(EXPR LINE_LENGTH "${LINE_LENGTH}-10-1") # Remove #include and the last "
                string(SUBSTRING ${LINE} 10 ${LINE_LENGTH} INCLUDED_FILE)
                string(SUBSTRING ${LINE} 9 1 INCLUDE_CHAR)

                # We'll ignore files which are included with <> instead of "".
                # We consider these to be C++ header includes wrapped in defined(__cplusplus) guards.
                string(SUBSTRING ${LINE} 9 1 INCLUDE_CHAR)
                if (${INCLUDE_CHAR} STREQUAL "<")
                    continue()
                endif()

                # Search for the shader in the include directories (note that these are ordered).
                foreach(INCL_DIR IN LISTS LOCAL_INCLUDE_DIRS)
                    set(INCLUDED_FILE_ABS ${INCL_DIR}/${INCLUDED_FILE})
                    if(EXISTS ${INCLUDED_FILE_ABS})
                        list(APPEND ${RETURN_LIST} ${INCLUDED_FILE_ABS})

                        # Load the included file, and parse it recursively
                        file(STRINGS ${INCLUDED_FILE_ABS} INCLUDED_FILE_CONTENTS)
                        parse_includes(${RETURN_LIST} INCLUDED_FILE_CONTENTS)
                        set(SHADER_FOUND TRUE)
                    endif()
                endforeach()

                if (NOT ${SHADER_FOUND})
                    message(STATUS "Failed to find included file: ${INCLUDED_FILE}")
                endif()
                set(SHADER_FOUND "")
            endif()
        endforeach()

        return(PROPAGATE ${RETURN_LIST})
    endfunction()

    # Read the file and parse includes recursively
    file(STRINGS ${SHADER_FILE} FILE_CONTENTS)
    set(FILE_INCLUDES "")
    parse_includes(FILE_INCLUDES FILE_CONTENTS)
    list(REMOVE_DUPLICATES FILE_INCLUDES)

    # Add a custom command to compile the shader
    cmake_path(GET SHADER_FILE FILENAME SHADER_FILENAME)
    set(COMPILED_SHADER_FILE "shaders/${PATH_HASH}.spirv")
    add_custom_command(
        OUTPUT ${SHADER_TIMESTAMP_NAME} ${COMPILED_SHADER_FILE}
        COMMAND ${GLSLANG_EXECUTABLE} $<$<CONFIG:DEBUG>:-g> --target-env vulkan1.3 "-I$<JOIN:${LOCAL_INCLUDE_DIRS},;-I>" -o ${COMPILED_SHADER_FILE} ${SHADER_FILE}
        COMMAND ${CMAKE_COMMAND} -E copy ${COMPILED_SHADER_FILE} shaders/${SHADER_FILENAME}.spirv
        COMMAND ${CMAKE_COMMAND} -E touch ${SHADER_TIMESTAMP_NAME}
        DEPENDS ${SHADER_FILE} ${FILE_INCLUDES}
        COMMAND_EXPAND_LISTS
        COMMENT "Processing ${SHADER_FILE}"
    )

    # Add a custom command that copies the generated SPIR-V into a .cpp file
    set(SPIRV_CPP_FILE "shaders/${PATH_HASH}.cpp")
    string(MAKE_C_IDENTIFIER ${SHADER_FILENAME} SHADER_FILENAME_C)
    add_custom_command(
        OUTPUT ${SPIRV_CPP_FILE}
        COMMAND ${CMAKE_COMMAND} -DPROCESSED_FILE_NAME=${SHADER_FILENAME_C} -DINPUT_FILE=${COMPILED_SHADER_FILE} -DOUTPUT_FILE=${SPIRV_CPP_FILE} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/configure_spirv.cmake
        DEPENDS ${SHADER_TIMESTAMP_NAME} ${COMPILED_SHADER_FILE}
        COMMENT "Packaging ${SHADER_FILE}"
    )

    list(APPEND SHADER_C_NAMES ${SHADER_FILENAME_C})

    # Finally, add the generated source file as a target source
    target_sources(vk_gltf_viewer PRIVATE ${SPIRV_CPP_FILE})
endforeach()

# Create a single header acting as a shader manifest which holds references to all shader data
# This effectively creates a list of "extern std::span" for each shader, which are all defined in
# their respective cpp files, as generated above.
set(SHADER_DATA_SPANS "")
foreach(SHADER_NAME IN LISTS SHADER_C_NAMES)
    list(APPEND SHADER_DATA_SPANS "\nextern const std::span<const std::uint32_t> ${SHADER_NAME}")
endforeach()

configure_file("${CMAKE_CURRENT_LIST_DIR}/cmake/spirv_manifest.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/include/spirv_manifest.hpp")
target_include_directories(vk_gltf_viewer PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

# Copy all required runtime DLLs into the build directory
add_custom_command(TARGET vk_gltf_viewer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:vk_gltf_viewer> $<TARGET_RUNTIME_DLLS:vk_gltf_viewer>
  COMMAND_EXPAND_LISTS
)
