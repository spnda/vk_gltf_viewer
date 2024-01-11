cmake_minimum_required(VERSION 3.27)
project(vk_gltf_viewer)

include(cmake/add_source_directory.cmake)
add_subdirectory(submodules)

add_executable(vk_gltf_viewer EXCLUDE_FROM_ALL)
target_compile_features(vk_gltf_viewer PUBLIC cxx_std_20)
target_link_libraries(vk_gltf_viewer PRIVATE fastgltf glfw::glfw glm::glm stb)
target_link_libraries(vk_gltf_viewer PRIVATE Vulkan::Headers volk::volk_headers vk-bootstrap::vk-bootstrap Vulkan::MemoryAllocator enkiTS::enkiTS)

add_source_directory(TARGET vk_gltf_viewer FOLDER "src")
