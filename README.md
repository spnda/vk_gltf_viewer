# vk_gltf_viewer

Originally meant as an example application for [fastgltf](https://github.com/spnda/fastgltf), this has evolved into an advanced Vulkan glTF renderer.
It still shares a lot of the codebase with the OpenGL example for fastgltf, but tries to connect advanced rendering using Vulkan with fastgltf.

### Supported glTF features

- `KHR_mesh_quantization`
- `KHR_materials_variants`
- `KHR_texture_basisu`
- `KHR_texture_transform`
- `EXT_meshopt_compression`
- `MSFT_texture_dds`

### Vulkan requirements

This application currently requires Vulkan 1.3 and `VK_EXT_mesh_shader` (with mesh & task shaders), as well as these core features:
- `multiDrawIndirect`
- `shaderDrawParameters`
- `storageBuffer8BitAccess`
- `shaderInt8`
- `shaderSampledImageArrayNonUniformIndexing`
- `runtimeDescriptorArray`
- `scalarBlockLayout`
- `hostQueryReset`
- `bufferDeviceAddress`
- `synchronization2`
- `dynamicRendering`
- `maintenance4`

### Dependencies

*vk_gltf_viewer* depends on quite a few libraries (sorted in alphabetical order):
- [dds_image](https://github.com/spnda/dds_image)
- [enkiTS](https://github.com/dougbinks/enkiTS)
- [fastgltf](https://github.com/spnda/fastgltf)
- [fmt](https://github.com/fmtlib/fmt)
- [glfw](https://github.com/glfw/glfw)
- [glm](https://github.com/g-truc/glm)
- [imgui](https://github.com/ocornut/imgui)
- [KTX-software](https://github.com/KhronosGroup/KTX-Software)
- [meshoptimizer](https://github.com/zeux/meshoptimizer)
- [tracy](https://github.com/wolfpld/tracy)
- [vk-bootstrap](https://github.com/charls-lunarg/vk-bootstrap)
- [volk](https://github.com/zeux/volk)
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers)
- [Vulkan-Utility-Libraries](https://github.com/KhronosGroup/Vulkan-Utility-Libraries)
- [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- [wuffs](https://github.com/google/wuffs)
