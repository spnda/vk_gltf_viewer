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
