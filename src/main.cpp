#include <filesystem>

#include <fmt/format.h>

#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/viewer.hpp>

#ifdef _MSC_VER
int wmain(int argc, wchar_t* argv[]) {
	if (argc < 2) {
		fmt::print("No glTF file specified\n");
		return -1;
	}
	std::filesystem::path gltfFile { argv[1] };
#else
int main(int argc, char* argv[]) {
	if (argc < 2) {
		fmt::print("No glTF file specified\n");
		return -1;
	}
	std::filesystem::path gltfFile { argv[1] };
#endif

	if (!std::filesystem::is_regular_file(gltfFile)) {
		return -1;
	}

	taskScheduler.Initialize();

	Viewer viewer;

	try {
		viewer.loadGltf(gltfFile);
		viewer.run();
	} catch (const vulkan_error& error) {
		fmt::print("{}: {}\n", error.what(), error.what_result());
	} catch (const std::runtime_error& error) {
		fmt::print("{}\n", error.what());
	}

	viewer.destroy();

	taskScheduler.WaitforAllAndShutdown();

	return 0;
}
