#include <filesystem>

#include <fmt/format.h>
#include <fmt/std.h> // For std::filesystem::path

#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/viewer.hpp>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__) || defined(unix)
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER
int wmain(int argc, wchar_t* argv[]) {
	if (argc < 2) {
		fmt::print("No glTF file specified\n");
		return -1;
	}
	std::vector<std::filesystem::path> gltfs(argc - 1);
	for (std::size_t i = 0; i < gltfs.size(); ++i) {
		gltfs[i] = argv[i + 1];
	}
#else
int main(int argc, char* argv[]) {
	if (argc < 2) {
		fmt::print("No glTF file specified\n");
		return -1;
	}
	std::vector<std::filesystem::path> gltfs(argc - 1);
	for (std::size_t i = 0; i < gltfs.size(); ++i) {
		gltfs[i] = argv[i + 1];
	}
#endif

	for (auto& path : gltfs) {
		if (!std::filesystem::is_regular_file(path)) {
			fmt::print(stderr, "Failed to find glTF file at given path: {}\n", path);
			return -1;
		}
	}

#if defined(_WIN32) && defined(NDEBUG)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#elif defined(__APPLE__) || defined(unix)
	setpriority(PRIO_PROCESS, getpid(), PRIO_MAX);
#endif

	taskScheduler.Initialize();

	Viewer viewer;

	try {
		for (auto& path : gltfs) {
			viewer.loadGltf(path);
		}
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
