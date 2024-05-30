#include <array>

#include <tracy/Tracy.hpp>

#include <vk_gltf_viewer/scheduler.hpp>

struct PinnedTaskRunLoop : enki::IPinnedTask {
	void Execute() override {
		while (!taskScheduler.GetIsShutdownRequested()) {
			// This thread will sleep until a new pinned task is available for the thread,
			// and then run it.
			taskScheduler.WaitForNewPinnedTasks();
			taskScheduler.RunPinnedTasks();
		}
	}
};

std::array<PinnedTaskRunLoop, std::to_underlying(PinnedThreadId::Count)> pinnedTaskRunners;

void initializeScheduler() {
	ZoneScoped;
	auto pinnedThreadCount = std::to_underlying(PinnedThreadId::Count);

	enki::TaskSchedulerConfig config;
	config.numTaskThreadsToCreate += pinnedThreadCount;

	taskScheduler.Initialize(config);

	// Start the pinned tasks
	std::uint32_t normalThreads = taskScheduler.GetNumTaskThreads() - pinnedThreadCount;
	for (std::uint32_t i = 0; i < pinnedThreadCount; ++i) {
		pinnedTaskRunners[i].threadNum = i + normalThreads;
		taskScheduler.AddPinnedTask(&pinnedTaskRunners[i]);
	}
}

std::uint32_t getPinnedThreadNum(PinnedThreadId id) {
	assert(!taskScheduler.GetIsShutdownRequested());
	return (taskScheduler.GetNumTaskThreads() - std::to_underlying(PinnedThreadId::Count)) + std::to_underlying(id);
}
