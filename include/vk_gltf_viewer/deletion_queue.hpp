#pragma once

#include <functional>
#include <utility>
#include <vector>

#include <vulkan/debug_utils.hpp>

struct Device;

/** Deletion queue, as used by vkguide.dev to ensure proper destruction order of global objects */
class DeletionQueue {
	std::vector<std::move_only_function<void()>> deletors;

public:
	explicit DeletionQueue() = default;
	~DeletionQueue();

	void push(std::move_only_function<void()>&& function);
	void flush();
};

/** DeletionQueue that uses a timeline semaphore to destroy GPU objects when they have actually finished */
struct TimelineDeletionQueue {
	std::reference_wrapper<const Device> device;
	VkSemaphore timelineSemaphore = VK_NULL_HANDLE;
	std::uint64_t hostValue = 0;

	struct Entry {
		std::uint64_t timelineValue = 0;
		std::move_only_function<void()> deletion;
	};
	std::vector<Entry> deletors;

public:
	explicit TimelineDeletionQueue(const Device& device);
	~TimelineDeletionQueue();

	[[nodiscard]] VkSemaphore getSemaphoreHandle() const noexcept {
		return timelineSemaphore;
	}

	void push(std::move_only_function<void()>&& function) {
		deletors.emplace_back(hostValue, std::move(function));
	}

	[[nodiscard]] std::uint64_t nextValue() {
		return ++hostValue;
	}

	/** Function to be called at the start of every frame, which deletes objects if they're old enough */
	void check();

	void flush() {
		ZoneScoped;
		for (auto& entry : deletors) {
			entry.deletion();
		}
		deletors.clear();
	}
};
