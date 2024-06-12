#include <ranges>

#include <tracy/Tracy.hpp>

#include <vk_gltf_viewer/deletion_queue.hpp>
#include <vk_gltf_viewer/device.hpp>

void DeletionQueue::push(std::move_only_function<void()>&& function) {
	deletors.emplace_back(std::move(function));
}

DeletionQueue::~DeletionQueue() {
	flush();
}

void DeletionQueue::flush() {
	ZoneScoped;
	for (auto& func : deletors | std::views::reverse) {
		func();
	}
	deletors.clear();
}

TimelineDeletionQueue::TimelineDeletionQueue(const Device& _device) : device(_device) {
	ZoneScoped;
	constexpr VkSemaphoreTypeCreateInfo timelineCreateInfo {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
		.initialValue = 0,
	};

	const VkSemaphoreCreateInfo createInfo {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = &timelineCreateInfo,
		.flags = 0,
	};

	vk::checkResult(vkCreateSemaphore(device.get(), &createInfo, vk::allocationCallbacks.get(), &timelineSemaphore),
					"Failed to create timeline semaphore for deletion queue: {}");
	vk::setDebugUtilsName(device.get(), timelineSemaphore, "Deletion queue timeline semaphore");
}

TimelineDeletionQueue::~TimelineDeletionQueue() {
	vkDestroySemaphore(device.get(), timelineSemaphore, vk::allocationCallbacks.get());
}

void TimelineDeletionQueue::check() {
	ZoneScoped;
	std::uint64_t currentValue;
	auto result = vkGetSemaphoreCounterValue(device.get(), timelineSemaphore, &currentValue);
	vk::checkResult(result, "Failed to get timeline semaphore counter value: {}");
	for (auto it = deletors.begin(); it != deletors.end();) {
		auto& [timelineValue, deletion] = *it;
		if (timelineValue < currentValue) {
			deletion();
			it = deletors.erase(it);
		} else {
			++it;
		}
	}
}
