#pragma once

#include <concepts>
#include <type_traits>

template <typename T>
requires requires (T t) {
	{ t > t } -> std::same_as<bool>;
}
[[nodiscard]] constexpr T max(T a, T b) noexcept {
	return (a > b) ? a : b;
}

template <typename T>
requires requires (T t) {
	{ t < t } -> std::same_as<bool>;
}
[[nodiscard]] constexpr T min(T a, T b) noexcept {
	return (a < b) ? a : b;
}

template <typename T>
[[nodiscard]] constexpr T alignDown(T base, T alignment) {
	return base - (base % alignment);
}

#include <TaskScheduler.h>

class TaskDeleter final : public enki::ICompletable {
	enki::Dependency dependency;

public:
	void use(enki::ICompletable* task) {
		SetDependency(dependency, task);
	}

	void OnDependenciesComplete(enki::TaskScheduler* scheduler, std::uint32_t threadnum) override {
		enki::ICompletable::OnDependenciesComplete(scheduler, threadnum);
		delete dependency.GetDependencyTask();
	}
};
