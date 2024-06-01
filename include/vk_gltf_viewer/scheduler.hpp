#pragma once

#include <exception>
#include <memory>

#include <TaskScheduler.h>

inline enki::TaskScheduler taskScheduler;

enum class PinnedThreadId : std::uint32_t {
	FileIO,
	Count,
};

void initializeScheduler();
std::uint32_t getPinnedThreadNum(PinnedThreadId id);

/** Simple wrapper around another TaskSet that catches any exceptions in the ExecuteRange function. */
struct ExceptionTaskSet : public enki::ITaskSet {
	std::exception_ptr exception;

	void ExecuteRange(enki::TaskSetPartition range, std::uint32_t threadnum) final {
		try {
			ExecuteRangeWithExceptions(range, threadnum);
		} catch (...) {
			exception = std::current_exception();
		}
	}

	virtual void ExecuteRangeWithExceptions(enki::TaskSetPartition range, std::uint32_t threadnum) = 0;
};

/**
 * Dependency for an enki task so that it can destroy itself after completion. OnDependenciesComplete
 * gets called right after the parent task completes, which is why we delete the parent task
 * there. This completable should be a member of the parent task, so that it also gets deleted
 * at the same time. To use this, just add this as a member to the task, and call use(this) in the
 * constructor.
 */
class TaskDeleter final : public enki::ICompletable {
	enki::Dependency dependency;

public:
	void use(enki::ITaskSet* task) {
		SetDependency(dependency, task);
	}

	void OnDependenciesComplete(enki::TaskScheduler* scheduler, std::uint32_t threadnum) override {
		enki::ICompletable::OnDependenciesComplete(scheduler, threadnum);
		delete dependency.GetDependencyTask();
	}
};

/**
 * Dependency for an enki task which keeps a shared_ptr of the parent task, which is dropped
 * as soon as the task completes. Should this be the only reference to the object, then it will be
 * destroyed. However, other references can exist which keep it alive.
 * The use() function requires a std::shared_ptr to be passed, at which point the deleter's shared_ptr
 * is also initialised and the dependency is registered.
 */
template <typename T>
class SharedTaskDeleter final : public enki::ICompletable {
	enki::Dependency dependency;
	std::shared_ptr<T> task;

public:
	void use(std::shared_ptr<T> sharedTask) {
		task = std::move(sharedTask);
		SetDependency(dependency, task.get());
	}

	void OnDependenciesComplete(enki::TaskScheduler* scheduler, std::uint32_t threadnum) override {
		enki::ICompletable::OnDependenciesComplete(scheduler, threadnum);
		task.reset(); // Remove our reference to the task.
	}
};
