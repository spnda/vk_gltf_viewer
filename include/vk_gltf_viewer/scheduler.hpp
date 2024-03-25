#pragma once

#include <TaskScheduler.h>

// See main.cpp for the declaration
extern enki::TaskScheduler taskScheduler;

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
