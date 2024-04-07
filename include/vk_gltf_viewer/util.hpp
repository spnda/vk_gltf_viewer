#pragma once

#include <concepts>
#include <type_traits>

namespace util {
	template<typename T>
	requires requires(T t) {
		{ t > t } -> std::same_as<bool>;
	}
	[[nodiscard]] constexpr T max(T a, T b) noexcept {
		return (a > b) ? a : b;
	}

	template<typename T>
	requires requires(T t) {
		{ t < t } -> std::same_as<bool>;
	}
	[[nodiscard]] constexpr T min(T a, T b) noexcept {
		return (a < b) ? a : b;
	}

	template<typename T>
	[[nodiscard]] constexpr T alignDown(T base, T alignment) {
		return base - (base % alignment);
	}
} // namespace util
