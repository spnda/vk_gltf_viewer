#ifndef SHADERS_RELATIONAL_H
#define SHADERS_RELATIONAL_H

#if defined(SHADER_METAL)
// The Metal stdlib only includes relational vector comparisons through operators,
// which is great but we can't use that in our shared shader headers. Therefore,
// this implements most of the GLSL relational functions

#include <metal_relational>
namespace metal {
template <typename T, size_t N>
vec<bool, N> greaterThan(vec<T, N> a, vec<T, N> b) {
	return a > b;
}
template <typename T, size_t N>
vec<bool, N> greaterThanEqual(vec<T, N> a, vec<T, N> b) {
	return a >= b;
}
template <typename T, size_t N>
vec<bool, N> lessThan(vec<T, N> a, vec<T, N> b) {
	return a < b;
}
template <typename T, size_t N>
vec<bool, N> lessThanEqual(vec<T, N> a, vec<T, N> b) {
	return a <= b;
}

template <typename T, size_t N>
vec<bool, N> equal(vec<T, N> a, vec<T, N> b) {
	return a == b;
}
template <typename T, size_t N>
vec<bool, N> notEqual(vec<T, N> a, vec<T, N> b) {
	return a != b;
}
} // namespace metal

#endif

#endif
