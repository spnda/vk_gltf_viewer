#ifndef SRGB_GLSL_H
#define SRGB_GLSL_H

#include "common.h.glsl"
GLSL_NAMESPACE_BEGIN

/** sRGB conversion functions shamelessly stolen from https://gamedev.stackexchange.com/a/194038/159451 */

float fromLinear(PARAMETER_COPY(float) linearRGB) {
	bool cutoff = linearRGB < 0.0031308f;
	float higher = 1.055 * pow(linearRGB, 1.f / 2.4f) - 0.055f;
	float lower = linearRGB * 12.92f;

	return mix(higher, lower, cutoff);
}

float toLinear(PARAMETER_COPY(float) sRGB) {
	bool cutoff = sRGB < 0.04045f;
	float higher = pow((sRGB + 0.055f) / 1.055f, 2.4f);
	float lower = sRGB / 12.92;

	return mix(higher, lower, cutoff);
}

// Converts a color from linear light gamma to sRGB gamma
vec4 fromLinear(PARAMETER_COPY(vec4) linearRGB) {
	bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308f));
	vec3 higher = vec3(1.055f) * pow(linearRGB.rgb, vec3(1.f / 2.4f)) - vec3(0.055f);
	vec3 lower = linearRGB.rgb * vec3(12.92f);

	return vec4(mix(higher, lower, cutoff), linearRGB.a);
}

// Converts a color from sRGB gamma to linear light gamma
vec4 toLinear(PARAMETER_COPY(vec4) sRGB) {
	bvec3 cutoff = lessThan(sRGB.rgb, vec3(0.04045f));
	vec3 higher = pow((sRGB.rgb + vec3(0.055f)) / vec3(1.055f), vec3(2.4f));
	vec3 lower = sRGB.rgb / vec3(12.92f);

	return vec4(mix(higher, lower, cutoff), sRGB.a);
}

GLSL_NAMESPACE_END
#endif
