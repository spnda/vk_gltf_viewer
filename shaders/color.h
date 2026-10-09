#ifndef SHADERS_COLOR_H
#define SHADERS_COLOR_H

#include "common.h"
#include "relational.h"

SHADER_NAMESPACE_BEGIN

#if defined(SHADER_METAL)
FUNCTION_INLINE f16vec4 srgb_eotf(f16vec4 srgb) {
	bvec3 cutoff = lessThan(srgb.rgb, f16vec3(0.04045f));
	f16vec3 higher = pow((srgb.rgb + f16vec3(0.055f)) / f16vec3(1.055f), f16vec3(2.4f));
	f16vec3 lower = srgb.rgb / f16vec3(12.92f);

	return f16vec4(mix(higher, lower, f16vec3(cutoff)), srgb.a);
}

FUNCTION_INLINE f16vec4 srgb_oetf(f16vec4 linear_srgb) {
	bvec3 cutoff = lessThan(linear_srgb.rgb, f16vec3(0.0031308f));
	f16vec3 higher = f16vec3(1.055f) * pow(linear_srgb.rgb, f16vec3(1.f / 2.4f)) - f16vec3(0.055f);
	f16vec3 lower = linear_srgb.rgb * f16vec3(12.92f);

	return f16vec4(mix(higher, lower, f16vec3(cutoff)), linear_srgb.a);
}
#endif

#if !defined(SHADER_CPP)
// nonlinear -> linear sRGB transfer function
FUNCTION_INLINE fvec3 srgb_eotf(fvec3 srgb) {
	bvec3 cutoff = lessThan(srgb.rgb, fvec3(0.04045f));
	fvec3 higher = pow((srgb.rgb + fvec3(0.055f)) / fvec3(1.055f), fvec3(2.4f));
	fvec3 lower = srgb.rgb / fvec3(12.92f);

	return mix(higher, lower, fvec3(cutoff));
}

// linear -> nonlinear sRGB transfer function
FUNCTION_INLINE fvec3 srgb_oetf(fvec3 linear_srgb) {
	bvec3 cutoff = lessThan(linear_srgb.rgb, fvec3(0.0031308f));
	fvec3 higher = fvec3(1.055f) * pow(linear_srgb.rgb, fvec3(1.f / 2.4f)) - fvec3(0.055f);
	fvec3 lower = linear_srgb.rgb * fvec3(12.92f);

	return mix(higher, lower, fvec3(cutoff));
}

// nonlinear -> linear sRGB transfer function
FUNCTION_INLINE fvec4 srgb_eotf(fvec4 srgb) {
	return fvec4(srgb_eotf(srgb.rgb), srgb.a);
}

// linear -> nonlinear sRGB transfer function
FUNCTION_INLINE fvec4 srgb_oetf(fvec4 linear_srgb) {
	return fvec4(srgb_oetf(linear_srgb.rgb), linear_srgb.a);
}
#endif

// linear -> nonlinear BT.709 transfer function
FUNCTION_INLINE fvec3 bt709_oetf(fvec3 x) {
	bvec3 cutoff = lessThanEqual(x, fvec3(0.018f));
	fvec3 higher = pow(x, fvec3(0.45)) * 1.099f - 0.099f;
	fvec3 lower = x * 4.5f;
	return mix(higher, lower, fvec3(cutoff));
}

// Inverse BT.1886 EOTF. CoreGraphics decodes BT.709 and BT.2020 with pure gamma 2.4.
FUNCTION_INLINE fvec3 bt1886_inverse_eotf(fvec3 x) {
	return pow(max(x, fvec3(0.f)), fvec3(1.f / 2.4f));
}

// linear -> nonlinear PQ transfer function
FUNCTION_INLINE float pq_oetf(float x) {
	const float m1 = 0.1593017578125;
	const float m2 = 78.84375;
	const float c1 = 0.8359375;
	const float c2 = 18.8515625;
	const float c3 = 18.6875;
	float ym = pow(x, m1);
	return pow((c1 + c2 * ym) / (1.0f + c3 * ym), m2);
}

// linear -> nonlinear PQ transfer function
FUNCTION_INLINE fvec3 pq_oetf(fvec3 c) {
	return fvec3(
		pq_oetf(c.r),
		pq_oetf(c.g),
		pq_oetf(c.b));
}

// linear -> nonlinear HLG transfer function
FUNCTION_INLINE float hlg_oetf(float e) {
	const float a = 0.17883277f;
	const float b = 1.f - 4.f * a;
	const float c = 0.5f - a * log(4.f * a);
	if (e >= 0 && e <= 1.f / 12.f)
		return sqrt(3.f * e);
	return a * log(12 * e - b) + c;
}

// linear -> nonlinear PQ transfer function
FUNCTION_INLINE fvec3 hlg_oetf(fvec3 c) {
	return fvec3(
		hlg_oetf(c.r),
		hlg_oetf(c.g),
		hlg_oetf(c.b));
}

#if !defined(SHADER_GLSL)
enum class color_space_e {
	cie_1931,
	srgb_linear,
	srgb_nonlinear,
	bt709_linear,
	bt709_nonlinear,
	bt2020_linear,
	bt2020_nonlinear,
	bt2100_pq,
	bt2100_hlg,
	display_p3_linear,
	display_p3_nonlinear,
	display_p3_pq,
	display_p3_hlg,
};

#if defined(SHADER_CPP)
FUNCTION_CONSTEXPR std::string_view get_color_space_name(const color_space_e color_space) {
	switch (color_space) {
		using enum color_space_e;
		case cie_1931:
			return "CIE XYZ 1931";
		case srgb_linear:
		case srgb_nonlinear:
			return "sRGB";
		case bt709_linear:
			return "BT.709";
		case bt709_nonlinear:
			return "BT.709 (sRGB)";
		case bt2020_linear:
			return "BT.2020";
		case bt2020_nonlinear:
			return "BT.2020 (sRGB)";
		case bt2100_pq:
			return "BT.2100 (PQ)";
		case bt2100_hlg:
			return "BT.2100 (HLG)";
		case display_p3_linear:
			return "Display P3";
		case display_p3_nonlinear:
			return "Display P3 (sRGB)";
		case display_p3_pq:
			return "Display P3 (PQ)";
		case display_p3_hlg:
			return "Display P3 (HLG)";
		default:
			std::unreachable();
	}
}

FUNCTION_CONSTEXPR bool is_hdr_color_space(const color_space_e color_space) {
	switch (color_space) {
		using enum color_space_e;
		case bt2100_pq:
		case bt2100_hlg:
		case display_p3_pq:
		case display_p3_hlg:
			return true;
		default:
			return false;
	}
}
#endif
#else
#define COLOR_SPACE_SRGB_LINEAR 0
#define COLOR_SPACE_SCRGB_LINEAR 1
#define COLOR_SPACE_BT2020_LINEAR 2
#endif

// https://www.russellcottrell.com/photo/matrixCalculator.htm
FUNCTION_INLINE fmat3 color_make_BT709_to_XYZ_matrix() {
	return transpose(fmat3(
		0.4123908, 0.3575843, 0.1804808,
		0.2126390, 0.7151687, 0.0721923,
		0.0193308, 0.1191948, 0.9505322));
}

FUNCTION_INLINE fmat3 color_make_XYZ_to_BT709_matrix() {
	return transpose(fmat3(
		3.2409699, -1.5373832, -0.4986108,
		-0.9692436, 1.8759675, 0.0415551,
		0.0556301, -0.2039770, 1.0569715));
}

FUNCTION_INLINE fmat3 color_make_DisplayP3_to_XYZ_matrix() {
	return transpose(fmat3(
		0.4865709,  0.2656677,  0.1982173,
		0.2289746,  0.6917385,  0.0792869,
		0.0000000,  0.0451134,  1.0439444));
}

FUNCTION_INLINE fmat3 color_make_XYZ_to_DisplayP3_matrix() {
	return transpose(fmat3(
		2.4934969, -0.9313836, -0.4027108,
		-0.8294890,  1.7626641,  0.0236247,
		0.0358458, -0.0761724,  0.9568845));
}

FUNCTION_INLINE fmat3 color_make_BT2020_to_XYZ_matrix() {
	return transpose(fmat3(
		0.6369580, 0.1446169, 0.1688810,
		0.2627002, 0.6779981, 0.0593017,
		0.0000000, 0.0280727, 1.0609851));
}

FUNCTION_INLINE fmat3 color_make_XYZ_to_BT2020_matrix() {
	return transpose(fmat3(
		1.7166512, -0.3556708, -0.2533663,
		-0.6666844, 1.6164812, 0.0157685,
		0.0176399, -0.0427706, 0.9421031));
}

FUNCTION_INLINE fmat3 color_make_sRGB_to_BT2020_matrix() {
	return color_make_XYZ_to_BT2020_matrix() * color_make_BT709_to_XYZ_matrix();
}

FUNCTION_INLINE fmat3 color_make_BT2020_to_sRGB_matrix() {
	return color_make_XYZ_to_BT709_matrix() * color_make_BT2020_to_XYZ_matrix();
}

FUNCTION_INLINE fvec3 color_convert_sRGB_to_BT2020(const fvec3 srgb_linear) {
	return color_make_sRGB_to_BT2020_matrix() * srgb_linear;
}

FUNCTION_INLINE fvec3 color_convert_BT2020_to_sRGB(const fvec3 bt2020_linear) {
	return color_make_BT2020_to_sRGB_matrix() * bt2020_linear;
}

// CoreGraphics maps 203 nits (BT.2408 reference white) to EDR 1.f
SHADER_CONSTANT float hdr_reference_white_nits = 203.f;
// HLG signal 1.f decodes to this nominal peak
SHADER_CONSTANT float hlg_nominal_peak_nits = 1000.f;

#if defined(SHADER_METAL)
FUNCTION_INLINE float3 convert_linear_color_space(float3 color, shaders::color_space_e input, shaders::color_space_e output) {
	if (input == output)
		return color;

	float3 xyz_color;
	switch (input) {
		using enum shaders::color_space_e;
		case srgb_linear:
		case bt709_linear:
			xyz_color = shaders::color_make_BT709_to_XYZ_matrix() * color;
			break;
		case bt2020_linear:
			xyz_color = shaders::color_make_BT2020_to_XYZ_matrix() * color;
			break;
		case display_p3_linear:
			xyz_color = shaders::color_make_DisplayP3_to_XYZ_matrix() * color;
			break;
		case cie_1931:
		default:
			xyz_color = color;
			break;
	}

	switch (output) {
		using enum shaders::color_space_e;
		case srgb_linear:
		case bt709_linear:
			return shaders::color_make_XYZ_to_BT709_matrix() * xyz_color;
		case bt2020_linear:
			return shaders::color_make_XYZ_to_BT2020_matrix() * xyz_color;
		case display_p3_linear:
			return shaders::color_make_XYZ_to_DisplayP3_matrix() * xyz_color;
		case cie_1931:
		default:
			return xyz_color;
	}
}

FUNCTION_INLINE float3 convert_to_output_color_space(float3 color, shaders::color_space_e input, shaders::color_space_e output) {
	if (input == output)
		return color;

	switch (output) {
		using enum shaders::color_space_e;
		case srgb_linear:
			return convert_linear_color_space(color, input, shaders::color_space_e::srgb_linear);
		case srgb_nonlinear: {
			auto linear_color = convert_linear_color_space(color, input, shaders::color_space_e::srgb_linear);
			return shaders::srgb_oetf(linear_color);
		}
		case bt709_linear:
			return convert_linear_color_space(color, input, shaders::color_space_e::bt709_linear);
		case bt709_nonlinear: {
			auto linear_color = convert_linear_color_space(color, input, shaders::color_space_e::bt709_linear);
			return shaders::bt1886_inverse_eotf(linear_color); // Important: This is only because of CoreGraphics
		}
		case bt2020_linear:
			return convert_linear_color_space(color, input, shaders::color_space_e::bt2020_linear);
		case bt2020_nonlinear: {
			auto linear_color = convert_linear_color_space(color, input, shaders::color_space_e::bt2020_linear);
			return shaders::bt1886_inverse_eotf(linear_color); // Important: This is only because of CoreGraphics
		}
		case bt2100_pq: {
			auto linear_color = convert_linear_color_space(color, input, shaders::color_space_e::bt2020_linear);
			return shaders::pq_oetf(max(linear_color, 0.f) * hdr_reference_white_nits / 10000.f);
		}
		case bt2100_hlg: {
			auto linear_color = convert_linear_color_space(color, input, shaders::color_space_e::bt2020_linear);
			// Inverse OOTF as in BT.2408
			const auto scene = pow(
				saturate(linear_color * (shaders::hdr_reference_white_nits / shaders::hlg_nominal_peak_nits)),
				fvec3(1.f / 1.2f));
			return shaders::hlg_oetf(scene);
		}
		case display_p3_linear:
			return convert_linear_color_space(color, input, shaders::color_space_e::display_p3_linear);
		case display_p3_nonlinear: {
			auto linear_color = convert_linear_color_space(color, input, shaders::color_space_e::display_p3_linear);
			return shaders::srgb_oetf(linear_color);
		}
		case display_p3_pq: {
			auto linear_color = convert_linear_color_space(color, input, shaders::color_space_e::display_p3_linear);
			return shaders::pq_oetf(max(linear_color, 0.f) * hdr_reference_white_nits / 10000.f);
		}
		case display_p3_hlg: {
			auto linear_color = convert_linear_color_space(color, input, shaders::color_space_e::display_p3_linear);
			// Inverse OOTF as in BT.2408
			const auto scene = pow(
				saturate(linear_color * (shaders::hdr_reference_white_nits / shaders::hlg_nominal_peak_nits)),
				fvec3(1.f / 1.2f));
			return shaders::hlg_oetf(scene);
		}
	}
}
#endif

#if defined(SHADER_METAL)
constant float3 kRec709Luma(.2126f,.7152f,.0722f);
FUNCTION_INLINE fvec3 reinhard_operator(fvec3 srcColor, float luminanceScale) {
	float luminance = dot(srcColor, kRec709Luma) + 0.001f;
	float targetLuminance = 1.f / (1.f + luminance);
	return srcColor * targetLuminance * luminanceScale;
}
#endif

struct gt_mapper_constants {
	float contrast;
	float startOfLinearSection;
	float lengthOfLinearSection;
	float toeCurviness;
	float toeFloor;
};

FUNCTION_INLINE float H_f(float x, float e0, float e1) {
	if (x <= e0)
		return 0;
	if (x >= e1)
		return 1;
	return (x - e0) / (e1 - e0);
}

// Gran-Turismo tonemapper suitable for HDR
// http://cdn2.gran-turismo.com/data/www/pdi_publications/PracticalHDRandWCGinGTS.pdf
// https://www.desmos.com/calculator/mbkwnuihbd
/* Defaults:
 *   peak = 1 (SDR; EDR headroom for HDR outputs)
 *   contrast = 1
 *   startOfLinearSection = 0.22
 *   lengthOfLinearSection = 0.4
 *   toeCurviness = 1.33 ("black tightness")
 *   toeFloor = 0 (darkest black)
*/
FUNCTION_INLINE float gt_mapper(float x, gt_mapper_constants gt, float peak) {
	float P = peak;
	float a = gt.contrast;
	float m = gt.startOfLinearSection;
	float l = gt.lengthOfLinearSection;
	float c = gt.toeCurviness;
	float b = gt.toeFloor;

	// Linear section
	float l0 = (P - m)*l / a;
	//float L0 = m - m / a;
	//float L1 = m + (1 - m) / a;
	float L_x = m + a * (x - m);

	// Toe
	float T_x = m * pow(x / m, c) + b;

	// Shoulder
	float S0 = m + l0;
	float S1 = m + a * l0;
	float C2 = a * P / (P - S1);
	float S_x = P - (P - S1) * exp(-(C2*(x-S0)/P));

	float w0_x = 1 - smoothstep(0.f, m, x);
	float w2_x = H_f(x, m + l0, m + l0);
	float w1_x = 1 - w0_x - w2_x;
	float f_x = T_x * w0_x + L_x * w1_x + S_x * w2_x;
	return f_x;
}

FUNCTION_INLINE fvec3 gt_mapper(fvec3 c, gt_mapper_constants gt, float peak) {
	return fvec3(
		gt_mapper(c.r, gt, peak),
		gt_mapper(c.g, gt, peak),
		gt_mapper(c.b, gt, peak)
	);
}

SHADER_NAMESPACE_END
#endif
