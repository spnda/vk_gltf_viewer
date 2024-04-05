#if defined(__cplusplus)
namespace glsl {
using uint = std::uint32_t;
using namespace std;
using namespace glm;
#else
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#endif

#if !defined(__cplusplus)
// TODO: Find a way to share the ImDrawVert definition with this shader header?
struct ImDrawVert {
    vec2 pos;
    vec2 uv;
    // The color is always in sRGB currently with ImGui.
    uint col;
};

layout(buffer_reference, scalar) readonly buffer Vertices { ImDrawVert v[]; };
#endif

struct PushConstants {
    vec2 scale;
    vec2 translate;
#if defined(__cplusplus)
	VkDeviceAddress vertexBufferAddress = 0;
#else
    Vertices vertices;
#endif
    uint imageIndex;
};

struct FragmentInput {
    vec4 color;
    vec2 uv;
};

#if defined(__cplusplus)
} // namespace glsl
#endif
