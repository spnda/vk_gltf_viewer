#version 460

layout (location = 0) in Inputs {
    vec3 color;
} inp;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = vec4(inp.color, 1.0f);
}
