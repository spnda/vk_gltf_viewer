#version 460

layout (location = 0) in Inputs {
    vec4 color;
} inp;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = inp.color;
}
