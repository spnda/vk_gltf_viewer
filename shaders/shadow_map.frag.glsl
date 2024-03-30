#version 460

// Empty shader which effectively just passes on the depth for the shadow map.
void main() {
    gl_FragDepth = gl_FragCoord.z;
}
