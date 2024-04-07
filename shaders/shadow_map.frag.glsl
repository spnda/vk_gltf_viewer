#version 460

// Empty shader which effectively just passes on the depth for the shadow map.
void main() {
    // TODO: Do we want to discard alpha fragments?
    gl_FragDepth = gl_FragCoord.z;
}
