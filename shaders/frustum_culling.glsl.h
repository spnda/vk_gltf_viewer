// Frustum culling using 6 planes on an AABB
bool isAabbInFrustum(in vec3 center, in vec3 extents, in uint view) {
    [[unroll]] for (uint i = 0; i < 6; ++i) {
        const vec4 plane = camera.views[view].frustum[i];

        const float radius = dot(extents, abs(plane.xyz));
        const float distance = dot(plane.xyz, center) - plane.w;
        if (-radius > distance) {
            return false;
        }
    }
    return true;
}

// See https://gist.github.com/cmf028/81e8d3907035640ee0e3fdd69ada543f#file-aabb_transform-comp-L109-L132
vec3 getWorldSpaceAabbExtent(in vec3 extent, in mat4 transform) {
    const mat3 transformExtents = mat3(
        abs(vec3(transform[0])),
        abs(vec3(transform[1])),
        abs(vec3(transform[2]))
    );
    return transformExtents * extent;
}
