#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 inNormal;

layout(location=0) out vec3 outNormal;

//push constants block
layout( push_constant ) uniform constants
{
	mat4 model;
	mat4 view;
    mat4 proj;
} PushConstants;

void main() {
    outNormal = inNormal;

    gl_Position = PushConstants.proj *
    PushConstants.view *
    PushConstants.model *
    vec4(position.x, position.y, position.z, 1);
}
