/*
 * This code heavily references the following tutorial:
 * https://learnopengl.com/Lighting/Basic-Lighting
 */

#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 fragPos;
layout(location = 2) out vec3 cameraPos;

//push constants block
layout( push_constant ) uniform constants
{
	mat4 model;
	mat4 view;
    mat4 proj;
    vec3 cameraPos;
} PushConstants;

void main() {
    outNormal = normalize(mat3(PushConstants.model) * inNormal);

    fragPos = vec3(PushConstants.model * vec4(position, 1.0));
    cameraPos = PushConstants.cameraPos;

    gl_Position = PushConstants.proj *
    PushConstants.view *
    PushConstants.model *
    vec4(position.x, position.y, position.z, 1);
}
