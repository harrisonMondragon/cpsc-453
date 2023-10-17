#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 fragPos;

// layout(location = X) out vec3 viewDir;
// layout(location = X) out vec3 lightDir;

//push constants block
layout( push_constant ) uniform constants
{
	mat4 model;
	mat4 view;
    mat4 proj;
} PushConstants;

void main() {
    outNormal = mat3(transpose(inverse(PushConstants.model))) * inNormal;
    fragPos = vec3(PushConstants.model * vec4(position, 1.0));

    gl_Position = PushConstants.proj *
    PushConstants.view *
    PushConstants.model *
    vec4(position.x, position.y, position.z, 1);
}
