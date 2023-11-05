#version 450

layout (push_constant) uniform PushConstants {
	mat4 model;
	vec4 color;
} pushConstants;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	outFragColor = pushConstants.color;
}