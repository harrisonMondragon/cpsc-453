#version 450

layout (location = 0) in vec3 inPos;

layout (push_constant) uniform PushConstants {
	mat4 model;
	vec4 color;
} pushConstants;

layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 view;
} ubo;

out gl_PerVertex 
{
	vec4 gl_Position;   
};

void main() 
{
	gl_Position = ubo.projection * ubo.view * pushConstants.model * vec4(inPos, 1.0);
}