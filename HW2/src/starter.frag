#version 450

layout(location = 0) out vec4 color;

layout(location = 0) in vec3 normal;  

void main() {
	color = vec4(normal, 1);
}