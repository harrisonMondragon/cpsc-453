#version 450

layout(location = 0) out vec4 FragColor;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 fragPos;

// layout(location = X) in vec3 viewDir;
// layout(location = X) in vec3 lightDir;

void main() {

	// Constants
	vec3 lightPos = vec3(1,1,1);	// Light position top right
	vec3 lightColor = vec3(1,1,1);	// Light color whie
	vec3 objectColor = vec3(1,0,0);	// Object color green

	// Ambient
	float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

	// Diffuse
	vec3 norm = normalize(normal);
	vec3 lightDir = normalize(lightPos - fragPos);

	float diff = max(dot(normal, lightDir), 0.0);
	vec3 diffuse = diff * lightColor;

	// All together
	vec3 result = (ambient + diffuse) * objectColor;
	FragColor = vec4(result, 1.0);
}