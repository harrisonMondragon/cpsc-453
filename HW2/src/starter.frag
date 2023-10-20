/*
 * This code heavily references the following tutorial:
 * https://learnopengl.com/Lighting/Basic-Lighting
 */

#version 450

layout(location = 0) out vec4 FragColor;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 fragPos;
layout(location = 2) in vec3 cameraPos;

void main() {

	// Constants
	vec3 lightPos = vec3(50,50,20);	// Light position
	vec3 lightColor = vec3(1,1,1);	// Light color white
	vec3 objectColor = vec3(34.f/255.f,139.f/255.f,34/255.f);	// Object color

	// Ambient
	float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

	// Diffuse
	vec3 norm = normalize(normal);
	vec3 lightDir = normalize(lightPos - fragPos);

	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diff * lightColor;

	// Specular
	float specularStrength = 0.5;
	vec3 viewDir = normalize(cameraPos - fragPos);
	vec3 reflectDir = reflect(-lightDir, norm);

	float gamma = 64;
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), gamma);
	vec3 specular = specularStrength * spec * lightColor;

	// All together
	vec3 result = (ambient + diffuse + specular) * objectColor;
	FragColor = vec4(result, 1.0);
}