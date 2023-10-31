#version 450

layout(location = 0) out vec4 color;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 view;
layout(location = 2) in vec2 tex;

// Procedural texture to generate a checkerboard
int cboard( vec2 t ) {
    int NUM_CHECKS = 64;
    int x = int(t.x * NUM_CHECKS);
    int y = int(t.y * NUM_CHECKS);
    if ((x + y) % 2 == 0) {
        return 0;
    } else {
        return 1;
    }

}

vec3 inLight = normalize(vec3(1,1,1));
// Material properties
vec3 color1 = vec3(0.0, 0.25, 0.25);
vec3 color2 = vec3(0.75, 0.75, 0.75);

vec3 specular_albedo = vec3(0.5, 0.5, 0.5);
float specular_power = 76.8;

void main() {
	// Normalize N, L and V vectors
    vec3 N = normalize(normal);
    vec3 L = normalize(inLight);
    vec3 V = normalize(view);

    // Calculate R locally
    vec3 R = reflect(-L, N);

    // Compute the diffuse and specular components for each fragment
    vec3 diffuse = max(dot(N, L), 0.0) * vec3(1.0, 1.0, 1.0);
    vec3 ambient;
    if( cboard(tex) == 0 ) {
        diffuse *= color1;
        ambient = 0.1*color1;
    } else {
        diffuse *= color2;
        ambient = 0.1*color2;
    }
    
    vec3 specular = pow(max(dot(R, V), 0.0), specular_power) * specular_albedo;

    // Write final color to the framebuffer
    color = vec4(ambient + diffuse + specular, 1.0);
}