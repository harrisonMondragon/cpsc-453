#version 450

// global constants
vec3 colour1 = vec3(0.0, 0.25, 0.25);
vec3 colour2 = vec3(0.75, 0.75, 0.75);
float ambient_strength = 0.1;

vec3 specular_albedo = vec3(0.5, 0.5, 0.5);
float specular_power = 76.8;

vec3 inLight = normalize(vec3(1,1,1));


// data layout
layout(location = 0) out vec4 colour;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 view;
layout(location = 2) in vec2 tex;


// Procedural texture to generate a checkerboard
bool cboard( vec2 t ) {

    int NUM_CHECKS = 64;
    int x = int(t.x * NUM_CHECKS);
    int y = int(t.y * NUM_CHECKS);
    return ((x + y) % 2 == 0);
}

void main() {

    // Normalize N, L and V vectors
    vec3 N = normalize(normal);
    vec3 L = normalize(inLight);
    vec3 V = normalize(view);

    // Calculate R locally
    vec3 R = reflect(-L, N);

    // Compute the diffuse and specular components for each fragment
    vec3 basecol = (cboard(tex)) ? colour1 : colour2;
    vec3 diffuse = max(dot(N, L), 0.0) * basecol;
    vec3 ambient = ambient_strength * basecol;
    
    vec3 specular = pow(max(dot(R, V), 0.0), specular_power) * specular_albedo;

    // Write final colour to the framebuffer
    colour = vec4(ambient + diffuse + specular, 1.0);

}
