#version 450

// global constants
vec3 colour1 = vec3(0.0, 0.25, 0.25);		// for the checkerboard
vec3 colour2 = vec3(0.75, 0.75, 0.75);

float ambient_strength = 0.1;
vec3 specular_strength = vec3(0.5, 0.5, 0.5);	// allow some diffuse through
float specular_power = 76.8;

vec3 lightDir = vec3(1.0, 2.0, 1.0);	// view space vector, will be normalized later!
vec3 lightCol = vec3(1.0, 1.0, 1.0);	// overall light colour


// data layout
layout(location = 0) out vec4 colour;

layout(location = 0) in vec3 normal;	// view space vector
layout(location = 1) in vec3 viewDir;	// view space vector
layout(location = 2) in vec2 tex;	// texture space

layout(binding = 0) uniform MyUBO {
    int grid[16][16];
} myUBO;

// Procedural texture to generate a checkerboard
bool cboard( vec2 t ) {

    int NUM_CHECKS = 64;
    int x = int(t.x * NUM_CHECKS);
    int y = int(t.y * NUM_CHECKS);
    return ((x + y) % 2 == 0);
}

float Noise(vec2 interp){
    float u = interp.x;
    float v = interp.y;

    float fj0k0 = myUBO.grid[int(floor(u))][int(floor(v))];
    float fj0k1 = myUBO.grid[int(floor(u))][int(ceil(v))];
    float fj1k0 = myUBO.grid[int(ceil(u))][int(floor(v))];
    float fj1k1 = myUBO.grid[int(ceil(u))][int(ceil(v))];

    float left_edge = (1-v)*fj0k0 + (v)*fj0k1;
    float right_edge = (1-v)*fj1k0 + (v)*fj1k1;

    return (1-u)*left_edge + (u)*right_edge;
}

float T(vec2 tex_coords){
    float turb = 0;

    for(int i = 0; i <= 4; i++){
        turb += (Noise(vec2(pow(2,i)*tex_coords.x, pow(2,i)*tex_coords.y)))/(pow(2,i));
    }

    return turb;
}

float S(vec2 tex_coords){
    float u = tex_coords.x;
    float v = tex_coords.y;

    float m = 24;

    float sin_input = m * radians(180) * (u+v+(T(tex_coords)));

    return 0.5 * (1+sin(sin_input));
}

void main() {

    // Normalize N, L and V vectors
    vec3 N = normalize(normal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(viewDir);

    float perlin = S(tex);

    // Calculate R locally
    vec3 R = reflect(-L, N);

    // Compute the diffuse and specular components for each fragment
    //vec3 basecol = (cboard(tex)) ? colour1 : colour2;

    vec3 basecol = vec3(0, 1, 0);
    vec3 diffuse = max(dot(N, L), 0.0) * basecol * perlin;

    vec3 ambient = ambient_strength * basecol;

    vec3 specular = pow(max(dot(R, V), 0.0), specular_power) * specular_strength;

    // Write final colour to the framebuffer
    colour = vec4((ambient + diffuse + specular)*lightCol, 1.0);

}
