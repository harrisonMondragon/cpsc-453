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

int grid[16][16] = {
    {20, 45, 30, 5, 12, 36, 85, 74, 63, 11, 95, 42, 50, 7, 26, 19},
    {40, 72, 15, 81, 8, 58, 33, 22, 69, 25, 60, 90, 2, 38, 18, 55},
    {65, 14, 44, 70, 28, 52, 17, 80, 6, 49, 75, 93, 35, 78, 87, 3},
    {68, 99, 41, 23, 59, 21, 89, 9, 27, 84, 46, 10, 47, 16, 82, 32},
    {98, 24, 54, 61, 91, 37, 77, 53, 1, 79, 4, 71, 48, 29, 62, 96},
    {34, 56, 76, 13, 67, 88, 43, 57, 94, 64, 83, 31, 66, 73, 39, 51},
    {100, 47, 78, 20, 35, 63, 8, 72, 57, 41, 96, 14, 53, 91, 29, 83},
    {42, 86, 18, 69, 27, 50, 5, 37, 74, 23, 68, 3, 45, 94, 12, 60},
    {11, 80, 31, 77, 48, 26, 66, 21, 59, 9, 54, 89, 16, 40, 98, 7},
    {67, 15, 44, 93, 22, 62, 38, 84, 71, 32, 56, 1, 76, 36, 97, 19},
    {24, 79, 5, 70, 33, 85, 51, 10, 65, 2, 47, 75, 17, 92, 4, 88},
    {39, 73, 12, 61, 28, 55, 82, 6, 90, 25, 52, 99, 13, 43, 81, 30},
    {58, 97, 34, 64, 19, 49, 100, 11, 78, 46, 80, 7, 70, 28, 94, 42},
    {20, 86, 27, 73, 4, 41, 89, 14, 55, 37, 67, 8, 60, 35, 79, 3},
    {62, 17, 52, 96, 25, 44, 85, 6, 74, 21, 68, 1, 51, 38, 83, 9},
    {43, 77, 15, 94, 32, 69, 100, 13, 80, 47, 76, 5, 90, 26, 72, 18}
};


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

    float fj0k0 = grid[int(floor(u))][int(floor(v))];
    float fj0k1 = grid[int(floor(u))][int(ceil(v))];
    float fj1k0 = grid[int(ceil(u))][int(floor(v))];
    float fj1k1 = grid[int(ceil(u))][int(ceil(v))];

    float left_edge = (1-v)*fj0k0 + (v)*fj0k1;
    float right_edge = (1-v)*fj1k0 + (v)*fj1k1;

    return (1-u)*left_edge + (u)*right_edge;
}

float T(vec2 tex_coords){
    float turb = 0;

    for(int i = 0; i <= 4; i++){
        turb += Noise(vec2(pow(2,i)*tex_coords.x, pow(2,i)*tex_coords.y));
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

    vec3 basecol = vec3(0, 0.5, 0.5);
    vec3 diffuse = max(dot(N, L), 0.0) * basecol * perlin;

    vec3 ambient = ambient_strength * basecol;

    vec3 specular = pow(max(dot(R, V), 0.0), specular_power) * specular_strength;

    // Write final colour to the framebuffer
    colour = vec4((ambient + diffuse + specular)*lightCol, 1.0);

}
