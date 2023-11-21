/*
 * This code heavily references the following tutorial:
 * https://vulkan-tutorial.com/Texture_mapping/Images
 *
 * The github for the tutorial can be found at:
 * https://github.com/Overv/VulkanTutorial
 */

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

// All samplers
layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D aoSampler;
layout(binding = 3) uniform sampler2D procSampler;

//push constants block
layout(push_constant) uniform constants
{
	mat4 model;
	mat4 view;
	mat4 proj;
    bool proc;
    bool ao;
} pushConstants;

// Noise function. The texture function linarly interpolates
float Noise(vec2 interp){
    return texture(procSampler, interp).r;
}

// Turbulance function
float T(vec2 tex_coords){
    float turb = 0.0;
    for(int i = 0; i <= 4; i++){
        turb += ((Noise(pow(2,i)*tex_coords))/(pow(2,i)));
    }

    return turb;
}

// S function.  Obtain the required 2D procedural texture within the domain
float S(vec2 tex_coords){
    float u = tex_coords.x;
    float v = tex_coords.y;

    float m = 96.0;
    float sin_input = m * radians(180.0) * (u+v+(T(tex_coords)));

    return (0.5 * (1.0+sin(sin_input)));
}

void main() {

    // Normalize N, L and V vectors
    vec3 N = normalize(normal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(viewDir);

    // Calculate R locally
    vec3 R = reflect(-L, N);

    // Base color
    vec4 basecol = texture(texSampler, tex);

    // Change basecol if using procedural texturing
    if(pushConstants.proc){
        float perlin = S(tex);
        basecol = vec4(vec3(perlin), 1.0);
    }

    // Calculate ambient occlusion
    float ao_value = 1.0;
    if(pushConstants.ao){
        ao_value = texture(aoSampler, tex).r;
    }

    // Compute the ambient, diffuse, and specular components for each fragment
    vec3 ambient = ambient_strength * ao_value * basecol.rgb;
    vec3 diffuse = max(dot(N, L), 0.0) * basecol.rgb;
    vec3 specular = pow(max(dot(R, V), 0.0), specular_power) * specular_strength;

    // Write final colour to the framebuffer
    colour = vec4((ambient + diffuse + specular)*lightCol, basecol.a);
}
