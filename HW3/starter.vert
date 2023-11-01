#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 inNormal; 
layout(location = 2) in vec2 inTex;

// normal, view vector and tex coords. to be passed to fragment shader
layout(location=0) out vec3 N;
layout(location=1) out vec3 V;
layout(location=2) out vec2 T;

//push constants block
layout( push_constant ) uniform constants
{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 orientation;

} pushConstants;

void main() {
    N = mat3(pushConstants.orientation) * inNormal;

    // Calculate view-vector
    vec4 P = pushConstants.view * vec4(position.xyz, 1);
    V = -P.xyz;

    // texture coordinates are passed through
    T = inTex;
	
    gl_Position = pushConstants.proj * pushConstants.view * 
    	pushConstants.model * vec4(position.xyz, 1);	// for depth buffer
}
