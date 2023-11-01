#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 inNormal; 
layout(location = 2) in vec2 inTex;

// normal, view vector and tex coords. to be passed to fragment shader
layout(location=0) out vec3 N;		// view space
layout(location=1) out vec3 V;		// view space
layout(location=2) out vec2 T;		// ???

//push constants block
layout( push_constant ) uniform constants
{
	mat4 model;
	mat4 view;
	mat4 proj;

} pushConstants;

void main() {

    mat4 mv = pushConstants.view * pushConstants.model;

    N = mat3(mv) * inNormal;	// view space, but without translation

    // Calculate view-vector
    vec4 P = mv * vec4(position.xyz, 1);
    V = -P.xyz / P.w;		// view space

    // texture coordinates are passed through
    T = inTex;
	
    // for depth buffer, otherwise ignored (homogenized camera)
    gl_Position = pushConstants.proj * mv * vec4(position.xyz, 1);	
}
