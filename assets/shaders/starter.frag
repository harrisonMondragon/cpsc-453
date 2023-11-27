#version 450

#define PI 3.1415926535897932384626433832795
#define MAX_TEXTURES 4		// save some space in the push constants by hard-wiring this

layout(location = 0) out vec4 color;

// interpolated position and direction of ray in world space
layout(location = 0) in vec3 p;
layout(location = 1) in vec3 d;

//push constants block
layout( push_constant ) uniform constants
{
	mat4 invView; // camera-to-world
	vec4 proj; // (near, far, aspect, fov)
	float time;

} pc;

layout(binding = 0) uniform sampler2D textures[ MAX_TEXTURES ];

// Material properties
vec3 bg_color = vec3(0.00,0.00,0.05);

float closest = -1.0;

void ray_trace(int texture_index, float radius, vec3 center, float rot){

    vec3 dir = normalize(d);

    vec3 pnot = p - center;
    float prod = 2.0 * dot(pnot,dir);

    float normp = length(pnot);
    float discriminant = prod*prod -4.0*(-radius*radius + normp*normp);

    if( discriminant >= 0.0) {
        // determine intersection point
        float t1 = 0.5 * (-prod - sqrt(discriminant));
        float t2 = 0.5 * (-prod + sqrt(discriminant));
        float tmin, tmax;
        float t;
        if(t1 < t2) {
            tmin = t1;
            tmax = t2;
        } else {
            tmin = t2;
            tmax = t1;
        }
        if(tmax > 0.0 && (closest < 0.0 || tmin < closest)) {

            closest = tmin;

            t = (tmin > 0) ? tmin : tmax;
            vec3 ipoint = pnot + t*(dir);

            mat3 intrinsic = mat3(
                cos(rot),-sin(rot),0,
                sin(rot),cos(rot),0,
                0,0,1
            );

            ipoint = intrinsic * ipoint;

            vec3 normal = normalize(ipoint);

            // determine texture coordinates in spherical coordinates

            // First rotate about x through 90 degrees so that y is up.
            normal.z = -normal.z;
            normal = normal.xzy;

            float phi = acos(normal.z);
            float theta;
            if(abs(normal.x) < 0.001) {
                theta = sign(normal.y)*PI*0.5;
            } else {
                theta = atan(normal.y, normal.x);
            }
            // normalize coordinates for texture sampling.
            // Top-left of texture is (0,0) in Vulkan, so we can stick to spherical coordinates
            color = texture(textures[texture_index], vec2(1.0+0.5*theta/PI, phi/PI ));
        }
    }
}

void main() {

    color = vec4(bg_color, 1.0);

    //starry background
    ray_trace(0, 100, vec3(0,0,0), 0);

    //sun
    ray_trace(1, 1.5, vec3(0,0,0), pc.time/100);

    //earth
    //ray_trace(2, 0.75, vec3(7,0,0), 0);
    float earthx = 7*cos(pc.time/50);
    float earthy = 7*sin(pc.time/50);
    ray_trace(2, 0.75, vec3(earthx,earthy,0), pc.time/5);

    //moon
    ray_trace(3, 0.25, vec3(2*cos(pc.time/20)+earthx,2*sin(pc.time/20)+earthy,0), 0);

}
