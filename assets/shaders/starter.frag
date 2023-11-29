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

// To tell which planet to render on top
float closest = -1.0;

// Lighting constants
float ambient_strength = 0.1;
vec3 specular_strength = vec3(0.5, 0.5, 0.5);	// allow some diffuse through
float specular_power = 128;
vec3 lightCol = vec3(1.0, 1.0, 1.0);	// overall light colour


void ray_trace(int texture_index, float radius, vec3 center, float rot, bool lighting){

    vec3 dir = normalize(d);

    vec3 pminusc = p - center;
    float prod = 2.0 * dot(pminusc,dir);

    float normp = length(pminusc);
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

            closest = tmin > 0.0 ? tmin : tmax;

            t = (tmin > 0) ? tmin : tmax;
            vec3 ipoint = pminusc + t*(dir);

            // mat3 intrinsic = mat3(
            //     cos(rot),-sin(rot),0,
            //     sin(rot),cos(rot),0,
            //     0,0,1
            // );

            // vec3 normal = normalize(intrinsic * ipoint);

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
            // vec4 basecol = texture(textures[texture_index], vec2(1.0+0.5*theta/PI, phi/PI ));

            vec4 basecol = texture(textures[texture_index], vec2(rot+0.5*theta/PI, phi/PI ));

            // Lighting
            if(lighting){
                vec3 N = normalize(ipoint - center);
                vec3 L = normalize(ipoint);
                vec3 V = normalize(p - ipoint);
                vec3 R = reflect(-L, N);
                
                vec3 ambient = ambient_strength * basecol.rgb;
                vec3 diffuse = max(dot(N, L), 0.0) * basecol.rgb;
                
                vec3 specular = vec3(0,0,0);
                if(dot(N,L) > 0){
                    specular = pow(max(dot(R, V), 0.0), specular_power) * specular_strength;
                }
                
                color = vec4((ambient + diffuse + specular)*lightCol, basecol.a);
            } else {
                color = basecol;
            }
        }
    }
}

void main() {

    float click_per_day = 3.0;

    float moon_axial_period = 27.00 * click_per_day;
    float moon_orbital_period = moon_axial_period;
    float earth_axial_period = moon_axial_period/27.00;
    float earth_orbital_period = moon_orbital_period*12;
    float sun_axial_period = moon_axial_period;

    float ahh = pc.time/5.0;

    color = vec4(bg_color, 1.0);

    //starry background
    ray_trace(0, 100, vec3(0,0,0), 0, false);

    //sun
    // ray_trace(1, 1.5, vec3(0,0,0), pc.time/sun_axial_period, false);
    // ray_trace(1, 1.5, vec3(0,0,0), pc.time/sun_axial_period, false);
    ray_trace(1, 1.5, vec3(0,0,0), ahh/sun_axial_period, false);

    //earth
    float earthx = 10*cos(pc.time/earth_orbital_period);
    float earthy = 10*sin(pc.time/earth_orbital_period);
    // ray_trace(2, 0.75, vec3(earthx,earthy,0), pc.time/earth_axial_period, true);
    // ray_trace(2, 0.75, vec3(earthx,0,earthy), pc.time/click_per_day, true);
    ray_trace(2, 0.75, vec3(earthx,0,earthy), ahh/click_per_day, true);

    //moon
    float moonx = 2*cos(pc.time/moon_orbital_period)+earthx;
    float moony = 2*sin(pc.time/moon_orbital_period)+earthy;
    // ray_trace(3, 0.25, vec3(moonx,moony,0), pc.time/moon_axial_period, true);
    // ray_trace(3, 0.25, vec3(moonx,0,moony), pc.time/moon_axial_period, true);
    ray_trace(3, 0.25, vec3(moonx,0,moony), ahh/moon_axial_period, true);

}
