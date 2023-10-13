/*
 * Copyright 2023 University of Calgary, Visualization and Graphics Group
 */

#include "Camera.h"
#include "Object.h"
#include <VulkanLaunchpad.h>
#include <vulkan/vulkan.hpp>
#include <random>
#include <string>

// buffers that will live on the GPU.
// No geometry retained on the CPU, all data sent to the GPU.

uint32_t mNumObjectIndices;
VkBuffer mObjectVertexData;
VkBuffer mObjectIndices;
VklCameraHandle mCameraHandle;

// A pipeline that can be used for HW2
VkPipeline pipeline;

// Struct to pack object vertex data
struct Vertex {
	glm::vec3 position;
    glm::vec3 normal;
};

// Send model, view and projection matrices as push constants
// which are packed in this struct
struct ObjectPushConstants {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

// MVP matrices that are updated interactively
ObjectPushConstants pushConstants;

// Simple interactive rotation of object controlled by an angle
extern float roll;
extern float pitch;
extern float yaw;

// Organize geometry data and send it to the GPU
void objectCreateGeometryAndBuffers(std::string objPath, GLFWwindow* window)
{
	VKL_LOG("Your command line arg was [" << objPath << "]...");
	// TODO: Use this to read in obj files
	// Load geometry from file specified via command line
	// VklGeometryData modelGeometry = vklLoadModelGeometry(objPath);

	// Create a camera object for the window size
	mCameraHandle = vklCreateCamera(window);

	// Teapot vertices
	std::vector<glm::vec3> positions = {
		glm::vec3(-0.0112664,0.188986,-0.392027), glm::vec3(0.187941,0.188986,-0.339176), glm::vec3(0.327909,0.188986,-0.199208), glm::vec3(0.38076,0.188986,-9.72432e-10),
		glm::vec3(-0.0112664,0.213487,-0.387619), glm::vec3(0.185702,0.213487,-0.335362), glm::vec3(0.324096,0.213487,-0.196968), glm::vec3(0.376353,0.213487,-9.72432e-10),
		glm::vec3(-0.0112664,0.213487,-0.401102), glm::vec3(0.192553,0.213487,-0.347027), glm::vec3(0.335761,0.213487,-0.203819), glm::vec3(0.389835,0.213487,-9.72432e-10),
		glm::vec3(-0.0112664,0.188986,-0.420029), glm::vec3(0.20217,0.188986,-0.363403), glm::vec3(0.352136,0.188986,-0.213437), glm::vec3(0.408762,0.188986,-9.72432e-10),
		glm::vec3(-0.403293,0.188986,-9.72432e-10), glm::vec3(-0.350442,0.188986,-0.199208), glm::vec3(-0.210474,0.188986,-0.339176), glm::vec3(-0.0112664,0.188986,-0.392027),
		glm::vec3(-0.398886,0.213487,-9.72432e-10), glm::vec3(-0.346629,0.213487,-0.196968), glm::vec3(-0.208234,0.213487,-0.335362), glm::vec3(-0.0112664,0.213487,-0.387619),
		glm::vec3(-0.412368,0.213487,-9.72432e-10), glm::vec3(-0.358294,0.213487,-0.203819), glm::vec3(-0.215085,0.213487,-0.347027), glm::vec3(-0.0112664,0.213487,-0.401102),
		glm::vec3(-0.431295,0.188986,-9.72432e-10), glm::vec3(-0.374669,0.188986,-0.213437), glm::vec3(-0.224703,0.188986,-0.363403), glm::vec3(-0.0112664,0.188986,-0.420029),
		glm::vec3(0.38076,0.188986,-9.72432e-10), glm::vec3(0.327909,0.188986,0.199208), glm::vec3(0.187941,0.188986,0.339176), glm::vec3(-0.0112664,0.188986,0.392027),
		glm::vec3(0.376353,0.213487,-9.72432e-10), glm::vec3(0.324096,0.213487,0.196968), glm::vec3(0.185702,0.213487,0.335362), glm::vec3(-0.0112664,0.213487,0.387619),
		glm::vec3(0.389835,0.213487,-9.72432e-10), glm::vec3(0.335761,0.213487,0.203819), glm::vec3(0.192553,0.213487,0.347027), glm::vec3(-0.0112664,0.213487,0.401102),
		glm::vec3(0.408762,0.188986,-9.72432e-10), glm::vec3(0.352136,0.188986,0.213437), glm::vec3(0.20217,0.188986,0.363403), glm::vec3(-0.0112664,0.188986,0.420029),
		glm::vec3(-0.0112664,0.188986,0.392027), glm::vec3(-0.210474,0.188986,0.339176), glm::vec3(-0.350442,0.188986,0.199208), glm::vec3(-0.403293,0.188986,-9.72432e-10),
		glm::vec3(-0.0112664,0.213487,0.387619), glm::vec3(-0.208234,0.213487,0.335362), glm::vec3(-0.346629,0.213487,0.196968), glm::vec3(-0.398886,0.213487,-9.72432e-10),
		glm::vec3(-0.0112664,0.213487,0.401102), glm::vec3(-0.215085,0.213487,0.347027), glm::vec3(-0.358294,0.213487,0.203819), glm::vec3(-0.412368,0.213487,-9.72432e-10),
		glm::vec3(-0.0112664,0.188986,0.420029), glm::vec3(-0.224703,0.188986,0.363403), glm::vec3(-0.374669,0.188986,0.213437), glm::vec3(-0.431295,0.188986,-9.72432e-10),
		glm::vec3(-0.0112664,0.188986,-0.420029), glm::vec3(0.20217,0.188986,-0.363403), glm::vec3(0.352136,0.188986,-0.213437), glm::vec3(0.408762,0.188986,-9.72432e-10),
		glm::vec3(-0.0112664,0.0427534,-0.487441), glm::vec3(0.236426,0.0427534,-0.421727), glm::vec3(0.41046,0.0427533,-0.247692), glm::vec3(0.476174,0.0427534,-9.72432e-10),
		glm::vec3(-0.0112664,-0.0988118,-0.539296), glm::vec3(0.262776,-0.0988118,-0.466591), glm::vec3(0.455325,-0.0988118,-0.274042), glm::vec3(0.52803,-0.0988118,-9.72432e-10),
		glm::vec3(-0.0112664,-0.231043,-0.560038), glm::vec3(0.273316,-0.231043,-0.484537), glm::vec3(0.473271,-0.231043,-0.284582), glm::vec3(0.548772,-0.231043,-9.72432e-10),
		glm::vec3(-0.431295,0.188986,-9.72432e-10), glm::vec3(-0.374669,0.188986,-0.213437), glm::vec3(-0.224703,0.188986,-0.363403), glm::vec3(-0.0112664,0.188986,-0.420029),
		glm::vec3(-0.498707,0.0427534,-9.72432e-10), glm::vec3(-0.432993,0.0427534,-0.247692), glm::vec3(-0.258959,0.0427533,-0.421727), glm::vec3(-0.0112664,0.0427534,-0.487441),
		glm::vec3(-0.550563,-0.0988118,-9.72432e-10), glm::vec3(-0.477857,-0.0988118,-0.274042), glm::vec3(-0.285309,-0.0988118,-0.466591), glm::vec3(-0.0112664,-0.0988118,-0.539296),
		glm::vec3(-0.571305,-0.231043,-9.72432e-10), glm::vec3(-0.495803,-0.231043,-0.284582), glm::vec3(-0.295849,-0.231043,-0.484537), glm::vec3(-0.0112664,-0.231043,-0.560038),
		glm::vec3(0.408762,0.188986,-9.72432e-10), glm::vec3(0.352136,0.188986,0.213437), glm::vec3(0.20217,0.188986,0.363403), glm::vec3(-0.0112664,0.188986,0.420029),
		glm::vec3(0.476174,0.0427534,-9.72432e-10), glm::vec3(0.41046,0.0427534,0.247692), glm::vec3(0.236426,0.0427533,0.421727), glm::vec3(-0.0112664,0.0427534,0.487441),
		glm::vec3(0.52803,-0.0988118,-9.72432e-10), glm::vec3(0.455325,-0.0988118,0.274042), glm::vec3(0.262776,-0.0988118,0.466591), glm::vec3(-0.0112664,-0.0988118,0.539296),
		glm::vec3(0.548772,-0.231043,-9.72432e-10), glm::vec3(0.47327,-0.231043,0.284582), glm::vec3(0.273316,-0.231043,0.484537), glm::vec3(-0.0112664,-0.231043,0.560038),
		glm::vec3(-0.0112664,0.188986,0.420029), glm::vec3(-0.224703,0.188986,0.363403), glm::vec3(-0.374669,0.188986,0.213437), glm::vec3(-0.431295,0.188986,-9.72432e-10),
		glm::vec3(-0.0112664,0.0427534,0.487441), glm::vec3(-0.258959,0.0427534,0.421727), glm::vec3(-0.432993,0.0427533,0.247692), glm::vec3(-0.498707,0.0427534,-9.72432e-10),
		glm::vec3(-0.0112664,-0.0988118,0.539296), glm::vec3(-0.285309,-0.0988118,0.466591), glm::vec3(-0.477858,-0.0988118,0.274042), glm::vec3(-0.550563,-0.0988118,-9.72432e-10),
		glm::vec3(-0.0112664,-0.231043,0.560038), glm::vec3(-0.295849,-0.231043,0.484537), glm::vec3(-0.495803,-0.231043,0.284582), glm::vec3(-0.571305,-0.231043,-9.72432e-10),
		glm::vec3(-0.0112664,-0.231043,-0.560038), glm::vec3(0.273316,-0.231043,-0.484537), glm::vec3(0.473271,-0.231043,-0.284582), glm::vec3(0.548772,-0.231043,-9.72432e-10),
		glm::vec3(-0.0112664,-0.336828,-0.52374), glm::vec3(0.254871,-0.336828,-0.453132), glm::vec3(0.441865,-0.336828,-0.266137), glm::vec3(0.512473,-0.336828,-9.72432e-10),
		glm::vec3(-0.0112664,-0.405277,-0.456328), glm::vec3(0.220616,-0.405277,-0.394808), glm::vec3(0.383541,-0.405277,-0.231882), glm::vec3(0.445061,-0.405277,-9.72432e-10),
		glm::vec3(-0.0112664,-0.441058,-0.420029), glm::vec3(0.20217,-0.441058,-0.363403), glm::vec3(0.352136,-0.441058,-0.213437), glm::vec3(0.408762,-0.441058,-9.72432e-10),
		glm::vec3(-0.571305,-0.231043,-9.72432e-10), glm::vec3(-0.495803,-0.231043,-0.284582), glm::vec3(-0.295849,-0.231043,-0.484537), glm::vec3(-0.0112664,-0.231043,-0.560038),
		glm::vec3(-0.535006,-0.336828,-9.72432e-10), glm::vec3(-0.464398,-0.336828,-0.266137), glm::vec3(-0.277404,-0.336828,-0.453132), glm::vec3(-0.0112664,-0.336828,-0.52374),
		glm::vec3(-0.467594,-0.405277,-9.72432e-10), glm::vec3(-0.406074,-0.405277,-0.231882), glm::vec3(-0.243148,-0.405277,-0.394808), glm::vec3(-0.0112664,-0.405277,-0.456328),
		glm::vec3(-0.431295,-0.441058,-9.72432e-10), glm::vec3(-0.374669,-0.441058,-0.213437), glm::vec3(-0.224703,-0.441058,-0.363403), glm::vec3(-0.0112664,-0.441058,-0.420029),
		glm::vec3(0.548772,-0.231043,-9.72432e-10), glm::vec3(0.47327,-0.231043,0.284582), glm::vec3(0.273316,-0.231043,0.484537), glm::vec3(-0.0112664,-0.231043,0.560038),
		glm::vec3(0.512473,-0.336828,-9.72432e-10), glm::vec3(0.441865,-0.336828,0.266137), glm::vec3(0.254871,-0.336828,0.453132), glm::vec3(-0.0112664,-0.336828,0.52374),
		glm::vec3(0.445061,-0.405277,-9.72432e-10), glm::vec3(0.383541,-0.405277,0.231882), glm::vec3(0.220616,-0.405277,0.394808), glm::vec3(-0.0112664,-0.405277,0.456328),
		glm::vec3(0.408762,-0.441058,-9.72432e-10), glm::vec3(0.352136,-0.441058,0.213437), glm::vec3(0.20217,-0.441058,0.363403), glm::vec3(-0.0112664,-0.441058,0.420029),
		glm::vec3(-0.0112664,-0.231043,0.560038), glm::vec3(-0.295849,-0.231043,0.484537), glm::vec3(-0.495803,-0.231043,0.284582), glm::vec3(-0.571305,-0.231043,-9.72432e-10),
		glm::vec3(-0.0112664,-0.336828,0.52374), glm::vec3(-0.277404,-0.336828,0.453132), glm::vec3(-0.464398,-0.336828,0.266137), glm::vec3(-0.535006,-0.336828,-9.72432e-10),
		glm::vec3(-0.0112664,-0.405277,0.456328), glm::vec3(-0.243148,-0.405277,0.394808), glm::vec3(-0.406074,-0.405277,0.231882), glm::vec3(-0.467594,-0.405277,-9.72432e-10),
		glm::vec3(-0.0112664,-0.441058,0.420029), glm::vec3(-0.224703,-0.441058,0.363403), glm::vec3(-0.374669,-0.441058,0.213437), glm::vec3(-0.431295,-0.441058,-9.72432e-10),
		glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10),
		glm::vec3(-0.0112664,0.375665,-0.1118), glm::vec3(0.0456664,0.375665,-0.0967887), glm::vec3(0.0855223,0.375665,-0.0569328), glm::vec3(0.100534,0.375665,-9.72432e-10),
		glm::vec3(-0.0112664,0.324328,-0.0730124), glm::vec3(0.0258955,0.324328,-0.0631997), glm::vec3(0.0519333,0.324328,-0.037162), glm::vec3(0.061746,0.324328,-9.72432e-10),
		glm::vec3(-0.0112664,0.272991,-0.0616042), glm::vec3(0.0200377,0.272991,-0.0532991), glm::vec3(0.0420326,0.272991,-0.0313041), glm::vec3(0.0503378,0.272991,-9.72432e-10),
		glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10),
		glm::vec3(-0.123067,0.375665,-9.72432e-10), glm::vec3(-0.108055,0.375665,-0.0569328), glm::vec3(-0.0681992,0.375665,-0.0967888), glm::vec3(-0.0112664,0.375665,-0.1118),
		glm::vec3(-0.0842788,0.324328,-9.72432e-10), glm::vec3(-0.0744661,0.324328,-0.037162), glm::vec3(-0.0484284,0.324328,-0.0631997), glm::vec3(-0.0112664,0.324328,-0.0730124),
		glm::vec3(-0.0728707,0.272991,-9.72432e-10), glm::vec3(-0.0645655,0.272991,-0.0313041), glm::vec3(-0.0425705,0.272991,-0.0532991), glm::vec3(-0.0112664,0.272991,-0.0616042),
		glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10),
		glm::vec3(0.100534,0.375665,-9.72432e-10), glm::vec3(0.0855223,0.375665,0.0569328), glm::vec3(0.0456663,0.375665,0.0967888), glm::vec3(-0.0112664,0.375665,0.1118),
		glm::vec3(0.061746,0.324328,-9.72432e-10), glm::vec3(0.0519333,0.324328,0.037162), glm::vec3(0.0258955,0.324328,0.0631997), glm::vec3(-0.0112664,0.324328,0.0730124),
		glm::vec3(0.0503378,0.272991,-9.72432e-10), glm::vec3(0.0420326,0.272991,0.0313041), glm::vec3(0.0200377,0.272991,0.0532991), glm::vec3(-0.0112664,0.272991,0.0616042),
		glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10), glm::vec3(-0.0112664,0.399,-9.72432e-10),
		glm::vec3(-0.0112664,0.375665,0.1118), glm::vec3(-0.0681992,0.375665,0.0967887), glm::vec3(-0.108055,0.375665,0.0569328), glm::vec3(-0.123067,0.375665,-9.72432e-10),
		glm::vec3(-0.0112664,0.324328,0.0730124), glm::vec3(-0.0484284,0.324328,0.0631997), glm::vec3(-0.0744661,0.324328,0.037162), glm::vec3(-0.0842788,0.324328,-9.72432e-10),
		glm::vec3(-0.0112664,0.272991,0.0616042), glm::vec3(-0.0425705,0.272991,0.0532991), glm::vec3(-0.0645655,0.272991,0.0313041), glm::vec3(-0.0728707,0.272991,-9.72432e-10),
		glm::vec3(-0.0112664,0.272991,-0.0616042), glm::vec3(0.0200377,0.272991,-0.0532991), glm::vec3(0.0420326,0.272991,-0.0313041), glm::vec3(0.0503378,0.272991,-9.72432e-10),
		glm::vec3(-0.0112664,0.241878,-0.176827), glm::vec3(0.0785879,0.241878,-0.152988), glm::vec3(0.141722,0.241878,-0.0898543), glm::vec3(0.165561,0.241878,-9.72432e-10),
		glm::vec3(-0.0112664,0.220099,-0.326274), glm::vec3(0.154529,0.220099,-0.282288), glm::vec3(0.271021,0.220099,-0.165796), glm::vec3(0.315008,0.220099,-9.72432e-10),
		glm::vec3(-0.0112664,0.188986,-0.400427), glm::vec3(0.19221,0.188986,-0.346444), glm::vec3(0.335177,0.188986,-0.203476), glm::vec3(0.389161,0.188986,-9.72432e-10),
		glm::vec3(-0.0728707,0.272991,-9.72432e-10), glm::vec3(-0.0645655,0.272991,-0.0313041), glm::vec3(-0.0425705,0.272991,-0.0532991), glm::vec3(-0.0112664,0.272991,-0.0616042),
		glm::vec3(-0.188093,0.241878,-9.72432e-10), glm::vec3(-0.164254,0.241878,-0.0898543), glm::vec3(-0.101121,0.241878,-0.152988), glm::vec3(-0.0112664,0.241878,-0.176827),
		glm::vec3(-0.337541,0.220099,-9.72432e-10), glm::vec3(-0.293554,0.220099,-0.165796), glm::vec3(-0.177062,0.220099,-0.282288), glm::vec3(-0.0112664,0.220099,-0.326274),
		glm::vec3(-0.411694,0.188986,-9.72432e-10), glm::vec3(-0.35771,0.188986,-0.203476), glm::vec3(-0.214743,0.188986,-0.346444), glm::vec3(-0.0112664,0.188986,-0.400427),
		glm::vec3(0.0503378,0.272991,-9.72432e-10), glm::vec3(0.0420326,0.272991,0.0313041), glm::vec3(0.0200377,0.272991,0.0532991), glm::vec3(-0.0112664,0.272991,0.0616042),
		glm::vec3(0.165561,0.241878,-9.72432e-10), glm::vec3(0.141722,0.241878,0.0898543), glm::vec3(0.0785879,0.241878,0.152988), glm::vec3(-0.0112664,0.241878,0.176827),
		glm::vec3(0.315008,0.220099,-9.72432e-10), glm::vec3(0.271021,0.220099,0.165796), glm::vec3(0.154529,0.220099,0.282288), glm::vec3(-0.0112664,0.220099,0.326274),
		glm::vec3(0.389161,0.188986,-9.72432e-10), glm::vec3(0.335177,0.188986,0.203476), glm::vec3(0.19221,0.188986,0.346444), glm::vec3(-0.0112664,0.188986,0.400427),
		glm::vec3(-0.0112664,0.272991,0.0616042), glm::vec3(-0.0425705,0.272991,0.0532991), glm::vec3(-0.0645655,0.272991,0.0313041), glm::vec3(-0.0728707,0.272991,-9.72432e-10),
		glm::vec3(-0.0112664,0.241878,0.176827), glm::vec3(-0.101121,0.241878,0.152988), glm::vec3(-0.164254,0.241878,0.0898543), glm::vec3(-0.188093,0.241878,-9.72432e-10),
		glm::vec3(-0.0112664,0.220099,0.326274), glm::vec3(-0.177062,0.220099,0.282288), glm::vec3(-0.293554,0.220099,0.165796), glm::vec3(-0.337541,0.220099,-9.72432e-10),
		glm::vec3(-0.0112664,0.188986,0.400427), glm::vec3(-0.214743,0.188986,0.346444), glm::vec3(-0.35771,0.188986,0.203476), glm::vec3(-0.411694,0.188986,-9.72432e-10),
		glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10),
		glm::vec3(0.274975,-0.476838,-9.72432e-10), glm::vec3(0.236386,-0.476838,-0.145453), glm::vec3(0.134187,-0.476838,-0.247652), glm::vec3(-0.0112664,-0.476838,-0.286242),
		glm::vec3(0.388539,-0.461281,-9.72432e-10), glm::vec3(0.334639,-0.461281,-0.20316), glm::vec3(0.191894,-0.461281,-0.345906), glm::vec3(-0.0112664,-0.461281,-0.399805),
		glm::vec3(0.408762,-0.441058,-9.72432e-10), glm::vec3(0.352136,-0.441058,-0.213437), glm::vec3(0.20217,-0.441058,-0.363403), glm::vec3(-0.0112664,-0.441058,-0.420029),
		glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10),
		glm::vec3(-0.0112664,-0.476838,-0.286242), glm::vec3(-0.15672,-0.476838,-0.247652), glm::vec3(-0.258919,-0.476838,-0.145453), glm::vec3(-0.297508,-0.476838,-9.72432e-10),
		glm::vec3(-0.0112664,-0.461281,-0.399805), glm::vec3(-0.214427,-0.461281,-0.345905), glm::vec3(-0.357172,-0.461281,-0.20316), glm::vec3(-0.411072,-0.461281,-9.72432e-10),
		glm::vec3(-0.0112664,-0.441058,-0.420029), glm::vec3(-0.224703,-0.441058,-0.363403), glm::vec3(-0.374669,-0.441058,-0.213437), glm::vec3(-0.431295,-0.441058,-9.72432e-10),
		glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10),
		glm::vec3(-0.0112664,-0.476838,0.286242), glm::vec3(0.134187,-0.476838,0.247652), glm::vec3(0.236386,-0.476838,0.145453), glm::vec3(0.274975,-0.476838,-9.72432e-10),
		glm::vec3(-0.0112664,-0.461281,0.399805), glm::vec3(0.191894,-0.461281,0.345905), glm::vec3(0.334639,-0.461281,0.20316), glm::vec3(0.388539,-0.461281,-9.72432e-10),
		glm::vec3(-0.0112664,-0.441058,0.420029), glm::vec3(0.20217,-0.441058,0.363403), glm::vec3(0.352136,-0.441058,0.213437), glm::vec3(0.408762,-0.441058,-9.72432e-10),
		glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10), glm::vec3(-0.0112664,-0.48306,-9.72432e-10),
		glm::vec3(-0.297508,-0.476838,-9.72432e-10), glm::vec3(-0.258919,-0.476838,0.145453), glm::vec3(-0.15672,-0.476838,0.247652), glm::vec3(-0.0112664,-0.476838,0.286242),
		glm::vec3(-0.411072,-0.461281,-9.72432e-10), glm::vec3(-0.357172,-0.461281,0.20316), glm::vec3(-0.214427,-0.461281,0.345906), glm::vec3(-0.0112664,-0.461281,0.399805),
		glm::vec3(-0.431295,-0.441058,-9.72432e-10), glm::vec3(-0.374669,-0.441058,0.213437), glm::vec3(-0.224703,-0.441058,0.363403), glm::vec3(-0.0112664,-0.441058,0.420029),
		glm::vec3(-0.431295,0.146983,-9.72432e-10), glm::vec3(-0.438555,0.130648,-0.0560038), glm::vec3(-0.452037,0.100313,-0.0560038), glm::vec3(-0.459297,0.0839785,-9.72432e-10),
		glm::vec3(-0.664645,0.142316,-9.72432e-10), glm::vec3(-0.654696,0.126586,-0.0560038), glm::vec3(-0.63622,0.0973744,-0.0560038), glm::vec3(-0.626271,0.0816449,-9.72432e-10),
		glm::vec3(-0.804654,0.109647,-9.72432e-10), glm::vec3(-0.785564,0.0981522,-0.0560038), glm::vec3(-0.75011,0.0768052,-0.0560038), glm::vec3(-0.731019,0.0653105,-9.72432e-10),
		glm::vec3(-0.851324,0.0209741,-9.72432e-10), glm::vec3(-0.829545,0.0209741,-0.0560038), glm::vec3(-0.789097,0.0209741,-0.0560038), glm::vec3(-0.767318,0.0209741,-9.72432e-10),
		glm::vec3(-0.459297,0.0839785,-9.72432e-10), glm::vec3(-0.452037,0.100313,0.0560038), glm::vec3(-0.438555,0.130648,0.0560038), glm::vec3(-0.431295,0.146983,-9.72432e-10),
		glm::vec3(-0.626271,0.0816449,-9.72432e-10), glm::vec3(-0.63622,0.0973743,0.0560038), glm::vec3(-0.654696,0.126586,0.0560038), glm::vec3(-0.664645,0.142316,-9.72432e-10),
		glm::vec3(-0.731019,0.0653105,-9.72432e-10), glm::vec3(-0.75011,0.0768051,0.0560038), glm::vec3(-0.785564,0.0981523,0.0560038), glm::vec3(-0.804654,0.109647,-9.72432e-10),
		glm::vec3(-0.767318,0.0209741,-9.72432e-10), glm::vec3(-0.789097,0.0209741,0.0560038), glm::vec3(-0.829545,0.0209741,0.0560038), glm::vec3(-0.851324,0.0209741,-9.72432e-10),
		glm::vec3(-0.851324,0.0209741,-9.72432e-10), glm::vec3(-0.829545,0.0209741,-0.0560038), glm::vec3(-0.789097,0.0209741,-0.0560038), glm::vec3(-0.767318,0.0209741,-9.72432e-10),
		glm::vec3(-0.818137,-0.101145,-9.72432e-10), glm::vec3(-0.799853,-0.0900541,-0.0560038), glm::vec3(-0.765897,-0.069456,-0.0560038), glm::vec3(-0.747613,-0.0583647,-9.72432e-10),
		glm::vec3(-0.7165,-0.213931,-9.72432e-10), glm::vec3(-0.708165,-0.197798,-0.0560038), glm::vec3(-0.692685,-0.167837,-0.0560038), glm::vec3(-0.68435,-0.151704,-9.72432e-10),
		glm::vec3(-0.543303,-0.315049,-9.72432e-10), glm::vec3(-0.550563,-0.29327,-0.0560038), glm::vec3(-0.564045,-0.252822,-0.0560038), glm::vec3(-0.571305,-0.231043,-9.72432e-10),
		glm::vec3(-0.767318,0.0209741,-9.72432e-10), glm::vec3(-0.789097,0.0209741,0.0560038), glm::vec3(-0.829545,0.0209741,0.0560038), glm::vec3(-0.851324,0.0209741,-9.72432e-10),
		glm::vec3(-0.747613,-0.0583647,-9.72432e-10), glm::vec3(-0.765897,-0.069456,0.0560038), glm::vec3(-0.799853,-0.0900541,0.0560038), glm::vec3(-0.818137,-0.101145,-9.72432e-10),
		glm::vec3(-0.68435,-0.151704,-9.72432e-10), glm::vec3(-0.692685,-0.167837,0.0560038), glm::vec3(-0.708165,-0.197798,0.0560038), glm::vec3(-0.7165,-0.213931,-9.72432e-10),
		glm::vec3(-0.571305,-0.231043,-9.72432e-10), glm::vec3(-0.564045,-0.252822,0.0560038), glm::vec3(-0.550563,-0.29327,0.0560038), glm::vec3(-0.543303,-0.315049,-9.72432e-10),
		glm::vec3(0.464766,-0.315049,-9.72432e-10), glm::vec3(0.464766,-0.255156,-0.123208), glm::vec3(0.464766,-0.143926,-0.123208), glm::vec3(0.464766,-0.084033,-9.72432e-10),
		glm::vec3(0.699153,-0.179706,-9.72432e-10), glm::vec3(0.679793,-0.141391,-0.103365), glm::vec3(0.64384,-0.0702338,-0.103365), glm::vec3(0.624481,-0.0319184,-9.72432e-10),
		glm::vec3(0.77175,0.0256412,-9.72432e-10), glm::vec3(0.747551,0.039959,-0.0665132), glm::vec3(0.70261,0.0665494,-0.0665132), glm::vec3(0.678411,0.0808671,-9.72432e-10),
		glm::vec3(0.912797,0.188986,-9.72432e-10), glm::vec3(0.869238,0.188986,-0.0466699), glm::vec3(0.788344,0.188986,-0.0466699), glm::vec3(0.744785,0.188986,-9.72432e-10),
		glm::vec3(0.464766,-0.084033,-9.72432e-10), glm::vec3(0.464766,-0.143926,0.123208), glm::vec3(0.464766,-0.255156,0.123208), glm::vec3(0.464766,-0.315049,-9.72432e-10),
		glm::vec3(0.624481,-0.0319184,-9.72432e-10), glm::vec3(0.64384,-0.0702338,0.103365), glm::vec3(0.679793,-0.141391,0.103365), glm::vec3(0.699153,-0.179706,-9.72432e-10),
		glm::vec3(0.678411,0.0808671,-9.72432e-10), glm::vec3(0.70261,0.0665492,0.0665132), glm::vec3(0.747551,0.0399591,0.0665132), glm::vec3(0.77175,0.0256412,-9.72432e-10),
		glm::vec3(0.744785,0.188986,-9.72432e-10), glm::vec3(0.788344,0.188986,0.0466699), glm::vec3(0.869238,0.188986,0.0466699), glm::vec3(0.912797,0.188986,-9.72432e-10),
		glm::vec3(0.912797,0.188986,-9.72432e-10), glm::vec3(0.869238,0.188986,-0.0466699), glm::vec3(0.788344,0.188986,-0.0466699), glm::vec3(0.744785,0.188986,-9.72432e-10),
		glm::vec3(0.949096,0.207654,-9.72432e-10), glm::vec3(0.902848,0.206444,-0.04183), glm::vec3(0.816961,0.204196,-0.04183), glm::vec3(0.770713,0.202987,-9.72432e-10),
		glm::vec3(0.937169,0.20882,-9.72432e-10), glm::vec3(0.897509,0.207308,-0.0328418), glm::vec3(0.823855,0.204499,-0.0328418), glm::vec3(0.784196,0.202987,-9.72432e-10),
		glm::vec3(0.884795,0.188986,-9.72432e-10), glm::vec3(0.855756,0.188986,-0.0280019), glm::vec3(0.801826,0.188986,-0.0280019), glm::vec3(0.772787,0.188986,-9.72432e-10),
		glm::vec3(0.744785,0.188986,-9.72432e-10), glm::vec3(0.788344,0.188986,0.0466699), glm::vec3(0.869238,0.188986,0.0466699), glm::vec3(0.912797,0.188986,-9.72432e-10),
		glm::vec3(0.770713,0.202987,-9.72432e-10), glm::vec3(0.81696,0.204196,0.04183), glm::vec3(0.902848,0.206444,0.04183), glm::vec3(0.949096,0.207654,-9.72432e-10),
		glm::vec3(0.784196,0.202987,-9.72432e-10), glm::vec3(0.823855,0.204499,0.0328418), glm::vec3(0.897509,0.207308,0.0328418), glm::vec3(0.937169,0.20882,-9.72432e-10),
		glm::vec3(0.772787,0.188986,-9.72432e-10), glm::vec3(0.801826,0.188986,0.0280019), glm::vec3(0.855756,0.188986,0.0280019), glm::vec3(0.884795,0.188986,-9.72432e-10)
	};

// Teapot indices of triangles
	std::vector<unsigned int> indices = {
		0,5,4, 0,1,5, 1,6,5, 1,2,6, 2,7,6, 2,3,7, 4,9,8, 4,5,9, 5,10,9, 5,6,10,
		6,11,10, 6,7,11, 8,13,12, 8,9,13, 9,14,13, 9,10,14, 10,15,14, 10,11,15, 16,21,20, 16,17,21,
		17,22,21, 17,18,22, 18,23,22, 18,19,23, 20,25,24, 20,21,25, 21,26,25, 21,22,26, 22,27,26, 22,23,27,
		24,29,28, 24,25,29, 25,30,29, 25,26,30, 26,31,30, 26,27,31, 32,37,36, 32,33,37, 33,38,37, 33,34,38,
		34,39,38, 34,35,39, 36,41,40, 36,37,41, 37,42,41, 37,38,42, 38,43,42, 38,39,43, 40,45,44, 40,41,45,
		41,46,45, 41,42,46, 42,47,46, 42,43,47, 48,53,52, 48,49,53, 49,54,53, 49,50,54, 50,55,54, 50,51,55,
		52,57,56, 52,53,57, 53,58,57, 53,54,58, 54,59,58, 54,55,59, 56,61,60, 56,57,61, 57,62,61, 57,58,62,
		58,63,62, 58,59,63, 64,69,68, 64,65,69, 65,70,69, 65,66,70, 66,71,70, 66,67,71, 68,73,72, 68,69,73,
		69,74,73, 69,70,74, 70,75,74, 70,71,75, 72,77,76, 72,73,77, 73,78,77, 73,74,78, 74,79,78, 74,75,79,
		80,85,84, 80,81,85, 81,86,85, 81,82,86, 82,87,86, 82,83,87, 84,89,88, 84,85,89, 85,90,89, 85,86,90,
		86,91,90, 86,87,91, 88,93,92, 88,89,93, 89,94,93, 89,90,94, 90,95,94, 90,91,95, 96,101,100, 96,97,101,
		97,102,101, 97,98,102, 98,103,102, 98,99,103, 100,105,104, 100,101,105, 101,106,105, 101,102,106, 102,107,106, 102,103,107,
		104,109,108, 104,105,109, 105,110,109, 105,106,110, 106,111,110, 106,107,111, 112,117,116, 112,113,117, 113,118,117, 113,114,118,
		114,119,118, 114,115,119, 116,121,120, 116,117,121, 117,122,121, 117,118,122, 118,123,122, 118,119,123, 120,125,124, 120,121,125,
		121,126,125, 121,122,126, 122,127,126, 122,123,127, 128,133,132, 128,129,133, 129,134,133, 129,130,134, 130,135,134, 130,131,135,
		132,137,136, 132,133,137, 133,138,137, 133,134,138, 134,139,138, 134,135,139, 136,141,140, 136,137,141, 137,142,141, 137,138,142,
		138,143,142, 138,139,143, 144,149,148, 144,145,149, 145,150,149, 145,146,150, 146,151,150, 146,147,151, 148,153,152, 148,149,153,
		149,154,153, 149,150,154, 150,155,154, 150,151,155, 152,157,156, 152,153,157, 153,158,157, 153,154,158, 154,159,158, 154,155,159,
		160,165,164, 160,161,165, 161,166,165, 161,162,166, 162,167,166, 162,163,167, 164,169,168, 164,165,169, 165,170,169, 165,166,170,
		166,171,170, 166,167,171, 168,173,172, 168,169,173, 169,174,173, 169,170,174, 170,175,174, 170,171,175, 176,181,180, 176,177,181,
		177,182,181, 177,178,182, 178,183,182, 178,179,183, 180,185,184, 180,181,185, 181,186,185, 181,182,186, 182,187,186, 182,183,187,
		184,189,188, 184,185,189, 185,190,189, 185,186,190, 186,191,190, 186,187,191, 192,197,196, 192,193,197, 193,198,197, 193,194,198,
		194,199,198, 194,195,199, 196,201,200, 196,197,201, 197,202,201, 197,198,202, 198,203,202, 198,199,203, 200,205,204, 200,201,205,
		201,206,205, 201,202,206, 202,207,206, 202,203,207, 208,213,212, 208,209,213, 209,214,213, 209,210,214, 210,215,214, 210,211,215,
		212,217,216, 212,213,217, 213,218,217, 213,214,218, 214,219,218, 214,215,219, 216,221,220, 216,217,221, 217,222,221, 217,218,222,
		218,223,222, 218,219,223, 224,229,228, 224,225,229, 225,230,229, 225,226,230, 226,231,230, 226,227,231, 228,233,232, 228,229,233,
		229,234,233, 229,230,234, 230,235,234, 230,231,235, 232,237,236, 232,233,237, 233,238,237, 233,234,238, 234,239,238, 234,235,239,
		240,245,244, 240,241,245, 241,246,245, 241,242,246, 242,247,246, 242,243,247, 244,249,248, 244,245,249, 245,250,249, 245,246,250,
		246,251,250, 246,247,251, 248,253,252, 248,249,253, 249,254,253, 249,250,254, 250,255,254, 250,251,255, 256,261,260, 256,257,261,
		257,262,261, 257,258,262, 258,263,262, 258,259,263, 260,265,264, 260,261,265, 261,266,265, 261,262,266, 262,267,266, 262,263,267,
		264,269,268, 264,265,269, 265,270,269, 265,266,270, 266,271,270, 266,267,271, 272,277,276, 272,273,277, 273,278,277, 273,274,278,
		274,279,278, 274,275,279, 276,281,280, 276,277,281, 277,282,281, 277,278,282, 278,283,282, 278,279,283, 280,285,284, 280,281,285,
		281,286,285, 281,282,286, 282,287,286, 282,283,287, 288,293,292, 288,289,293, 289,294,293, 289,290,294, 290,295,294, 290,291,295,
		292,297,296, 292,293,297, 293,298,297, 293,294,298, 294,299,298, 294,295,299, 296,301,300, 296,297,301, 297,302,301, 297,298,302,
		298,303,302, 298,299,303, 304,309,308, 304,305,309, 305,310,309, 305,306,310, 306,311,310, 306,307,311, 308,313,312, 308,309,313,
		309,314,313, 309,310,314, 310,315,314, 310,311,315, 312,317,316, 312,313,317, 313,318,317, 313,314,318, 314,319,318, 314,315,319,
		320,325,324, 320,321,325, 321,326,325, 321,322,326, 322,327,326, 322,323,327, 324,329,328, 324,325,329, 325,330,329, 325,326,330,
		326,331,330, 326,327,331, 328,333,332, 328,329,333, 329,334,333, 329,330,334, 330,335,334, 330,331,335, 336,341,340, 336,337,341,
		337,342,341, 337,338,342, 338,343,342, 338,339,343, 340,345,344, 340,341,345, 341,346,345, 341,342,346, 342,347,346, 342,343,347,
		344,349,348, 344,345,349, 345,350,349, 345,346,350, 346,351,350, 346,347,351, 352,357,356, 352,353,357, 353,358,357, 353,354,358,
		354,359,358, 354,355,359, 356,361,360, 356,357,361, 357,362,361, 357,358,362, 358,363,362, 358,359,363, 360,365,364, 360,361,365,
		361,366,365, 361,362,366, 362,367,366, 362,363,367, 368,373,372, 368,369,373, 369,374,373, 369,370,374, 370,375,374, 370,371,375,
		372,377,376, 372,373,377, 373,378,377, 373,374,378, 374,379,378, 374,375,379, 376,381,380, 376,377,381, 377,382,381, 377,378,382,
		378,383,382, 378,379,383, 384,389,388, 384,385,389, 385,390,389, 385,386,390, 386,391,390, 386,387,391, 388,393,392, 388,389,393,
		389,394,393, 389,390,394, 390,395,394, 390,391,395, 392,397,396, 392,393,397, 393,398,397, 393,394,398, 394,399,398, 394,395,399,
		400,405,404, 400,401,405, 401,406,405, 401,402,406, 402,407,406, 402,403,407, 404,409,408, 404,405,409, 405,410,409, 405,406,410,
		406,411,410, 406,407,411, 408,413,412, 408,409,413, 409,414,413, 409,410,414, 410,415,414, 410,411,415, 416,421,420, 416,417,421,
		417,422,421, 417,418,422, 418,423,422, 418,419,423, 420,425,424, 420,421,425, 421,426,425, 421,422,426, 422,427,426, 422,423,427,
		424,429,428, 424,425,429, 425,430,429, 425,426,430, 426,431,430, 426,427,431, 432,437,436, 432,433,437, 433,438,437, 433,434,438,
		434,439,438, 434,435,439, 436,441,440, 436,437,441, 437,442,441, 437,438,442, 438,443,442, 438,439,443, 440,445,444, 440,441,445,
		441,446,445, 441,442,446, 442,447,446, 442,443,447, 448,453,452, 448,449,453, 449,454,453, 449,450,454, 450,455,454, 450,451,455,
		452,457,456, 452,453,457, 453,458,457, 453,454,458, 454,459,458, 454,455,459, 456,461,460, 456,457,461, 457,462,461, 457,458,462,
		458,463,462, 458,459,463, 464,469,468, 464,465,469, 465,470,469, 465,466,470, 466,471,470, 466,467,471, 468,473,472, 468,469,473,
		469,474,473, 469,470,474, 470,475,474, 470,471,475, 472,477,476, 472,473,477, 473,478,477, 473,474,478, 474,479,478, 474,475,479,
		480,485,484, 480,481,485, 481,486,485, 481,482,486, 482,487,486, 482,483,487, 484,489,488, 484,485,489, 485,490,489, 485,486,490,
		486,491,490, 486,487,491, 488,493,492, 488,489,493, 489,494,493, 489,490,494, 490,495,494, 490,491,495, 496,501,500, 496,497,501,
		497,502,501, 497,498,502, 498,503,502, 498,499,503, 500,505,504, 500,501,505, 501,506,505, 501,502,506, 502,507,506, 502,503,507,
		504,509,508, 504,505,509, 505,510,509, 505,506,510, 506,511,510, 506,507,511
	};

	// random number generator for assigning random per-vertex normal data.
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

	// Create a vector to interleave and pack all vertex data into one vector.
	std::vector<Vertex> vData( positions.size() );
	for( unsigned int i = 0; i < vData.size(); i++ ) {
		vData[i].position = positions[i];
		vData[i].normal = glm::vec3( dis(gen), dis(gen), dis(gen) );
	}

	mNumObjectIndices = static_cast<uint32_t>(indices.size());
	const auto device = vklGetDevice();
	auto dispatchLoader = vk::DispatchLoaderStatic();

	// 1. Vertex BUFFER (Buffer, Memory, Bind 'em together, copy data into it)
	{
		// Use VulkanLaunchpad functionality to manage buffers
		// All vertex data is in one vector, copied to one buffer
		// on the GPU
		mObjectVertexData = vklCreateHostCoherentBufferAndUploadData(
			vData.data(), sizeof(vData[0]) * vData.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	}

	// 2. INDICES BUFFER (Buffer, Memory, Bind 'em together, copy data into it)
	{
		mObjectIndices = vklCreateHostCoherentBufferAndUploadData(
			indices.data(), sizeof(indices[0]) * indices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	}

	// Now Create the pipeline
	objectCreatePipeline();
}


// Cleanup buffers and pipeline created on the GPU
void objectDestroyBuffers() {
	auto device = vklGetDevice();
	vkDeviceWaitIdle( device );
	vklDestroyGraphicsPipeline(pipeline);
	vklDestroyHostCoherentBufferAndItsBackingMemory( mObjectVertexData );
	vklDestroyHostCoherentBufferAndItsBackingMemory( mObjectIndices );
}

void objectDraw() {
	objectDraw( pipeline );
}

void objectDraw(VkPipeline pipeline)
{
	if (!vklFrameworkInitialized()) {
		VKL_EXIT_WITH_ERROR("Framework not initialized. Ensure to invoke vklFrameworkInitialized beforehand!");
	}
	const vk::CommandBuffer& cb = vklGetCurrentCommandBuffer();
	auto currentSwapChainImageIndex = vklGetCurrentSwapChainImageIndex();
	assert(currentSwapChainImageIndex < vklGetNumFramebuffers());
	assert(currentSwapChainImageIndex < vklGetNumClearValues());

	cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

	cb.bindVertexBuffers(0u, { vk::Buffer{ objectGetVertexBuffer() } }, { vk::DeviceSize{ 0 } });
	cb.bindIndexBuffer(vk::Buffer{ objectGetIndicesBuffer() }, vk::DeviceSize{ 0 }, vk::IndexType::eUint32);

	// update push constants on every draw call and send them over to the GPU.
    // upload the matrix to the GPU via push constants
	objectUpdateConstants();
    vklSetPushConstants(
			pipeline,
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			&pushConstants,
			sizeof(ObjectPushConstants)
		);

	cb.drawIndexed(objectGetNumIndices(), 1u, 0u, 0, 0u);
}

VkBuffer objectGetVertexBuffer() {
	return static_cast<VkBuffer>(mObjectVertexData);
}

VkBuffer objectGetIndicesBuffer() {
	return static_cast<VkBuffer>(mObjectIndices);
}

uint32_t objectGetNumIndices() {
	return mNumObjectIndices;
}

void objectCreatePipeline() {

	// initialize push constants
	pushConstants.model = glm::mat4{ 1.0f };

	// a right-handed view coordinate system coincident with the x y and z axes
	// and located along the positive z axis, looking down the negative z axis.
	glm::mat4 view = glm::mat4{
		glm::vec4{ 1.f,  0.f,  0.f,  0.f},
		glm::vec4{ 0.f,  1.f,  0.f,  0.f},
		glm::vec4{ 0.f,  0.f,  1.f,  0.f},
		glm::vec4{ 0.f,  0.25f,  2.f,  1.f},
	};
	pushConstants.view = glm::inverse( view );

	// Create a projection matrix compatible with Vulkan.
	// The resulting matrix takes care of the y-z flip.
	pushConstants.proj = vklCreatePerspectiveProjectionMatrix(glm::pi<float>() / 3.0f, 1.0f, 1.0f, 3.0f );

	// ------------------------------
	// Pipeline creation
	// ------------------------------

	VklGraphicsPipelineConfig config{};
		config.enableAlphaBlending = false;
		// path to shaders may need to be modified depending on the location
		// of the executable
		config.vertexShaderPath = "../../HW2/src/starter.vert";
		config.fragmentShaderPath = "../../HW2/src/starter.frag";

		// Can set polygonDrawMode to VK_POLYGON_MODE_LINE for wireframe rendering
		config.polygonDrawMode = VK_POLYGON_MODE_FILL;
		config.triangleCullingMode = VK_CULL_MODE_BACK_BIT;

		// Binding for vertex buffer, using 1 buffer with per-vertex rate.
		// This will send per-vertex data to the GPU.
		config.vertexInputBuffers.emplace_back(VkVertexInputBindingDescription{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		});

		// Positions at locaion 0
		config.inputAttributeDescriptions.emplace_back(VkVertexInputAttributeDescription{
			//.location = static_cast<uint32_t>(config.inputAttributeDescriptions.size()),
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		});

		// Normals at location 1
		config.inputAttributeDescriptions.emplace_back(VkVertexInputAttributeDescription{
			//.location = static_cast<uint32_t>(config.inputAttributeDescriptions.size()),
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		});

		// Push constants should be available in both the vertex and fragment shaders
		config.pushConstantRanges.emplace_back(VkPushConstantRange{
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(ObjectPushConstants),
		});
	pipeline = vklCreateGraphicsPipeline( config );
}

// Function to update push constants
// Left click for rotate, right click for pan, scroll to zoom
void objectUpdateConstants() {

	// Update camera with current mouse input
	// Works because glfwPollEvents is called in render loop in main
	vklUpdateCamera(mCameraHandle);

	// Update projection and view matrix using camera
	pushConstants.view = vklGetCameraViewMatrix(mCameraHandle);
	pushConstants.proj = vklGetCameraProjectionMatrix(mCameraHandle);

	// Roll pitch and yaw model transformations
	glm::mat4 roll_matrix = glm::rotate(glm::mat4(1.0f), roll, glm::vec3(0.0f, 0.0f, 1.0f) );
	glm::mat4 pitch_matrix = glm::rotate(glm::mat4(1.0f), pitch, glm::vec3(1.0f, 0.0f, 0.0f) );
	glm::mat4 yaw_matrix = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f) );

	pushConstants.model = roll_matrix * pitch_matrix * yaw_matrix;
}
