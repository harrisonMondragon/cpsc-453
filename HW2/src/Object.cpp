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

// Variables for bounding box
float bound_x, bound_y, bound_z;
float centroid_x, centroid_y, centroid_z;

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
	glm::vec3 cameraPos;
};

// MVP matrices that are updated interactively
ObjectPushConstants pushConstants;

// Model transformations
extern float scale;
extern float extrinsic_x;
extern float extrinsic_y;
extern float extrinsic_z;
extern float intrinsic_x;
extern float intrinsic_y;
extern float intrinsic_z;

// Organize geometry data and send it to the GPU
void objectCreateGeometryAndBuffers(std::string objPath, GLFWwindow* window)
{
	// Load geometry from file specified via command line
	VklGeometryData modelGeometry = vklLoadModelGeometry(objPath);

	// Create a camera object for the window size
	mCameraHandle = vklCreateCamera(window);

	// random number generator for assigning random per-vertex normal data.
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

	// Set up variables for bounding box and centroid calcs
	float max_x =  modelGeometry.positions[0].x;
	float max_y =  modelGeometry.positions[0].y;
	float max_z =  modelGeometry.positions[0].z;

	float min_x =  modelGeometry.positions[0].x;
	float min_y =  modelGeometry.positions[0].y;
	float min_z =  modelGeometry.positions[0].z;

	float total_x = 0;
	float total_y = 0;
	float total_z = 0;

	// Create a vector to interleave and pack all vertex data into one vector.
	std::vector<Vertex> vData( modelGeometry.positions.size() );
	for(unsigned int i = 0; i < vData.size(); i++ ) {
		// Copy position
		vData[i].position = modelGeometry.positions[i];

		// Get max coordinates
		if(vData[i].position[0] > max_x){
			max_x = vData[i].position.x;
		}
		if(vData[i].position[1] > max_y){
			max_y = vData[i].position.y;
		}
		if(vData[i].position[2] > max_z){
			max_z = vData[i].position.z;
		}

		// Get min coordinates
		if(vData[i].position[0] < min_x){
			min_x = vData[i].position.x;
		}
		if(vData[i].position[1] < min_y){
			min_y = vData[i].position.y;
		}
		if(vData[i].position[2] < min_z){
			min_z = vData[i].position.z;
		}

		// Add to totals
		total_x +=  vData[i].position.x;
		total_y +=  vData[i].position.y;
		total_z +=  vData[i].position.z;

		// Calculate face normals and add them to the all positions of the face
		// These sums get normalized by the fragment shader
		glm::vec3 triA = modelGeometry.positions[modelGeometry.indices[3*i]];
		glm::vec3 triB = modelGeometry.positions[modelGeometry.indices[3*i+1]];
		glm::vec3 triC = modelGeometry.positions[modelGeometry.indices[3*i+2]];

		glm::vec3 faceNormal = glm::cross(triB - triA, triC - triA);

		vData[modelGeometry.indices[3*i]].normal += faceNormal;
		vData[modelGeometry.indices[3*i+1]].normal += faceNormal;
		vData[modelGeometry.indices[3*i+2]].normal += faceNormal;
	}

	// Calculate bounding box
	bound_x = max_x - min_x;
	bound_y = max_y - min_y;
	bound_z = max_z - min_z;

	// Calculate centroids
	centroid_x = total_x / modelGeometry.positions.size();
	centroid_y = total_y / modelGeometry.positions.size();
	centroid_z = total_z / modelGeometry.positions.size();

	mNumObjectIndices = static_cast<uint32_t>(modelGeometry.indices.size());
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
			modelGeometry.indices.data(), sizeof(modelGeometry.indices[0]) * modelGeometry.indices.size(),
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

	// TODO: Ask TA of this bounding box logic is good, hardcoded version is below
	float bounding_box = glm::max(bound_x, glm::max(bound_y, bound_z));
	pushConstants.model = glm::scale(glm::mat4(1.0f), glm::vec3(5.f/bounding_box));
	// pushConstants.model = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));

	// Initialize model. Do not need to initialize view and proj because they will get reset
	// by the camera in objectUpdateConstants anyways
	pushConstants.model = glm::translate(pushConstants.model, glm::vec3(-centroid_x,-centroid_y,-centroid_z));

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

		// Positions at location 0
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
	vklUpdateCamera(mCameraHandle);

	// Update view, proj, and cameraPos
	pushConstants.view = vklGetCameraViewMatrix(mCameraHandle);
	pushConstants.proj = vklGetCameraProjectionMatrix(mCameraHandle);
	pushConstants.cameraPos = vklGetCameraPosition(mCameraHandle);

	// Scale model transformation
	glm::mat4 scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
	pushConstants.model = pushConstants.model * scale_matrix;

	// Extrinsic rotation model transformations (rotate around global axes)
	glm::mat4 extrinsic_x_matrix = glm::rotate(glm::mat4(1.0f), extrinsic_x, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 extrinsic_y_matrix = glm::rotate(glm::mat4(1.0f), extrinsic_y, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 extrinsic_z_matrix = glm::rotate(glm::mat4(1.0f), extrinsic_z, glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 extrinsic_matrix =  extrinsic_x_matrix * extrinsic_y_matrix * extrinsic_z_matrix;
	pushConstants.model = extrinsic_matrix * pushConstants.model;

	// Intrinsic rotation model transformations (rotate around object axes)
	glm::mat4 intrinsic_x_matrix = glm::rotate(glm::mat4(1.0f), intrinsic_x, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 intrinsic_y_matrix = glm::rotate(glm::mat4(1.0f), intrinsic_y, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 intrinsic_z_matrix = glm::rotate(glm::mat4(1.0f), intrinsic_z, glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 intrinsic_matrix =  intrinsic_x_matrix * intrinsic_y_matrix * intrinsic_z_matrix;
	pushConstants.model = pushConstants.model * intrinsic_matrix;

	intrinsic_x = 0;
	intrinsic_y = 0;
	intrinsic_z = 0;
	extrinsic_x = 0;
	extrinsic_y = 0;
	extrinsic_z = 0;
	scale = 1;
}
