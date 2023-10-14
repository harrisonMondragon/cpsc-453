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

// Model transformations
extern float scale;
extern float roll;
extern float pitch;
extern float yaw;

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

	// Create a vector to interleave and pack all vertex data into one vector.
	std::vector<Vertex> vData( modelGeometry.positions.size() );
	for( unsigned int i = 0; i < vData.size(); i++ ) {
		vData[i].position = modelGeometry.positions[i];
		vData[i].normal = glm::vec3( dis(gen), dis(gen), dis(gen) );
	}

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

	// Scale model transformations
	glm::mat4 scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(scale));

	// Roll pitch and yaw model transformations
	glm::mat4 roll_matrix = glm::rotate(glm::mat4(1.0f), roll, glm::vec3(0.0f, 0.0f, 1.0f) );
	glm::mat4 pitch_matrix = glm::rotate(glm::mat4(1.0f), pitch, glm::vec3(1.0f, 0.0f, 0.0f) );
	glm::mat4 yaw_matrix = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f) );
	glm::mat4 intrinsic_matrix = roll_matrix * pitch_matrix * yaw_matrix;

	pushConstants.model = scale_matrix * intrinsic_matrix;
}
