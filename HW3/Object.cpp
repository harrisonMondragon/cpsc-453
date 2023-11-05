/*
 * Copyright 2023 University of Calgary, Visualization and Graphics Group
 */

#include "Object.h"
#include <VulkanLaunchpad.h>
#include <vulkan/vulkan.hpp>
#include <glm/gtx/normal.hpp>
#include "Camera.h"
#include "Path.hpp"

using namespace shared;

// buffers that will live on the GPU.
// No geometry retained on the CPU, all data sent to the GPU.

uint32_t mNumObjectIndices;
VkBuffer mObjectVertexData;
VkBuffer mObjectIndices;

// A pipeline that can be used for HW2
VkPipeline pipeline;

// Struct to pack object vertex data
struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 tex;
}; 

// Send model, view and projection matrices as push constants
// which are packed in this struct
struct ObjectPushConstants {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
} pushConstants;

// A structure to hold object bounds
struct ObjectBounds {
	glm::vec3 min; 
	glm::vec3 max; 
	float radius;
} ob;

// scale and rotation controls
extern float scale;
extern glm::mat4 orientation;

// Load geometry data from a specified obj file.
// Geometry data is assumed to have per-vertex positions, normals and texture corrdinates
// as well as face indices.
void objectCreateGeometryAndBuffers( const std::string& path_to_obj, GLFWwindow* window ) 
{
	VklGeometryData data = vklLoadModelGeometry( path_to_obj );

	std::cout << "Read...\n" << 
		'\t' << data.positions.size() << " vertices,\n" <<
		'\t' << data.normals.size() << " normals,\n" << 
		'\t' << data.textureCoordinates.size() << " texture coordinates,\n" <<
		'\t' << data.indices.size() << " indices\n"
		<<  "...from " << path_to_obj << std::endl;

	// Determine object bounding box and radius
	glm::vec3 min {FLT_MAX, FLT_MAX, FLT_MAX};
	glm::vec3 max {-FLT_MAX, -FLT_MAX, -FLT_MAX};
	for(auto & point : data.positions)  {
		for( uint32_t i = 0; i < 3; i++ ) {
			if( point[i] < min[i]) {
				min[i] = point[i];
			}
			if( point[i] > max[i]) {
				max[i] = point[i];
			}
		}
	}
	float radius = glm::length(max - (min + max) * 0.5f);
	std::cout << "Bounding Box: " << "( " << min.x << ", " << min.y << ", " << min.z << " ) -- "  
		<< "( " << max.x << ", " << max.y << ", " << max.z << " )" << std::endl;
	std::cout << "Radius of circumscribing sphere: " << radius << std::endl; 

	// initialize bounds and orientation
	ob.min = min;
	ob.max = max;
	ob.radius = radius;
	orientation = glm::mat4(1.0f);

	// Create a vector to interleave and pack all vertex data into one vector.
	std::cout << "Packing vertices, normals and texture coordinates" << std::endl;
	std::vector<Vertex> vData;
	vData.resize( data.positions.size() );
	for( uint32_t i = 0; i < vData.size(); i++ ) {
		vData[i].position = data.positions[i];
		vData[i].normal = data.normals[i];
		vData[i].tex = data.textureCoordinates[i];
	}
	std::cout << "...done packing" << std::endl << std::flush;

	mNumObjectIndices = static_cast<uint32_t>(data.indices.size());
	const auto device = vklGetDevice();
	auto dispatchLoader = vk::DispatchLoaderStatic();

	std:: cout << "Sending packed data to GPU..." << std::endl;

	// 1. Vertex BUFFER (Buffer, Memory, Bind 'em together, copy data into it)
	{ 
		// Use VulkanLaunchpad functionality to manage buffers
		// All vertex data is in one vector, copied to one buffer
		// on the GPU
		mObjectVertexData = vklCreateHostCoherentBufferAndUploadData(
			vData.data(), sizeof(vData[0]) * vData.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	}

	std:: cout << "Sending face index data to GPU..." << std::endl;

	// 2. INDICES BUFFER (Buffer, Memory, Bind 'em together, copy data into it)
	{
		mObjectIndices = vklCreateHostCoherentBufferAndUploadData(
			data.indices.data(), sizeof(data.indices[0]) * data.indices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	}

	// Now Create the camera and pipeline
	std::cout << "Now Creating Camera and Pipeline " << std::endl;
	objectCreateCamera( window );
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

	// use vklCmdBindPipeline for shader hot reloading
	vklCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline );

	
	cb.bindVertexBuffers(0u, { vk::Buffer{ objectGetVertexBuffer() } }, { vk::DeviceSize{ 0 } });
	cb.bindIndexBuffer(vk::Buffer{ objectGetIndicesBuffer() }, vk::DeviceSize{ 0 }, vk::IndexType::eUint32);

	// Update things that need to be updated per draw call
	
	// update push constants on every draw call and send them over to the GPU.
    // upload the matrix to the GPU via push constants
	objectUpdateConstants( nullptr );
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

	// ------------------------------
	// Pipeline creation
	// ------------------------------

	auto const vertShaderPath = Path::Instance->Get("shaders/hW3/starter.vert");
	auto const fragShaderPath = Path::Instance->Get("shaders/hW3/starter.frag");

	VklGraphicsPipelineConfig config{};
		config.enableAlphaBlending = false;
		// path to shaders may need to be modified depending on the location
		// of the executable
		config.vertexShaderPath = vertShaderPath.c_str();
		config.fragmentShaderPath = fragShaderPath.c_str();
		
		// Can set polygonDrawMode to VK_POLYGON_MODE_LINE for wireframe rendering
		// if supported by GPU
		config.polygonDrawMode = VK_POLYGON_MODE_FILL;
		
		// Back face culling may or may not be needed.
		// Uncomment this to enable it.
		// config.triangleCullingMode = VK_CULL_MODE_BACK_BIT;

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

		// Texture coordinates at location 2
		config.inputAttributeDescriptions.emplace_back(VkVertexInputAttributeDescription{
			//.location = static_cast<uint32_t>(config.inputAttributeDescriptions.size()),
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, tex),
		});
            
		// Push constants should be available in both the vertex and fragment shaders
		config.pushConstantRanges.emplace_back(VkPushConstantRange{
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(ObjectPushConstants),
		});
	pipeline = vklCreateGraphicsPipeline( config );		
}

// Function to update push constants.
// For the starter example, only the model matrix is updated.
void objectUpdateConstants( GLFWwindow* window ) {
	
	// center the object so that rotations and scale are about the center.
	// This is applied as a modeling transformation before handling rotation/orientation
	// and scale
	
	glm::vec3 center = (ob.min + ob.max) * 0.5f;
	pushConstants.model = glm::scale(glm::mat4(1.0f), glm::vec3{scale, scale, scale} ) *
			orientation * glm::translate(glm::mat4(1.0f), -center );

	// fixed camera -> no view matrix update

	// if the window has changed size, update the push constants as well
	if( window != nullptr ) {
		int width, height;
		glfwGetWindowSize(window, &width, &height);

		pushConstants.proj = vklCreatePerspectiveProjectionMatrix(glm::radians(60.0f), 
			static_cast<float>(width) / static_cast<float>(height), ob.radius, 5.0f*ob.radius);
		}

}

// Function to create camera
void objectCreateCamera( GLFWwindow* window ) {
	int width, height;
	glfwGetWindowSize(window, &width, &height);

	// assume a default right handed camera looking down -z.
	// We'll translate the camera so that the full bounding box is in view.
	// This assumes that the object has been properly centered via model transformations
	// before the viewing.

	pushConstants.view = glm::translate(glm::mat4(1.0f), glm::vec3{0.f, 0.f, -3.0*ob.radius} );
	pushConstants.proj = vklCreatePerspectiveProjectionMatrix(glm::radians(60.0f), 
		static_cast<float>(width) / static_cast<float>(height), ob.radius, 5.0f*ob.radius);
}
