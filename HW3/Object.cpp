/*
 * Copyright 2023 University of Calgary, Visualization and Graphics Group
 */

#include "Object.h"
#include <VulkanLaunchpad.h>
#include <vulkan/vulkan.hpp>
#include <glm/gtx/normal.hpp>
#include "Camera.h"
#include "Path.hpp"
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace shared;

// Variables needed from main
extern VkPhysicalDevice vk_physical_device;
extern uint32_t selected_queue_family_index;
extern bool procedural_texturing;
extern bool ambient_occlusion;

// buffers that will live on the GPU.
// No geometry retained on the CPU, all data sent to the GPU.
uint32_t mNumObjectIndices;
VkBuffer mObjectVertexData;
VkBuffer mObjectIndices;

// Global variables
VkImageView textureImageView;
VkSampler textureSampler;
VkImage textureImage;
VkDeviceMemory textureImageMemory;

VkImageView aoImageView;
VkSampler aoSampler;
VkImage aoImage;
VkDeviceMemory aoImageMemory;

VkImageView proceduralImageView;
VkSampler proceduralSampler;
VkImage proceduralImage;
VkDeviceMemory proceduralImageMemory;

VkCommandPool commandPool;
VkDescriptorPool descriptorPool;
VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorSet ds;

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
	alignas(4) bool proc;
	alignas(4) bool ao;
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
void objectCreateGeometryAndBuffers( const std::string& path_to_obj, const char* path_to_tex, const char* path_to_ao, GLFWwindow* window )
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

	// START ----- createCommandPool from tutorial
	createCommandPool();
	// END ----- createCommandPool from tutorial

	// START ----- createTextureImage from tutorial
	createTextureImageFromFile(path_to_tex, textureImage, textureImageMemory);
	createTextureImageFromFile(path_to_ao, aoImage, aoImageMemory);
	createTextureImageProcedural(proceduralImage, proceduralImageMemory);
	// END ----- createTextureImage from tutorial

	// START ----- createTextureImageView from tutorial
	createImageView(textureImage, textureImageView, VK_FORMAT_R8G8B8A8_SRGB);
	createImageView(aoImage, aoImageView, VK_FORMAT_R8G8B8A8_SRGB);
	createImageView(proceduralImage, proceduralImageView, VK_FORMAT_R32_SFLOAT);
	// END ----- createTextureImageView from tutorial

	// START ----- createTextureSampler from tutorial
	createTextureSampler(textureSampler);
	createTextureSampler(aoSampler);
	createTextureSampler(proceduralSampler);
	// END ----- createTextureSampler from tutorial

	// START ----- createDescriptorPool from tutorial

	// poolsizes vector
	std::vector<VkDescriptorPoolSize> poolSizes;

	VkDescriptorPoolSize textureSamplerLayoutPoolSize{};
	textureSamplerLayoutPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	textureSamplerLayoutPoolSize.descriptorCount = 1;
	poolSizes.push_back(textureSamplerLayoutPoolSize);

	VkDescriptorPoolSize aoSamplerLayoutPoolSize{};
	aoSamplerLayoutPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	aoSamplerLayoutPoolSize.descriptorCount = 1;
	poolSizes.push_back(aoSamplerLayoutPoolSize);

	VkDescriptorPoolSize proceduralSamplerLayoutPoolSize{};
	proceduralSamplerLayoutPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	proceduralSamplerLayoutPoolSize.descriptorCount = 1;
	poolSizes.push_back(proceduralSamplerLayoutPoolSize);

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = 1;

	if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}
	// END ----- createDescriptorPool from tutorial

	// START ----- createDescriptorSets from tutorial + slides ish
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &descriptorSetLayout;

	if (vkAllocateDescriptorSets(vklGetDevice(), &allocInfo, &ds) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor set!");
	}

	VkDescriptorImageInfo textureImageInfo{};
	textureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	textureImageInfo.imageView = textureImageView;
	textureImageInfo.sampler = textureSampler;

	VkDescriptorImageInfo aoImageInfo{};
	aoImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	aoImageInfo.imageView = aoImageView;
	aoImageInfo.sampler = aoSampler;

	VkDescriptorImageInfo proceduralImageInfo{};
	proceduralImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	proceduralImageInfo.imageView = proceduralImageView;
	proceduralImageInfo.sampler = proceduralSampler;

	// write descriptor set vector
	std::vector<VkWriteDescriptorSet> wds;

	VkWriteDescriptorSet textureWds = {}; // multiples permitted, also used for uniforms
	textureWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	textureWds.dstSet = ds;
	textureWds.dstBinding = 1;
	textureWds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	textureWds.descriptorCount = 1;
	textureWds.pImageInfo = &textureImageInfo;
	wds.push_back(textureWds);

	VkWriteDescriptorSet aoWds = {}; // multiples permitted, also used for uniforms
	aoWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	aoWds.dstSet = ds;
	aoWds.dstBinding = 2;
	aoWds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	aoWds.descriptorCount = 1;
	aoWds.pImageInfo = &aoImageInfo;
	wds.push_back(aoWds);

	VkWriteDescriptorSet procWds = {}; // multiples permitted, also used for uniforms
	procWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	procWds.dstSet = ds;
	procWds.dstBinding = 3;
	procWds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	procWds.descriptorCount = 1;
	procWds.pImageInfo = &proceduralImageInfo;
	wds.push_back(procWds);

	vkUpdateDescriptorSets(vklGetDevice(), 3, wds.data(), 0, nullptr);
	// END ----- createDescriptorSets from tutorial
}


// Cleanup buffers and pipeline created on the GPU
void objectDestroyBuffers() {
	auto device = vklGetDevice();
	vkDeviceWaitIdle( device );
	vklDestroyGraphicsPipeline(pipeline);
	vklDestroyHostCoherentBufferAndItsBackingMemory( mObjectVertexData );
	vklDestroyHostCoherentBufferAndItsBackingMemory( mObjectIndices );

	vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);
	vkDestroySampler(device, textureSampler, nullptr);
	vkDestroyImageView(device, textureImageView, nullptr);

	vkDestroyImage(device, aoImage, nullptr);
    vkFreeMemory(device, aoImageMemory, nullptr);
	vkDestroySampler(device, aoSampler, nullptr);
	vkDestroyImageView(device, aoImageView, nullptr);
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

	vklBindDescriptorSetToPipeline(ds, pipeline);
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

		// Vertex data at binding 0
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

		// START ----- createDescriptorSetLayout from tutorial

		// Bindings vector
		std::vector<VkDescriptorSetLayoutBinding> bindings;

		// Texture sampler layout at binding 1
		VkDescriptorSetLayoutBinding textureSamplerLayoutBinding{};
		textureSamplerLayoutBinding.binding = 1;
		textureSamplerLayoutBinding.descriptorCount = 1;
		textureSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		textureSamplerLayoutBinding.pImmutableSamplers = nullptr;
		textureSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		bindings.push_back(textureSamplerLayoutBinding);

		// Ambient occlusion sampler layout at binding 2
		VkDescriptorSetLayoutBinding aoSamplerLayoutBinding{};
		aoSamplerLayoutBinding.binding = 2;
		aoSamplerLayoutBinding.descriptorCount = 1;
		aoSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		aoSamplerLayoutBinding.pImmutableSamplers = nullptr;
		aoSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		bindings.push_back(aoSamplerLayoutBinding);

		// Procedural sampler layout at binding 3
		VkDescriptorSetLayoutBinding proceduralSamplerLayoutBinding{};
		proceduralSamplerLayoutBinding.binding = 3;
		proceduralSamplerLayoutBinding.descriptorCount = 1;
		proceduralSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		proceduralSamplerLayoutBinding.pImmutableSamplers = nullptr;
		proceduralSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		bindings.push_back(proceduralSamplerLayoutBinding);

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(vklGetDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}

		// Attach bindings to config
		config.descriptorLayout = bindings;

		// END ----- createDescriptorSetLayout from tutorial

	// Actually create the pipeline using the config
	pipeline = vklCreateGraphicsPipeline(config);
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

	pushConstants.proc = procedural_texturing;
	pushConstants.ao = ambient_occlusion;

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

uint32_t getMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    auto device = vklGetDevice();

	VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = getMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    auto device = vklGetDevice();

	VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = getMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(vklGetDevice(), &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
	VkQueue graphicsQueue;
	vkGetDeviceQueue(vklGetDevice(), selected_queue_family_index, 0, &graphicsQueue);

	vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(vklGetDevice(), commandPool, 1, &commandBuffer);
}

void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = {0, 0, 0};
	region.imageExtent = {
		width,
		height,
		1
	};

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	endSingleTimeCommands(commandBuffer);
}

// Format isn't used here??
void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else {
		throw std::invalid_argument("unsupported layout transition!");
	}

	vkCmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

	endSingleTimeCommands(commandBuffer);
}

void createCommandPool() {
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = selected_queue_family_index;

	if (vkCreateCommandPool(vklGetDevice(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics command pool!");
	}
}

void createImageView(VkImage& image, VkImageView &imageView, VkFormat format) {
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	if (vkCreateImageView(vklGetDevice(), &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image view!");
	}
}

void createTextureSampler(VkSampler& sampler) {
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(vk_physical_device, &properties);

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_FALSE;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	if (vkCreateSampler(vklGetDevice(), &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture sampler!");
	}
}

void createTextureImageFromFile(const char* path_to_tex, VkImage& image, VkDeviceMemory& imageMemory){
	VkDevice device = vklGetDevice();
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(path_to_tex, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	VkDeviceSize imageSize = texWidth * texHeight * 4;

	if (!pixels) {
		throw std::runtime_error("failed to load texture image!");
	}

	VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
	createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* imageData;
	vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &imageData);
		memcpy(imageData, pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(device, stagingBufferMemory);

	stbi_image_free(pixels);

	createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory);

	transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void createTextureImageProcedural(VkImage& image, VkDeviceMemory& imageMemory){
	VkDevice device = vklGetDevice();

	std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(1, 100);  // Adjust the range as needed

    // Create an array of length 256 and fill it with random integers
    int randomArray[256];
    for (int i = 0; i < 256; ++i) {
        randomArray[i] = dist(mt);
    }
	uint32_t procedralSideLength = 16;
	size_t imageSize = sizeof(randomArray[0]) * procedralSideLength * procedralSideLength;

	VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
	createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* imageData;
	vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &imageData);
		memcpy(imageData, randomArray, static_cast<size_t>(imageSize));
	vkUnmapMemory(device, stagingBufferMemory);

	createImage(procedralSideLength, procedralSideLength, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory);

	transitionImageLayout(image, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	  copyBufferToImage(stagingBuffer, image, procedralSideLength, procedralSideLength);
	transitionImageLayout(image, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	// createImageProcedural(procedralSideLength, VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory);

	// transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	// 	copyBufferToImage(stagingBuffer, image, static_cast<uint32_t>(procedralSideLength), static_cast<uint32_t>(procedralSideLength));
	// transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
