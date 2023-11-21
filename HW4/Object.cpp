/*
 * Copyright 2023 University of Calgary, Visualization and Graphics Group
 */

#include "Object.h"
#include <VulkanLaunchpad.h>
#include <vulkan/vulkan.hpp>
#include <glm/gtx/normal.hpp>
#include "Camera.h"
#include "Path.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

extern vk::PhysicalDevice mPhysicalDevice;
extern vk::Device mDevice;
extern vk::UniqueCommandPool mCommandPool;
extern vk::Queue mQueue;

// The following variables are related to texture setup
// --- Begin texture setup variables
VkImage textureImage;
VkDeviceMemory textureImageMemory;

VkImageView textureImageView;
VkSampler textureSampler;

VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorPool descriptorPool;
std::vector<VkDescriptorSet> descriptorSets;
// --- End texture setup variables


// buffers that will live on the GPU.
// No geometry retained on the CPU, all data sent to the GPU.

uint32_t mNumObjectIndices;
VkBuffer mObjectVertexData;
VkBuffer mObjectIndices;

// A pipeline that can be used for HW4
VkPipeline pipeline;

// Struct to pack object vertex data
// Only positions for the HW4 Starter
struct Vertex {
	glm::vec3 position;
}; 

// Send model, view and projection information as push constants
// which are packed in this struct
struct ObjectPushConstants {
    glm::mat4 model; // placeholder, currently unused
    glm::mat4 invView;
    glm::vec4 proj; // {near, far, aspect, fov} rather than the matrix for ray-tracing
};

// Info that is updated interactively via push constants
ObjectPushConstants pushConstants;

// camera related variables
extern VklCamera* camera;
extern float aspect;
const float near = 0.01f;
const float far = 1000.0f;
const float fov = 45.0f;


// Setup geometry and textures for ray tracing.
void objectCreateGeometryAndBuffers( GLFWwindow* window ) 
{
	// Four vertices and six indices for two triangles
	// that will cover the full screen. 
	std::cout << "Packing vertices and normals..." << std::endl;
	std::vector<Vertex> vData(4);
	vData[0].position = glm::vec3{2.0f,2.0f, 0.5f};
	vData[1].position = glm::vec3{2.0f,-2.0f, 0.5f};
	vData[2].position = glm::vec3{-2.0f,-2.0f, 0.5f};
	vData[3].position = glm::vec3{-2.0f,2.0f, 0.5f};
	
	std::vector<uint32_t> indices {0, 1, 3, 1, 2, 3};
	
	mNumObjectIndices = static_cast<uint32_t>(indices.size());
	const auto device = vklGetDevice();
	auto dispatchLoader = vk::DispatchLoaderStatic();

	std:: cout << "Sending packed vertex-normal data to GPU..." << std::endl;

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
			indices.data(), sizeof(indices[0]) * indices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	}

	// Now Create the camera and pipeline
	std::cout << "Now Creating Camera and Pipeline " << std::endl;
	
	// All texture related setup
	auto path = shared::Path::Instantiate();

	auto const imagePath = shared::Path::Instance->Get("textures/sample.png");
	objectCreateTextureImage( imagePath.c_str() );
	createTextureImageView();
	createTextureSampler();
	createDescriptorPool();
	createDescriptorSets();

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

	
	// cleanup textures
	vkDestroySampler(device, textureSampler, nullptr);
	vkDestroyImageView(device, textureImageView, nullptr);
	vkDestroyImage(device, textureImage, nullptr);
	vkFreeMemory(device, textureImageMemory, nullptr);

	vkDestroyDescriptorPool(device, descriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);


	// Cleanup Camera
	vklDestroyCamera( camera);
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

	vklBindDescriptorSetToPipeline(descriptorSets[0], pipeline );

	cb.bindVertexBuffers(0u, { vk::Buffer{ objectGetVertexBuffer() } }, { vk::DeviceSize{ 0 } });
	cb.bindIndexBuffer(vk::Buffer{ objectGetIndicesBuffer() }, vk::DeviceSize{ 0 }, vk::IndexType::eUint32);
	
	// Update things that need to be updated per draw call
	
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

	// ------------------------------
	// Pipeline creation
	// ------------------------------

	auto const vertShaderPath = shared::Path::Instance->Get("shaders/starter.vert");
	auto const fragShaderPath = shared::Path::Instance->Get("shaders/starter.frag");


	VklGraphicsPipelineConfig config{};
		config.enableAlphaBlending = false;
		// path to shaders may need to be modified depending on the location
		// of the executable
		config.vertexShaderPath = vertShaderPath.c_str();
		config.fragmentShaderPath = fragShaderPath.c_str();
		
		// Can set polygonDrawMode to VK_POLYGON_MODE_LINE for wireframe rendering
		// if supported by GPU
		config.polygonDrawMode = VK_POLYGON_MODE_FILL;
		//config.triangleCullingMode = VK_CULL_MODE_BACK_BIT;

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
		//config.inputAttributeDescriptions.emplace_back(VkVertexInputAttributeDescription{
			//.location = static_cast<uint32_t>(config.inputAttributeDescriptions.size()),
	//		.location = 1,
	//		.binding = 0,
	//		.format = VK_FORMAT_R32G32B32_SFLOAT,
	//		.offset = offsetof(Vertex, normal),
	//	});

	config.descriptorLayout.emplace_back(
		VkDescriptorSetLayoutBinding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.pImmutableSamplers = nullptr,
		}
	);
            
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
void objectUpdateConstants() {
	pushConstants.model = glm::mat4(1.0);
	//pushConstants.invView = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 0.0f, 3.0f));
	pushConstants.invView = glm::inverse(vklGetCameraViewMatrix(camera));	
	pushConstants.proj = glm::vec4{near, far, aspect, fov};
}

// Function to create camera
void objectCreateCamera( GLFWwindow* window ) {
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	// assume a default right handed camera looking down -z.
	
	aspect = static_cast<float>(width) / static_cast<float>(height);

	camera = vklCreateCamera( window, 
		vklCreatePerspectiveProjectionMatrix(glm::radians(fov), 
		aspect, near, far) );
	camera->mPosition = glm::vec3(0.0f, 0.0f, 3.0f);
}

// The code that follows is based on the texture setup code 
// provided by:
// https://vulkan-tutorial.com/Texture_mapping/Images

void objectCreateTextureImage( const char* file ) {
	int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(file, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

	VkBuffer stagingBuffer = 
		vklCreateHostCoherentBufferAndUploadData( 
			pixels, 
			static_cast<size_t>(imageSize), 
	 		VK_BUFFER_USAGE_TRANSFER_SRC_BIT );
	
	stbi_image_free(pixels);

	createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, 
		VK_IMAGE_TILING_OPTIMAL, 
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
		textureImage, textureImageMemory);

	transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));	
	transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vklDestroyHostCoherentBufferAndItsBackingMemory(stagingBuffer);
}

void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
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

    if (vkCreateImage(mDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(mDevice, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(mDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(mDevice, image, imageMemory, 0);
}


uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
}

VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = mCommandPool.get();
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(mDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(mQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(mQueue);

    vkFreeCommandBuffers(mDevice, mCommandPool.get(), 1, &commandBuffer);
}

void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}

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

	vkCmdCopyBufferToImage(
    	commandBuffer,
    	buffer,
    	image,
    	VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    	1,
    	&region
	);

    endSingleTimeCommands(commandBuffer);
}

void createTextureImageView() {
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = textureImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	if (vkCreateImageView(mDevice, &viewInfo, nullptr, &textureImageView) != VK_SUCCESS) {
    	throw std::runtime_error("failed to create texture image view!");
	}
}

void createTextureSampler() {
	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_FALSE; // disabled since we have not talked about anisotropic filtering
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE; // Use normalized texture coordinates (in the [0,1] range)

	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

	// We'll not worry about mipmapping
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = 0.0f;

	if (vkCreateSampler(mDevice, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes(1);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(1);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(1);

    if (vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void createDescriptorSets() {
	VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

	std::vector<VkDescriptorSetLayout> layouts(1, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(1);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(1);
    if (vkAllocateDescriptorSets(mDevice, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

	VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = textureImageView;
    imageInfo.sampler = textureSampler;

	// A vector of descriptor sets so multiple
	// descriptors can be updated

	std::vector<VkWriteDescriptorSet> dwrite(1);
	dwrite[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	dwrite[0].dstSet = descriptorSets[0];
	dwrite[0].dstBinding = 0;
	dwrite[0].dstArrayElement = 0;
	dwrite[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	dwrite[0].descriptorCount = 1;
	dwrite[0].pImageInfo = &imageInfo;

	vkUpdateDescriptorSets(mDevice, 
		static_cast<uint32_t>(dwrite.size()), 
		dwrite.data(), 0, nullptr);
}


