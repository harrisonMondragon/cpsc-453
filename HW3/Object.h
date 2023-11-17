/*
 * Copyright 2023 University of Calgary, Visualization and Graphics Grpup
 */
#include <vulkan/vulkan.h>
#include "VulkanLaunchpad.h"
#include <string>

void objectCreateGeometryAndBuffers( const std::string& path_to_obj, const char* path_to_tex, const char* path_to_ao, GLFWwindow* window );
void objectDestroyBuffers();
void objectDraw();

void objectDraw(VkPipeline pipeline);

VkBuffer objectGetVertexBuffer();
VkBuffer objectGetIndicesBuffer();
uint32_t objectGetNumIndices();

void objectCreateCamera( GLFWwindow* window );
void objectCreatePipeline();
void objectUpdateConstants( GLFWwindow* window = nullptr );

// From tutorial
uint32_t getMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

VkCommandBuffer beginSingleTimeCommands();
void endSingleTimeCommands(VkCommandBuffer commandBuffer);

void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
void createCommandPool();
void createImageView(VkImage& image, VkImageView& imageView, VkFormat format);
void createTextureSampler(VkSampler& sampler);
void createTextureImageFromFile(const char* path_to_tex, VkImage& image, VkDeviceMemory& imageMemory);