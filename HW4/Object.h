/*
 * Copyright 2023 University of Calgary, Visualization and Graphics Grpup
 */
#include <vulkan/vulkan.h>
#include "VulkanLaunchpad.h"
#include <string>

void objectCreateGeometryAndBuffers( GLFWwindow* window );
void objectDestroyBuffers();
void objectDraw();

void objectDraw(VkPipeline pipeline);

VkBuffer objectGetVertexBuffer();
VkBuffer objectGetIndicesBuffer();
uint32_t objectGetNumIndices();

void objectCreateCamera( GLFWwindow* window );
void objectCreatePipeline();
void objectUpdateConstants();

// texture management functions
void objectCreateTextureImage( const char* file );
void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
VkCommandBuffer beginSingleTimeCommands();
void endSingleTimeCommands(VkCommandBuffer commandBuffer);
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

void createTextureImageView();
void createTextureSampler();
void createDescriptorPool();
void createDescriptorSets();
