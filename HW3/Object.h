/*
 * Copyright 2023 University of Calgary, Visualization and Graphics Grpup
 */
#include <vulkan/vulkan.h>
#include "VulkanLaunchpad.h"
#include <string>

void objectCreateGeometryAndBuffers( const std::string& path_to_obj, GLFWwindow* window );
void objectDestroyBuffers();
void objectDraw();

void objectDraw(VkPipeline pipeline);

VkBuffer objectGetVertexBuffer();
VkBuffer objectGetIndicesBuffer();
uint32_t objectGetNumIndices();

void objectCreateCamera( GLFWwindow* window );
void objectCreatePipeline();
void objectUpdateConstants();