/*
 * Copyright 2023 University of Calgary, Visualization and Graphics Grpup
 */
#include <vulkan/vulkan.h>

void objectCreateGeometryAndBuffers();
void objectDestroyBuffers();
void objectDraw();

void objectDraw(VkPipeline pipeline);

VkBuffer objectGetVertexBuffer();
VkBuffer objectGetIndicesBuffer();
uint32_t objectGetNumIndices();

void objectCreatePipeline();
void objectUpdateConstants();