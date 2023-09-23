#include <vulkan/vulkan.h>

void lineInitGeometryAndBuffers();
void lineUpdateGeometryAndBuffers();
void lineDestroyBuffers();
void lineDraw();
void increaseHilbertN();
void decreaseHilbertN();
void crunchNumbers();

VkBuffer lineGetVerticesBuffer();