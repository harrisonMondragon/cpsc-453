#include "LineGeometry.h"

#include <VulkanLaunchpad.h>
#include <vulkan/vulkan.hpp>

#include "LinePipeline.hpp"

// Maximum buffer size to create buffer -> 4^10 = 1024*1024
// NOTE: This might be too big for some machines
#define MAX_BUFFER_SIZE 1048576

// Some variables to perform calculations
uint32_t hilbertN;
double mNumLineVertices;
double totalSideSquares;
double fractalSideSquares;
double stretch;
double translation;
std::vector<glm::vec3> newVertices;

// A vector to organize geometry on the GPU.
// Also need a buffer on the GPU to copy the geometry data to.
std::vector<glm::vec3> vertices;
VkBuffer mLineVertices;

// If using indexed drawing, the following can be used.
// uint32_t mNumLineIndices;
// std::vector<glm::vec3> indices;
// VkBuffer mLineIndices;

// tx and ty keep track of wrap around
extern float tx;
extern float ty;

// dx and dy are used as incremental translations
extern float dx;
extern float dy;

std::shared_ptr<MyApp::LinePipeline> linePipeline{};

/**
 * Create line geometry that is to be drawn.
 * Allocate a buffer on the GPU that is large enough to hold the
 * geometry data. Copy date over to the GPU.
 *
 * First create hilbert curve for N = 1, then increase to N = 3 for the
 * curve that shows on init.
 */
void lineInitGeometryAndBuffers() {
  VKL_LOG("lineInitGeometryAndBuffers called");

  // Plot hilbertN = 1
  hilbertN = 1;

  vertices = {
    glm::vec3(1.0f, -1.0f, 1.0f),
    glm::vec3(1.0f, 1.0f, 1.0f),
    glm::vec3(-1.0f, 1.0f, 1.0f),
    glm::vec3(-1.0f, -1.0f, 1.0f),
  };

  // Create vertex buffer on GPU and copy data into it.
  mLineVertices = vklCreateHostCoherentBufferAndUploadData(
      vertices.data(), sizeof(vertices[0]) * MAX_BUFFER_SIZE,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

  // N = 2
  increaseHilbertN();

  // N = 3, inital phase
  increaseHilbertN();

  linePipeline = std::make_shared<MyApp::LinePipeline>();
}

/**
 * Update geomemtry by applying transformations. Copy updated
 * geometry to the GPU.
 */
void lineUpdateGeometryAndBuffers() {

  // Any transformations to the vertices can be applied as
  // follows:
  // (glm stores matrices in column-major order)
  // A simple translation for now that can be controlled
  // via the cursor keys.

  auto T = glm::mat3(1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  dx, dy, 1.0f);

  for (size_t i = 0; i < vertices.size(); i++) {
    vertices[i] = T * vertices[i];
  }

  vklCopyDataIntoHostCoherentBuffer(mLineVertices, vertices.data(),
    sizeof(vertices[0]) * vertices.size());
}

/**
 * Cleanup buffers created on the GPU that are no longer
 * needed.
 */
void lineDestroyBuffers() {
  auto device = vklGetDevice();
  vklDestroyHostCoherentBufferAndItsBackingMemory( mLineVertices );
  linePipeline.reset();
	// vklDestroyHostCoherentBufferAndItsBackingMemory( mLineIndices );
}

/**
 * Call this from within a render loop to draw.
 * To draw line strips using the basic graphics pipeline provided by
 * VulkanLaunchpad, please make the following change in the basic graphics
 * pipeline before calling this function:
 *
 * Set the appropriate primitive topology in the body of
 * vklCreateGraphicsPipeline() function.
 * For example, to draw line strips use the following:
 *
 *   ...setTopology(vk::PrimitiveTopology::eLineStrip)
 */
void lineDraw() {
  if (!vklFrameworkInitialized()) {
    VKL_EXIT_WITH_ERROR(
        "Framework not initialized. Ensure to invoke vklFrameworkInitialized "
        "beforehand!");
  }

  const vk::CommandBuffer& cb = vklGetCurrentCommandBuffer();
  auto currentSwapChainImageIndex = vklGetCurrentSwapChainImageIndex();
  assert(currentSwapChainImageIndex < vklGetNumFramebuffers());
  assert(currentSwapChainImageIndex < vklGetNumClearValues());

  cb.bindPipeline( vk::PipelineBindPoint::eGraphics, linePipeline->GetPipeline());

  cb.bindVertexBuffers(0u, {mLineVertices}, {vk::DeviceSize{0}});
  //cb.bindIndexBuffer(vk::Buffer{ mLineIndices }, vk::DeviceSize{ 0 }, vk::IndexType::eUint32);
  cb.draw(mNumLineVertices, 1u, 0u, 0u);
  //cb.drawIndexed(mNumLineIndices, 1u, 0u, 0, 0u);
}

VkBuffer lineGetVerticesBuffer() {
  return static_cast<VkBuffer>(mLineVertices);
}

/**
 * Update hilbert geomemtry to increase N by applying transformations. Copy updated
 * geometry to the GPU.
 */
void increaseHilbertN(){
  VKL_LOG("increaseHilbertN called on N = " << hilbertN);

  // Do not perform transformations if N is already at 10
  if(hilbertN == 10){
    VKL_LOG("Cannot increase N because 10 is the highest possible N, returning ...");
    return;
  }

  // Increase N and recalculate variables used for math
  hilbertN ++;
  crunchNumbers();

  // Calculate stretch and translation values (negatives will be used as needed)
  stretch = fractalSideSquares/totalSideSquares;
  translation = (fractalSideSquares+1)/totalSideSquares;

  // Bottom left corner
  // Rotate by 90 degrees clockwise, scale down, translate to bottom left
  auto T1 = glm::mat3(0.0f,-stretch,0.0f, stretch,0.0f,0.0f, -translation,-translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices/4; i++) {
    newVertices[i] = T1 * vertices[i];
  }

  // Reverse original vector to get the right order of points
  std::reverse(vertices.begin(),vertices.end());

  // Top left corner
  // Scale down, translate to top left
  auto T2 = glm::mat3(stretch,0.0f,0.0f, 0.0f,stretch,0.0f, -translation,translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices/4; i++) {
    newVertices[i + mNumLineVertices/4] = T2 * vertices[i];
  }

  // Top right corner
  // Scale down, translate to top right
  auto T3 = glm::mat3(stretch,0.0f,0.0f, 0.0f,stretch,0.0f, translation,translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices/4; i++) {
    newVertices[i + 2*mNumLineVertices/4] = T3 * vertices[i];
  }

  // Revert back to original vector to get the right order of points
  std::reverse(vertices.begin(),vertices.end());

  // Bottom right corner
  // Rotate by 90 degrees counter clockwise, scale down, translate to bottom right
  auto T4 = glm::mat3(0.0f,stretch,0.0f, -stretch,0.0f,0.0f, translation,-translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices/4; i++) {
    newVertices[i + 3*mNumLineVertices/4] = T4 * vertices[i];
  }

  // Reverse final vector so this function can be used for any N
  std::reverse(newVertices.begin(),newVertices.end());

  vertices = newVertices;

  // Update buffer
  vklCopyDataIntoHostCoherentBuffer(mLineVertices, vertices.data(),
    sizeof(vertices[0]) * vertices.size());
}

/**
 * Update hilbert geomemtry to decrease N by applying transformations. Copy updated
 * geometry to the GPU.
 */
void decreaseHilbertN(){
  VKL_LOG("decreaseHilbertN called on N = " << hilbertN);

  // Do not perform transformation if N is already at 1
  if(hilbertN == 1){
    VKL_LOG("Cannot decrease N because 1 is the lowest possible N, returning ...");
    return;
  }

  // Calculate stretch before updating N
  stretch = totalSideSquares/(fractalSideSquares);

  // Decrease N and recalculate variables used for math
  hilbertN --;
  crunchNumbers();

  // Calculate translation after updating N
  translation = (totalSideSquares+1)/totalSideSquares;

  /*
  Remember vertices was reversed at the end of increaseHilbertN for the
  algorithm to work, so this uses the bottom right fractal.
  Rotation is applied about the origin, so the bottom right fractal ends
  up in the bottom left, in the correct orientation
  */

  // Rotate by 90 degrees clockwise, scale up, translate to fill screen
  auto T = glm::mat3(0.0f,-stretch,0.0f, stretch,0.0f,0.0f, translation,translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices; i++) {
    newVertices[i] = T * vertices[i];
  }

  // Reverse final vector so this function can be used for any N
  std::reverse(newVertices.begin(),newVertices.end());

  vertices = newVertices;

  // Update buffer
  vklCopyDataIntoHostCoherentBuffer(mLineVertices, vertices.data(),
    sizeof(vertices[0]) * vertices.size());
}

/**
 * Crunch numbers needed in transformation calculations and update utility variables.
 */
void crunchNumbers(){
  mNumLineVertices = glm::pow(4.0f, hilbertN);
  totalSideSquares = glm::sqrt(mNumLineVertices) - 1.0f;
  fractalSideSquares = glm::floor(totalSideSquares/2);
  newVertices = std::vector<glm::vec3>(mNumLineVertices);
}