#include "LineGeometry.h"

#include <VulkanLaunchpad.h>
#include <vulkan/vulkan.hpp>

#include "LinePipeline.hpp"

// A vector to organize geometry on the GPU.
// Also need a buffer on the GPU to copy the geometry data to.

uint32_t hilbertN;
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
 * Explain hilbert curve ...
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

  // N = 2
  increaseHilbertN();

  // N = 3, inital phase
  increaseHilbertN();

  increaseHilbertN();

  linePipeline = std::make_shared<MyApp::LinePipeline>();

  // Create vertex buffer on GPU and copy data into it.
  mLineVertices = vklCreateHostCoherentBufferAndUploadData(
      vertices.data(), sizeof(vertices[0]) * vertices.size(),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
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
  cb.draw(glm::pow(4.0f, hilbertN), 1u, 0u, 0u);
  //cb.drawIndexed(mNumLineIndices, 1u, 0u, 0, 0u);
}

VkBuffer lineGetVerticesBuffer() {
  return static_cast<VkBuffer>(mLineVertices);
}

void increaseHilbertN(){
  VKL_LOG("increaseHilbertN called on N = " << hilbertN);

  //Crunch numbers
  hilbertN ++;
  uint32_t mNumLineVertices = glm::pow(4.0f, hilbertN);
  float sideSquares = glm::sqrt(mNumLineVertices) - 1.0f;
  float stretchNumerator = glm::floor(sideSquares/2);
  float stretch = stretchNumerator/sideSquares;
  float translation = (stretchNumerator+1)/sideSquares;
  std::vector<glm::vec3> finalVertices = std::vector<glm::vec3>(mNumLineVertices);

  //Bottom left corner
  //Rotate by 90 degrees clockwise, scale, translate to bottom left
  auto T1 = glm::mat3(0.0f,-stretch,0.0f, stretch,0.0f,0.0f, -translation,-translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices/4; i++) {
    finalVertices[i] = T1 * vertices[i];
  }

  //Reverse original vector to get the right order of points
  std::reverse(vertices.begin(),vertices.end());

  //Top left corner
  //Scale, translate to top left
  auto T2 = glm::mat3(stretch,0.0f,0.0f, 0.0f,stretch,0.0f, -translation,translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices/4; i++) {
    finalVertices[i + mNumLineVertices/4] = T2 * vertices[i];
  }

  //Top right corner
  //Scale, translate to top right
  auto T3 = glm::mat3(stretch,0.0f,0.0f, 0.0f,stretch,0.0f, translation,translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices/4; i++) {
    finalVertices[i + 2*mNumLineVertices/4] = T3 * vertices[i];
  }

  //Revert back to original vector to get the right order of points
  std::reverse(vertices.begin(),vertices.end());

  //Bottom right corner
  //Rotate by 90 degrees counter clockwise, scale, translate to bottom right
  auto T4 = glm::mat3(0.0f,stretch,0.0f, -stretch,0.0f,0.0f, translation,-translation,1.0f);
  for (size_t i = 0; i < mNumLineVertices/4; i++) {
    finalVertices[i + 3*mNumLineVertices/4] = T4 * vertices[i];
  }

  //Reverse final vector so this function can be used for any N
  std::reverse(finalVertices.begin(),finalVertices.end());

  vertices = finalVertices;
}
