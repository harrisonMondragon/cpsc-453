/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 */
#include <vulkan/vulkan.h>

#include "FlatShadingPipeline.hpp"

namespace BufferExample
{
	// This is just an example!
	class TeapotRenderer
	{
	public:

		explicit TeapotRenderer();

		~TeapotRenderer();

		void BindVertexBuffer() const;
		void BindIndexBuffer() const;
		void DrawIndexed() const;

	private:

		uint32_t mNumTeapotIndices;
		VkBuffer mTeapotPositions;
		VkDeviceMemory mTeapotPositionsMemory;
		VkBuffer mTeapotIndices;
		VkDeviceMemory mTeapotIndicesMemory;
	};
}