#include "FlatShadingPipeline.hpp"

#include  <filesystem>

namespace BufferExample
{

	//-------------------------------------------------------------------------------------------------

	FlatShadingPipeline::FlatShadingPipeline()
	{
		auto path = std::filesystem::path(".");
		auto abPath = std::filesystem::absolute(path);

		VklGraphicsPipelineConfig config{};
		config.enableAlphaBlending = false;
		config.vertexShaderPath = "assets/shaders/FlatShading.vert";
		config.fragmentShaderPath = "assets/shaders/FlatShading.frag";
		config.triangleCullingMode = VK_CULL_MODE_NONE;

		// Position
		config.inputAttributeDescriptions.emplace_back(VkVertexInputAttributeDescription{
			.location = static_cast<uint32_t>(config.inputAttributeDescriptions.size()),
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		});

		config.vertexInputBuffers.emplace_back(VkVertexInputBindingDescription{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		});
            
		// ModelViewProjection
		VkDescriptorSetLayoutBinding modelViewProjectionBinding{
			.binding = static_cast<uint32_t>(config.descriptorLayout.size()),
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		};
		config.descriptorLayout.emplace_back(modelViewProjectionBinding);

		config.pushConstantRanges.emplace_back(VkPushConstantRange{
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(PushConstants),
		});
            
		_pipeline = vklCreateGraphicsPipeline(config);
	}

	//-------------------------------------------------------------------------------------------------

	FlatShadingPipeline::~FlatShadingPipeline()
	{
		vklDestroyGraphicsPipeline(_pipeline);
	}

	//-------------------------------------------------------------------------------------------------

	VkPipeline FlatShadingPipeline::GetPipeline() const
	{
		return _pipeline;
	}

	//-------------------------------------------------------------------------------------------------

}
