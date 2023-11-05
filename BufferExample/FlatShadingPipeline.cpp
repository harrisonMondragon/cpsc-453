#include "FlatShadingPipeline.hpp"

#include "Path.hpp"

#include  <filesystem>

namespace BufferExample
{

	using namespace shared;

	//-------------------------------------------------------------------------------------------------

	FlatShadingPipeline::FlatShadingPipeline()
	{
		auto const vertexShaderPath = Path::Instance->Get("shaders/buffer-example/FlatShading.vert");
		auto const fragmentShaderPath = Path::Instance->Get("shaders/buffer-example/FlatShading.frag");

		VklGraphicsPipelineConfig config{};
		config.enableAlphaBlending = false;
		config.vertexShaderPath = vertexShaderPath.c_str();
		config.fragmentShaderPath = fragmentShaderPath.c_str();
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


		_cameraBuffer = vklCreateHostCoherentBufferAndUploadData(&_cameraData, sizeof(_cameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		{// Creating descriptor pool
			VkDescriptorPoolCreateInfo poolInfo = {};
			poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.maxSets = 1; // You only need one descriptor set
			poolInfo.poolSizeCount = 1;

			VkDescriptorPoolSize poolSize = {};
			poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // Example descriptor type
			poolSize.descriptorCount = 1; // You only need one of this type
			poolInfo.pPoolSizes = &poolSize;

			auto const result = vkCreateDescriptorPool(vklGetDevice(), &poolInfo, nullptr, &_descriptorPool);
			assert(result == VK_SUCCESS);
		}

		{// Create descriptor set
			auto const descriptorSetLayout = vklGetDescriptorLayout(_pipeline);

			VkDescriptorSetAllocateInfo allocInfo = {};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = _descriptorPool;
			allocInfo.descriptorSetCount = 1; // Allocate one descriptor set
			allocInfo.pSetLayouts = &descriptorSetLayout; // Provide the descriptor set layout

			auto result = (vkAllocateDescriptorSets(vklGetDevice(), &allocInfo, &_descriptorSet));
			assert(result == VK_SUCCESS);
		}
		{// Update descriptor set
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = _cameraBuffer;
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(CameraData);

			VkWriteDescriptorSet descriptorWrite = {};
			descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet = _descriptorSet;
			descriptorWrite.dstBinding = 0; // Binding index in your descriptor set layout
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pBufferInfo = &bufferInfo;

			vkUpdateDescriptorSets(vklGetDevice(), 1, &descriptorWrite, 0, nullptr);
		}

		_pipelineLayout = vklGetLayoutForPipeline(_pipeline);
	}

	//-------------------------------------------------------------------------------------------------

	FlatShadingPipeline::~FlatShadingPipeline()
	{
		
		vklDestroyGraphicsPipeline(_pipeline);

		vkDestroyDescriptorPool(vklGetDevice(), _descriptorPool, nullptr);

		vklDestroyHostCoherentBufferAndItsBackingMemory(_cameraBuffer);
	}

	//-------------------------------------------------------------------------------------------------

	void FlatShadingPipeline::BindPipeline() const
	{
		auto const& cb = vklGetCurrentCommandBuffer();
		vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline);
		vkCmdBindDescriptorSets(
			cb,
			VK_PIPELINE_BIND_POINT_GRAPHICS, // or VK_PIPELINE_BIND_POINT_COMPUTE for compute shaders
			_pipelineLayout,
			0, // First set to bind
			1, // Number of descriptor sets to bind
			&_descriptorSet,
			0, // Dynamic offsets (usually set to 0)
			nullptr // Array of dynamic offsets
		);
	}

	//-------------------------------------------------------------------------------------------------

	void FlatShadingPipeline::PushConstant(PushConstants const& pushConstant) const
	{
		vklSetPushConstants(
			_pipeline, 
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 
			&pushConstant, 
			sizeof(pushConstant)
		);
	}

	//-------------------------------------------------------------------------------------------------

	VkPipeline FlatShadingPipeline::GetPipeline() const
	{
		return _pipeline;
	}

	//-------------------------------------------------------------------------------------------------

	void FlatShadingPipeline::UpdateCameraData(const CameraData& cameraData)
	{
		vklCopyDataIntoHostCoherentBuffer(_cameraBuffer, &cameraData, sizeof(cameraData));
	}

	//-------------------------------------------------------------------------------------------------

}
