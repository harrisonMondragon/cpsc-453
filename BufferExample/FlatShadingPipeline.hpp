#pragma once

#include <vulkan/vulkan.h>
#include <VulkanLaunchpad.h>

namespace BufferExample
{
    class FlatShadingPipeline
    {
    public:

        struct Vertex
        {
            glm::vec3 position{};
        };

        struct PushConstants {
            glm::mat4 model{};
            glm::vec4 color{};
        };

        struct CameraData
        {
            glm::mat4 projection;
            glm::mat4 view;
        };

        explicit FlatShadingPipeline();

        ~FlatShadingPipeline();

        void BindPipeline() const;

        void PushConstant(PushConstants const & pushConstant) const;

        [[nodiscard]]
        VkPipeline GetPipeline() const;

        void UpdateCameraData(const CameraData& cameraData);

    private:

        VkPipeline _pipeline {};

        CameraData _cameraData{};

        VkBuffer _cameraBuffer{};

        VkDescriptorPool _descriptorPool{};

        VkDescriptorSet _descriptorSet{};

        VkPipelineLayout _pipelineLayout{};

    };
}
