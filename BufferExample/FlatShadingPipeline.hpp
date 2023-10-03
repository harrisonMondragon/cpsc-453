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

        [[nodiscard]]
        VkPipeline GetPipeline() const;

    private:

        VkPipeline _pipeline {};

    };
}
