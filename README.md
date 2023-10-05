# 453HW2Starter

This repository contains starter code that you can use for homework assignments in CPSC 453 (Introduction to Computer Graphics) at the University of Calgary.

It is based on the [VulkanLaunchpadStarter](https://github.com/cg-tuwien/VulkanLaunchpadStarter) repository from TU Wien, which in turn is based on [VulkanLaunchpad](https://github.com/cg-tuwien/VulkanLaunchpad). `VulkanLaunchpad` is a framework by TU Wien targeted at Vulkan beginners. It abstracts some of the hard and overly verbose parts of the Vulkan C API.

## Setup

1. Clone this repository using the ```https``` address and switch to the `HW2` branch. 
    ```
    git clone <https address>
    ```
2. In your project directory, create a new folder called build:
    ```
    mkdir build
    ```
3. Use cmake to generate the necessary files for compiling the project. Please review the [Setup Instructions](https://github.com/cg-tuwien/VulkanLaunchpad#setup-instructions) for `VulkanLaunchpadStarter`` to setup your IDE and build environment. For this step, you can either use cmake-gui or run the cmake command directly in the terminal:
    ```
    cd build
    cmake ..
    ```
4. For lab computers to setup vulkan sdk env use:
    ```
    source /home/share/gfx/vulkanSDK/1.3.261.1/setup-env.sh
    ```
5. You can now compile and run your project.

## Troubleshooting

- Make sure that Vulkan SDK is installed correctly. You can verify this by running the vkCube application. (Installed automatically when you install Vulkan SDK)

- For Linux, you need to install build-essentials.

- For windows, ensure that the c++ desktop development kit is installed inside visual studio. You can use developer command prompt in windows.

- For macOS, you must have the latest version of OS and Xcode installed.

## Available Examples and Skeletons

This branch consists of the following:
- `/HW2`:

  Starter code for HW2 which is a functioning example that renders a teapot using a graphics pipeline that supports vertex and fragment shaders, and push constants to send model, view and projection matrices to the shaders. It also adds a depth attachment so that models can be rendered with correct depth. Shaders demonstrate how vertex data (random per-vertex colors in this example) are interpolated inside triangles 
  between the vertex and fragment shader stages. 
  
  A very simple rotation control is provided. Use the LEFT and RIGHT arrows to rotate the teapot. 