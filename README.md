# 453VulkanStarter

This repository contains starter code that you can use for homework assignments in CPSC 453 (Introduction to Computer Graphics) at the University of Calgary.

It is based on the [VulkanLaunchpadStarter](https://github.com/cg-tuwien/VulkanLaunchpadStarter) repository from TU Wien, which in turn is based on [VulkanLaunchpad](https://github.com/cg-tuwien/VulkanLaunchpad). `VulkanLaunchpad` is a framework by TU Wien targeted at Vulkan beginners. It abstracts some of the hard and overly verbose parts of the Vulkan C API.

## Setup

1. Clone this repository using the ```https``` address.
    ```
    git clone <https address>
    ```
2. In your project directory, create a new folder called build:
    ```
    mkdir build
    ```
3. Use the cmake to generate the necessary files for compiling the project. Please review the [Setup Instructions](https://github.com/cg-tuwien/VulkanLaunchpad#setup-instructions) for `VulkanLaunchpadStarter` to setup your IDE and build environment. For this step, you can either use cmake-gui or run the cmake command directly in the terminal:
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

- Make sure that Vulkan sdk is installed correctly. You can verify this by running the vkCube application. (Installed automatically when you install Vulkan sdk)

- For Linux, you need to install build-essentials.

- For windows, ensure that the c++ desktop development kit is installed inside visual studio. You can use developer command prompt in windows.

- For macOS, you must have the latest version of OS and Xcode installed.

## Available Examples and Skeletons

This repository consists of the following:

- `/HW3`:

  Code that you can use as starter code for HW3. The code implements most of the required functionality from HW2. HW3 relies on models that come with normals, so the code does not compute normals.
  Additionally, given an OBJ model with texture coordinates (specified as a command-line argument), it shows how to augment Phong shading with a checkerboard texture procedurally generated in the fragment shader.

  Rotation and Scaling controls are as follows:
  - Use the `RIGHT` and `LEFT` cursor keys to rotate about the $x$ axis. 
  - Use the `UP` and `DOWN` cursor keys to rotate about the $y$ axis.
  - Use the `x` and `z` keys to rotate about the $z$ axis.
  - Press `i` to toggle between intrinsic and extrinsic rotation modes.
  - Use the `=` and `-` keys to scale up and down. 

  <br>
 
  Shader hot reloading is also supported.
  - Press `F5` to hot-reload shaders.
  
  <br>
 
  Several OBJ models are provided under `/HW3/models` for testing purposes.

- `ImGui` has been integrated into this branch. See `BufferExample` on how to use it if you wish you incorporate it in `HW3`. 