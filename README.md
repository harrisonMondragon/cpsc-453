# 453VulkanStarter

This repository contains starter code that you can use for homework assignments in CPSC 453 (Introduction to Computer Graphics) at the University of Calgary.

It is based on the [VulkanLaunchpadStarter](https://github.com/cg-tuwien/VulkanLaunchpadStarter) repository from TU Wien, which in turn is based on [VulkanLaunchpad](https://github.com/cg-tuwien/VulkanLaunchpad). `VulkanLaunchpad` is a framework by TU Wien targeted at Vulkan beginners. It abstracts some of the hard and overly verbose parts of the Vulkan C API.

## Setup

1. Clone this repository.
2. `VulkanLaunchpadStarter` and its dependencies are added to this repository as submodules. Please update submodules by issuing the following command:
    ```bash
    git submodule update --init --recursive
    ```  
3. The code in this repository is tested with the following commit hash for `VulkanLaunchpadStarter`:
    - `63d91d5855e5ceaed3babee7bec4fe28f5c968d1`
   
   If needed, roll back the submodule to the above commit by issuing the following command in the `VulkanLaunchpadStarter` directory:
   ```bash
    git checkout <commit hash>
   ``` 
3. Compile via CMake. Please review the [Setup Instructions](https://github.com/cg-tuwien/VulkanLaunchpad#setup-instructions) for `VulkanLaunchpadStarter`` to setup your IDE and build environment.
4. To setup vulkan sdk env use 
```
source /home/share/gfx/vulkanSDK/1.3.261.1/setup-env.sh
```

## Available Examples and Skeletons

This repository consists of the following:
- `/Demo`:

  A fully functioning example that renders a teapot using the basic graphics pipeline provided by `VulkanLaunchpad`. It is just a completion of the TO DO tasks left out in the example provided by `VulkanLaunchpadStarter`.
