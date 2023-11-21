Harrison Mondragon
30088805

CPSC 435: Fall 2023
HW3: Mappings

Building and Running Instructions:

 1. Build following the instructions in the 453VulkanStarter repository's README.md
    a. Choose [Release] when doing [CMake: Select Variant]
    b. Choose [HW3] when doing [CMAKE: Build Target]

 2. Ensure build finished successfully

 3. Navigate to [build] where there should be a HW3/HW3 executible

 4. Run HW1Starter.exe by doing [./HW3 obj/path.obj regular/colour/path.png ao/colour/path.png]
    a. To avoid issues, please use absolute path to the files you intend to use

 5. Model rotations:
    a. Use [Left] and [Right] keys for x rotations
    b. Use [Up] and [Down] keys for y rotations
    c. Use [Z] and [X] keys for z rotations
    d. Use [i] to toggle between intrinsic and extrinsic modes

 6. Model scaling:
    a. Use [=] for scaling up
    b. Use [-] for scaling down

 7. Ambient Occlusion:
    a. Use [A] to toggle ambient occlusion on and off

 8. Procedural Texturing:
    a. Use [P] to toggle between procedural texturing and image mapping texturing

 9. Shader hot reloading:
    a. The m value for procedural texturing is located in starter.frag if you wish to change it
    b. Light related constants are located in starter.frag if you wish to change any of them
    c. Saving the changes to the shader and pressing [F5] will hot reload the window and apply the changes

Note: As stated in the source code, I heavily referenced the following tutorial:
      https://vulkan-tutorial.com/Texture_mapping/Images