Harrison Mondragon
30088805

CPSC 435: Fall 2023
HW3: Virtual Orrery and Realtime Raytracing

Building and Running Instructions:

 1. Build following the instructions in the 453VulkanStarter repository's README.md
    a. Choose [Release] when doing [CMake: Select Variant]
    b. Choose [HW4] when doing [CMAKE: Build Target]

 2. Ensure build finished successfully

 3. Navigate to [build] where there should be a HW4 executible

 4. Run HW4.exe by doing ./HW4

 5. Model rotations:
    a. Use [Left] and [Right] keys to move the celestial bodies in lock step

 6. Camera controls:
    a. Scroll to zoom camera
    b. Left click to rotate camera
    c. Right click to pan camera

 7. Shader hot reloading:
    a. The rotations are all in correct relation to eachother. If you update moon_axial_period located in the
       main function of starter.frag, the rest of the rotations will update proportionally.
    b. Light related constants are located in starter.frag if you wish to change any of them
    c. Saving the changes to the shader and pressing [F5] will hot reload the window and apply the changes

Requirements completed:
 1. All the base assignment functional requiremets are completed. These include:
    a. Spheres and Texture Coordinates
    b. Transformations and Control
    c. Shading

 2. Only some of the bonus requirements are completed. These include:
    a. Shadow Rays
    b. Orbital Tilt
    c. Axial tilt was NOT completed
