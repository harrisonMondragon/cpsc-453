/**
 * Starter main file for HW2.
 *
 * CPSC 453 | Fall 2023 | University of Calgary
 *
 * @author Usman Alim
 */


// Include our framework and the Vulkan headers:
#include "VulkanLaunchpad.h"
#include "Camera.h"
#include <vulkan/vulkan.h>

// Include some local helper functions:
#include "VulkanHelpers.h"
#include "Object.h"

// Include functionality from the standard library:
#include <vector>
#include <unordered_map>
#include <limits>
#include <set>

// vulkan related variables that are globally accessible.

VkDevice vk_device = VK_NULL_HANDLE;
VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
VkSwapchainKHR vk_swapchain = VK_NULL_HANDLE;

/* ------------------------------------------------ */
// Some more little helpers directly declared here:
// (Definitions of functions below the main function)
/* ------------------------------------------------ */

/*!
 *	This callback function gets invoked by GLFW whenever a GLFW error occured.
 */
void errorCallbackFromGlfw(int error, const char* description);

/*!
 *	Function that is invoked by GLFW to handle key events like key presses or key releases.
 *	If the ESC key has been pressed, the window will be marked that it should close.
 */
void handleGlfwKeyCallback(GLFWwindow* glfw_window, int key, int scancode, int action, int mods);

/*!
 *	Functions needed to handle window resize.
 */
void handleResizeCallback(GLFWwindow* window, int width, int height);
VklSwapchainConfig GenerateSwapChainConfig();


/*!
 *	Function that can be used to query whether or not currently, i.e. NOW, a certain button
 *  is pressed down, or not.
 *  @param	glfw_key_code	One of the GLFW key codes.
 *                          I.e., use one of the defines that start with GLFW_KEY_*
 *  @return True if the given key is currently pressed down, false otherwise (i.e. released).
 */
bool isKeyDown(int glfw_key_code);

/*!
 *	Determine the Vulkan instance extensions that are required by GLFW and Vulkan Launchpad.
 *	Required extensions from both sources are combined into one single vector (i.e., in
 *	contiguous memory, so that they can easily be passed to:
 *  VkInstanceCreateInfo::enabledExtensionCount and to VkInstanceCreateInfo::ppEnabledExtensionNames.
 *	@return     A std::vector of const char* elements, containing all required instance extensions.
 *	@example    std::vector<const char*> extensions = getRequiredInstanceExtensions();
 *	            VkInstanceCreateInfo create_info    = {};
 *	            create_info.enabledExtensionCount   = extensions.size();
 *	            create_info.ppEnabledExtensionNames = extensions.data();
 */
std::vector<const char*> getRequiredInstanceExtensions();

/*!
 *	Based on the given physical device and the surface, select a queue family which supports both,
 *	graphics and presentation to the given surface. Return the INDEX of an appropriate queue family!
 *	@return		The index of a queue family which supports the required features shall be returned.
 */
uint32_t selectQueueFamilyIndex(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

static std::set<std::string> QuerySupportedLayers() {
	static std::set<std::string> supportedLayers{};
	static bool layersAreSet = false;

	if (layersAreSet == false)
	{
		uint32_t count;
		vkEnumerateInstanceLayerProperties(&count, nullptr); //get number of extensions

		std::vector<VkLayerProperties> layers(count);
		vkEnumerateInstanceLayerProperties(&count, layers.data()); //populate buffer

		for (auto const& layer : layers) {
			supportedLayers.insert(layer.layerName);
		}
		layersAreSet = true;
	}
	return supportedLayers;
}

static std::vector<char const*> FilterSupportedLayers(std::vector<char const*> const& layers)
{
	auto const supportedLayers = QuerySupportedLayers();
	std::vector<char const*> result{};
	for (auto const& layer : layers)
	{
		if (supportedLayers.find(layer) != supportedLayers.end()) {
			printf("Layer %s is supported by this device.\n", layer);
			result.emplace_back(layer);
		}
		else {
			printf("Layer %s is not supported by this device.\n", layer);
		}
	}
	return result;
}


// variables that are modified via keyboard controls
float scale = 1.0f;
float extrinsic_x = 0.0f;
float extrinsic_y = 0.0f;
float extrinsic_z = 0.0f;

/* ------------------------------------------------ */
// Main
/* ------------------------------------------------ */
int main(int argc, char* argv[])
{
	VKL_LOG(":::::: WELCOME TO VULKAN LAUNCHPAD ::::::");

	// Install a callback function, which gets invoked whenever a GLFW error occurred:
	glfwSetErrorCallback(errorCallbackFromGlfw);

	// Initialize GLFW:
	if (!glfwInit()) {
		VKL_EXIT_WITH_ERROR("Failed to init GLFW");
	}

	/* --------------------------------------------- */
	// Task 1.1: Create a Window with GLFW
	/* --------------------------------------------- */
	static constexpr int window_width  = 800;
	static constexpr int window_height = 800;
	static constexpr bool fullscreen = false;
	static constexpr char const * window_title = "CPSC 453: HW2 Starter";

	// Use a monitor if we'd like to open the window in fullscreen mode:
	GLFWmonitor* monitor = nullptr;
	if (fullscreen) {
		monitor = glfwGetPrimaryMonitor();
	}

	// Set some window settings before creating the window:
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No need to create a graphics context for Vulkan
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	// Get a valid window handle and assign to window:
	GLFWwindow* window = glfwCreateWindow(window_width, window_height, window_title, nullptr, nullptr);;

	if (!window) {
		VKL_LOG("If your program reaches this point, that means two things:");
		VKL_LOG("1) Project setup was successful. Everything is working fine.");
		VKL_LOG("2) You haven't implemented the first task, which is creating a window with GLFW.");
		VKL_EXIT_WITH_ERROR("No GLFW window created.");
	}
	VKL_LOG("Task 1.1 done.");

	// Set up a key callback via GLFW here to handle keyboard user input:
	glfwSetKeyCallback(window, handleGlfwKeyCallback);

	/* --------------------------------------------- */
	// Task 1.2: Create a Vulkan Instance
	/* --------------------------------------------- */
	VkInstance vk_instance = VK_NULL_HANDLE;

	// Describe some meta data about this application, and define which Vulkan API version is required:
	VkApplicationInfo application_info = {};                     // Zero-initialize every member
	application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // Set this struct instance's type
	application_info.pEngineName = "Vulkan Launchpad";           // Set some properties...
	application_info.engineVersion = VK_MAKE_API_VERSION(0, 2023, 1, 0);
	application_info.pApplicationName = "An Introduction to Vulkan";
	application_info.applicationVersion = VK_MAKE_API_VERSION(0, 2023, 1, 1);
	application_info.apiVersion = VK_API_VERSION_1_1;            // Your system needs to support this Vulkan API version.

	// We'll require some extensions (e.g., for presenting something on a window surface, and more):
	std::vector<const char*> required_extensions = getRequiredInstanceExtensions();

	// Layers enable additional functionality. We'd like to enable the standard validation layer, 
	// so that we get meaningful and descriptive error messages whenever we mess up something:
	std::vector<const char*> enabled_layers{};
	if (!hlpIsInstanceLayerSupported("VK_LAYER_KHRONOS_validation")) {
		VKL_WARNING("Validation layers are not supported!");
		//VKL_EXIT_WITH_ERROR("Validation layer \"VK_LAYER_KHRONOS_validation\" is not supported.");
	}
	else
	{
		VKL_LOG("Validation layer \"VK_LAYER_KHRONOS_validation\" is supported.");
		enabled_layers.emplace_back("VK_LAYER_KHRONOS_validation");
	}

	// Tie everything from above together in an instance of VkInstanceCreateInfo:
	VkInstanceCreateInfo instance_create_info = {}; // Zero-initialize every member
	instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; // Set this struct instance's type
	instance_create_info.pApplicationInfo = &application_info;
	// Hook in required_extensions using VkInstanceCreateInfo::enabledExtensionCount and VkInstanceCreateInfo::ppEnabledExtensionNames!
	// Hook in enabled_layers using VkInstanceCreateInfo::enabledLayerCount and VkInstanceCreateInfo::ppEnabledLayerNames!
#if __APPLE__
	required_extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	instance_create_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
	instance_create_info.enabledExtensionCount = required_extensions.size();
	instance_create_info.ppEnabledExtensionNames = required_extensions.data();
	instance_create_info.enabledLayerCount = enabled_layers.size();
	instance_create_info.ppEnabledLayerNames = enabled_layers.data();


	// Use vkCreateInstance to create a vulkan instance handle! Assign it to vk_instance!
	VkResult result = vkCreateInstance(&instance_create_info, nullptr, &vk_instance);
	VKL_CHECK_VULKAN_RESULT(result);

	if (!vk_instance) {
		VKL_EXIT_WITH_ERROR("No VkInstance created or handle not assigned.");
	}
	VKL_LOG("Task 1.2 done.");

	/* --------------------------------------------- */
	// Task 1.3: Create a Vulkan Window Surface
	/* --------------------------------------------- */
	vk_surface = VK_NULL_HANDLE;

	// Use glfwCreateWindowSurface to create a window surface! Assign its handle to vk_surface!
	result = glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface);
	VKL_CHECK_VULKAN_RESULT(result);

	if (!vk_surface) {
		VKL_EXIT_WITH_ERROR("No VkSurfaceKHR created or handle not assigned.");
	}
	VKL_LOG("Task 1.3 done.");

	/* --------------------------------------------- */
	// Task 1.4 Pick a Physical Device
	/* --------------------------------------------- */
	vk_physical_device = VK_NULL_HANDLE;

	// Use vkEnumeratePhysicalDevices get all the available physical device handles! 
	// Select one that is suitable using hlpSelectPhysicalDeviceIndex and assign it to vk_physical_device!

	uint32_t device_count = 0;
	vkEnumeratePhysicalDevices(vk_instance, &device_count, nullptr);
	std::vector<VkPhysicalDevice> physical_devices(device_count);
	result = vkEnumeratePhysicalDevices(vk_instance, &device_count, physical_devices.data());
	VKL_CHECK_VULKAN_RESULT(result);
	//std::cout << "Device Count: " << device_count << std::endl;
	vk_physical_device = physical_devices[hlpSelectPhysicalDeviceIndex( physical_devices, vk_surface)];

	if (!vk_physical_device) {
		VKL_EXIT_WITH_ERROR("No VkPhysicalDevice selected or handle not assigned.");
	}

	{
		// Print out some information about selected device
		VkPhysicalDeviceProperties device_properties = {};
		vkGetPhysicalDeviceProperties(vk_physical_device, &device_properties);
		VKL_LOG("Selected Device Name and Driver Version:");
		VKL_LOG(device_properties.deviceName);
		VKL_LOG(device_properties.driverVersion);
	}

	VKL_LOG("Task 1.4 done.");

	/* --------------------------------------------- */
	// Task 1.5: Select a Queue Family
	/* --------------------------------------------- */

	// Find a suitable queue family and assign its index to the following variable:
	// Hint: Use selectQueueFamilyIndex, but complete its implementation before!
	//uint32_t selected_queue_family_index = std::numeric_limits<uint32_t>::max();
	uint32_t selected_queue_family_index = selectQueueFamilyIndex(vk_physical_device, vk_surface);

	// Sanity check if we have selected a valid queue family index:
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &queue_family_count, nullptr);
	if (selected_queue_family_index >= queue_family_count) {
		VKL_EXIT_WITH_ERROR("Invalid queue family index selected.");
	}
	VKL_LOG("Task 1.5 done.");

	/* --------------------------------------------- */
	// Task 1.6: Create a Logical Device and Get Queue
	/* --------------------------------------------- */
	vk_device = VK_NULL_HANDLE;
	VkQueue  vk_queue  = VK_NULL_HANDLE;

	constexpr float queue_priority = 1.0f;

	VkDeviceQueueCreateInfo queue_create_info = {};
	queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queue_create_info.queueFamilyIndex = selected_queue_family_index;
	queue_create_info.queueCount = 1;
	queue_create_info.pQueuePriorities = &queue_priority;

	// Create an instance of VkDeviceCreateInfo and use it to create one queue!
	// - Hook in queue_create_info at the right place!
	// - Use VkDeviceCreateInfo::enabledExtensionCount and VkDeviceCreateInfo::ppEnabledExtensionNames
	//   to enable the VK_KHR_SWAPCHAIN_EXTENSION_NAME device extension!
	// - The other parameters are not required (ensure that they are zero-initialized).
	//   Finally, use vkCreateDevice to create the device and assign its handle to vk_device!

	std::vector<const char*> device_extensions = {
    	VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	// For Macos compatibility
	auto const supportedExtension = FilterSupportedLayers(std::vector<char const *>{
		VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
	});
	device_extensions.insert(device_extensions.end(), supportedExtension.begin(), supportedExtension.end());

	VkDeviceCreateInfo device_create_info = {};
	device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	device_create_info.pQueueCreateInfos = &queue_create_info;
	device_create_info.queueCreateInfoCount = 1;
	device_create_info.enabledExtensionCount = device_extensions.size();
	device_create_info.ppEnabledExtensionNames = device_extensions.data();

	VkPhysicalDeviceFeatures features = {};
	vkGetPhysicalDeviceFeatures(vk_physical_device, &features);
	features.fillModeNonSolid = VK_TRUE;
	features.wideLines = VK_TRUE;

	device_create_info.pEnabledFeatures = &features;
	result = vkCreateDevice(vk_physical_device, &device_create_info, nullptr, &vk_device);
	VKL_CHECK_VULKAN_RESULT(result);

	if (!vk_device) {
		VKL_EXIT_WITH_ERROR("No VkDevice created or handle not assigned.");
	}

	// After device creation, use vkGetDeviceQueue to get the one and only created queue!
	// Assign its handle to vk_queue!
	vkGetDeviceQueue(vk_device, selected_queue_family_index, 0, &vk_queue);
	if (!vk_queue) {
		VKL_EXIT_WITH_ERROR("No VkQueue selected or handle not assigned.");
	}


	VKL_LOG("Task 1.6 done.");

	/* --------------------------------------------- */
	// Task 1.7: Create Swap Chain
	/* --------------------------------------------- */

	// swapchain config moved to a separate function
	// so that the swapchain can be recreated on
	// window resize.
	auto const swapchain_config = GenerateSwapChainConfig();

	VKL_LOG("Task 1.7 done.");

	/* --------------------------------------------- */
	// Task 1.8: Initialize Vulkan Launchpad
	/* --------------------------------------------- */

	// Init the framework:
	if (!vklInitFramework(vk_instance, vk_surface, vk_physical_device, vk_device, vk_queue, swapchain_config)) {
		VKL_EXIT_WITH_ERROR("Failed to init Vulkan Launchpad");
	}
	VKL_LOG("Task 1.8 done.");

 	// At this point, we can register window resize callback
	glfwSetWindowSizeCallback(window, handleResizeCallback);


	// Now create initial geometry and pass it to the GPU
	objectCreateGeometryAndBuffers(argv[1], window);

	/* --------------------------------------------- */
	// Task 1.9:  Implement the Render Loop
	/* --------------------------------------------- */
	while (!glfwWindowShouldClose(window)) {
		vklWaitForNextSwapchainImage();
    	vklStartRecordingCommands();

    	// Your commands here
		objectDraw();

    	vklEndRecordingCommands();
    	vklPresentCurrentSwapchainImage();
		glfwPollEvents(); // Handle user input
	}

	// Wait for all GPU work to finish before cleaning up:
	vkDeviceWaitIdle(vk_device);

	/* --------------------------------------------- */
	// Task 1.10: Cleanup
	/* --------------------------------------------- */
	objectDestroyBuffers();
	vklDestroyFramework();

	return EXIT_SUCCESS;
}

/* ------------------------------------------------ */
// Definitions of little helpers defined above main:
/* ------------------------------------------------ */

void errorCallbackFromGlfw(int error, const char* description) {
	std::cout << "GLFW error " << error << ": " << description << std::endl;
}

std::unordered_map<int, bool> g_isGlfwKeyDown;

void handleGlfwKeyCallback(GLFWwindow* glfw_window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS) {
		g_isGlfwKeyDown[key] = true;
	}
	if (action == GLFW_RELEASE) {
		g_isGlfwKeyDown[key] = false;
	}

	// Scale commands are [enter] for bigger and [backspace] for smaller
	if( key == GLFW_KEY_ENTER && action == GLFW_REPEAT ) {
		scale += 0.01f;
	}
	if( key == GLFW_KEY_BACKSPACE && action == GLFW_REPEAT ) {
		scale -= 0.01f;
	}

	// Extrinsic x commands are [Q] and [W]
	if( key == GLFW_KEY_Q && action == GLFW_REPEAT ) {
		extrinsic_x += 0.1f;
	}
	if( key == GLFW_KEY_W && action == GLFW_REPEAT ) {
		extrinsic_x -= 0.1f;
	}

	// Extrinsic y commands are [A] and [S]
	if( key == GLFW_KEY_A && action == GLFW_REPEAT ) {
		extrinsic_y -= 0.1f;
	}
	if( key == GLFW_KEY_S && action == GLFW_REPEAT ) {
		extrinsic_y += 0.1f;
	}

	// Extrinsic z commands are [Z] and [X]
	if( key == GLFW_KEY_Z && action == GLFW_REPEAT ) {
		extrinsic_z += 0.1f;
	}
	if( key == GLFW_KEY_X && action == GLFW_REPEAT ) {
		extrinsic_z -= 0.1f;
	}

	// We mark the window that it should close if ESC is pressed:
	if (action == GLFW_RELEASE && key == GLFW_KEY_ESCAPE) {
		glfwSetWindowShouldClose(glfw_window, true);
	}
}

void handleResizeCallback(GLFWwindow* window, int width, int height)
{
	vklNotifyResized(GenerateSwapChainConfig());
}

VklSwapchainConfig GenerateSwapChainConfig()
{
	VkSurfaceCapabilitiesKHR surface_capabilities = hlpGetPhysicalDeviceSurfaceCapabilities(vk_physical_device, vk_surface);

	// Build the swapchain config struct:
	// Provide values for:
	// - VkSwapchainCreateInfoKHR::queueFamilyIndexCount
	// - VkSwapchainCreateInfoKHR::pQueueFamilyIndices
	// - VkSwapchainCreateInfoKHR::imageFormat
	// - VkSwapchainCreateInfoKHR::imageColorSpace
	// - VkSwapchainCreateInfoKHR::imageExtent
	// - VkSwapchainCreateInfoKHR::presentMode
	auto oldSwapChain = vk_swapchain;

	VkSwapchainCreateInfoKHR swapchain_create_info = {};
	swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchain_create_info.surface = vk_surface;
	swapchain_create_info.minImageCount = surface_capabilities.minImageCount;
	swapchain_create_info.imageArrayLayers = 1u;
	swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swapchain_create_info.preTransform = surface_capabilities.currentTransform;
	swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchain_create_info.clipped = VK_TRUE;
	swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swapchain_create_info.oldSwapchain = oldSwapChain;

	VkSurfaceFormatKHR surface_format = hlpGetSurfaceImageFormat(vk_physical_device, vk_surface);

	swapchain_create_info.queueFamilyIndexCount = 0;
	swapchain_create_info.pQueueFamilyIndices = nullptr;
	swapchain_create_info.imageFormat = surface_format.format;
	swapchain_create_info.imageColorSpace = surface_format.colorSpace;
	swapchain_create_info.imageExtent = surface_capabilities.currentExtent;
	swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;

	// Create the swapchain using vkCreateSwapchainKHR and assign its handle to vk_swapchain!

	auto result = vkCreateSwapchainKHR(vk_device, &swapchain_create_info, nullptr, &vk_swapchain);
	VKL_CHECK_VULKAN_RESULT(result);

	if (!vk_swapchain) {
		VKL_EXIT_WITH_ERROR("No VkSwapchainKHR created or handle not assigned.");
	}

	// Create a vector of VkImages with enough memory for all the swap chain's images:
	std::vector<VkImage> swap_chain_images(surface_capabilities.minImageCount);
	// Use vkGetSwapchainImagesKHR to write VkImage handles into swap_chain_images.data()!

	result = vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &surface_capabilities.minImageCount,
		swap_chain_images.data());
	VKL_CHECK_VULKAN_RESULT(result);

	if (swap_chain_images.empty()) {
		VKL_EXIT_WITH_ERROR("Swap chain images not retrieved.");
	}

	// Gather swapchain config as required by the framework:
	VklSwapchainConfig swapchain_config = {};
	swapchain_config.imageExtent = swapchain_create_info.imageExtent;
	swapchain_config.swapchainHandle = vk_swapchain;
	for (VkImage vk_image : swap_chain_images) {
		VklSwapchainFramebufferComposition framebufferData;
		// Fill the data for the color attachment:
		//  - VklSwapchainImageDetails::imageHandle
		//  - VklSwapchainImageDetails::imageFormat
		//  - VklSwapchainImageDetails::imageUsage
		//  - VklSwapchainImageDetails::clearValue
		framebufferData.colorAttachmentImageDetails.imageHandle = vk_image;
		framebufferData.colorAttachmentImageDetails.imageFormat = surface_format.format;
		framebufferData.colorAttachmentImageDetails.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		VkClearValue clearValue;
		clearValue.color = { 0.2f, 0.2f, 0.2f, 1.0f };
		framebufferData.colorAttachmentImageDetails.clearValue = clearValue;

		// Let's also add a depth attachment for HW2
		VkImage depthImage = vklCreateDeviceLocalImageWithBackingMemory(
			vk_physical_device,
			vk_device,
			swapchain_config.imageExtent.width,
		 	swapchain_config.imageExtent.height,
			VK_FORMAT_D32_SFLOAT,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT );
		framebufferData.depthAttachmentImageDetails.imageHandle = depthImage;
		framebufferData.depthAttachmentImageDetails.imageFormat = VK_FORMAT_D32_SFLOAT;
		framebufferData.depthAttachmentImageDetails.imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		VkClearValue depthClear;
		depthClear.depthStencil.depth = 1.f;
		framebufferData.depthAttachmentImageDetails.clearValue  = depthClear;

		// Add it to the vector:
		swapchain_config.swapchainImages.push_back(framebufferData);
	}

	return swapchain_config;
}




bool isKeyDown(int glfw_key_code)
{
	return g_isGlfwKeyDown[glfw_key_code];
}

std::vector<const char*> getRequiredInstanceExtensions()
{
	// Get extensions which GLFW requires:
	uint32_t num_glfw_extensions;
	const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&num_glfw_extensions);

	// Get extensions which Vulkan Launchpad requires:
	uint32_t num_vkl_extensions;
	const char** vkl_extensions = vklGetRequiredInstanceExtensions(&num_vkl_extensions);

	// Merge both arrays of extensions:
	std::vector<const char*> all_required_extensions(glfw_extensions, glfw_extensions + num_glfw_extensions);
	all_required_extensions.insert(all_required_extensions.end(), vkl_extensions, vkl_extensions + num_vkl_extensions);

	// Perform a sanity check if all the extensions are really supported by Vulkan on 
	// this system (if they are not, we have a problem):
	for (auto ext : all_required_extensions) {
		if (!hlpIsInstanceExtensionSupported(ext)) {
			VKL_EXIT_WITH_ERROR("Required extension \"" << ext << "\" is not supported");
		}
		VKL_LOG("Extension \"" << ext << "\" is supported");
	}

	return all_required_extensions;
}

uint32_t selectQueueFamilyIndex(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
	// Get the number of different queue families for the given physical device:
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

	// Get the queue families' data:
	std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

	// Find a suitable queue family index and return it!

	// This is mostly done in VulkanHelpers.cpp. Borrowing from there.

	for (uint32_t queue_family_index = 0u; queue_family_index < queue_family_count; ++queue_family_index) {
			// If this physical device supports a queue family which supports both, graphics and presentation
			//  => select this physical device
			if ((queue_families[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				// This queue supports graphics! Let's see if it also supports presentation:
				VkBool32 presentation_supported;
				vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index, surface, &presentation_supported);

				if (VK_TRUE == presentation_supported) {
					// We've found a suitable queue
					return queue_family_index;
				}
			}
		}


	VKL_EXIT_WITH_ERROR("Unable to find a suitable queue family that supports graphics and presentation on the same queue.");
}
