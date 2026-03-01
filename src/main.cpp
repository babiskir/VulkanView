// ============================================================================
//  Standard library headers
// ============================================================================
#include <algorithm>    // std::ranges algorithms (find_if, any_of, all_of, none_of)
#include <cassert>      // assert() — runtime sanity checks (removed in release builds)
#include <cstdlib>      // EXIT_SUCCESS / EXIT_FAILURE
#include <cstring>      // strcmp — comparing C-string extension / layer names
#include <fstream>      // std::ifstream — reading SPIR-V shader binaries from disk
#include <iostream>     // std::cerr — printing validation-layer messages
#include <limits>       // std::numeric_limits (used implicitly by Vulkan-Hpp)
#include <memory>       // Smart pointers (unique_ptr, shared_ptr)
#include <stdexcept>    // std::runtime_error — fatal Vulkan errors
#include <vector>       // std::vector — dynamic arrays used everywhere

// ============================================================================
//  Vulkan-Hpp RAII header
//
//  Vulkan-Hpp is the official C++ wrapper around the raw C Vulkan API.
//  The RAII variant (vulkan_raii.hpp) destroys Vulkan objects automatically
//  when they go out of scope — no manual vkDestroy*() calls needed.
//
//  The __INTELLISENSE__ guard: VS Code's IntelliSense engine can't handle
//  C++20 module imports yet, so we fall back to #include for the IDE.
//
//  TIP: VULKAN_HPP_NO_STRUCT_CONSTRUCTORS is set in CMake so we can use
//  C++20 designated initialisers:  vk::Foo{ .bar = 42 }
// ============================================================================
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

// ============================================================================
//  GLFW — cross-platform window & input library
//
//  GLFW_INCLUDE_VULKAN makes GLFW pull in <vulkan/vulkan.h> so that
//  glfwCreateWindowSurface() and glfwGetRequiredInstanceExtensions() compile.
//
//  TIP: Always include GLFW *after* the Vulkan headers.
// ============================================================================
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// ============================================================================
//  Application constants
// ============================================================================
constexpr uint32_t WIDTH                = 800;   // Initial window width  (pixels)
constexpr uint32_t HEIGHT               = 600;   // Initial window height (pixels)
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;     // How many frames the CPU can
                                                  // prepare ahead of the GPU.
                                                  // 2 = classic double-buffering.

// ============================================================================
//  Validation layers
//
//  Validation layers intercept every Vulkan call and check for misuse:
//  wrong parameters, missing barriers, resource leaks, etc.
//  They are only active in debug builds (enableValidationLayers == true).
//
//  VK_LAYER_KHRONOS_validation is the single "uber-layer" that activates
//  every individual check the Khronos group provides.
//
//  TIP: Always develop with validation layers ON. Turn them off only for
//       release builds — they add measurable CPU overhead.
// ============================================================================
const std::vector<char const *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;   // Release → no overhead
#else
constexpr bool enableValidationLayers = true;    // Debug   → full checking
#endif

// ============================================================================
//  HelloTriangleApplication
//
//  One class encapsulates the entire Vulkan lifecycle.  The pattern is:
//    run()  →  initWindow()  →  initVulkan()  →  mainLoop()  →  cleanup()
//
//  Every Vulkan object is stored as a vk::raii:: handle.  RAII handles
//  destroy themselves when the class is destructed — in reverse declaration
//  order, which mirrors the "destroy in reverse creation order" rule.
//
//  TIP: If you add new Vulkan handles, put them *after* the objects they
//       depend on so C++ destroys them first.
// ============================================================================
class HelloTriangleApplication
{
  public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	// ── Window ──────────────────────────────────────────────────────────────
	GLFWwindow *window = nullptr;                    // Opaque OS window handle

	// ── Core Vulkan objects (created once) ──────────────────────────────────
	vk::raii::Context                context;         // Bootstraps the Vulkan dynamic loader
	vk::raii::Instance               instance       = nullptr; // App ↔ Vulkan loader connection
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr; // Validation-layer callback hook
	vk::raii::SurfaceKHR             surface        = nullptr; // Window → Vulkan presentation link
	vk::raii::PhysicalDevice         physicalDevice = nullptr; // The GPU we selected
	vk::raii::Device                 device         = nullptr; // Logical device (our "connection" to the GPU)
	uint32_t                         queueIndex     = ~0u;     // Index of the graphics+present queue family
	vk::raii::Queue                  queue          = nullptr; // The queue we submit commands to

	// ── Swap chain ──────────────────────────────────────────────────────────
	// The swap chain is a ring of images the GPU renders into while the
	// display is showing a previously rendered image.  We store:
	//   - the chain itself
	//   - its raw VkImage handles (owned by the chain, NOT by us)
	//   - one ImageView per image (so shaders / rendering can access them)
	//   - the format and extent so other code can reference them
	vk::raii::SwapchainKHR           swapChain      = nullptr;
	std::vector<vk::Image>           swapChainImages;
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;
	vk::Extent2D                     swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;

	// ── Graphics pipeline ──────────────────────────────────────────────────
	vk::raii::PipelineLayout pipelineLayout   = nullptr; // Describes push-constants / descriptor sets (empty for now)
	vk::raii::Pipeline       graphicsPipeline = nullptr; // The compiled pipeline (shaders + fixed-function state)

	// ── Command recording ──────────────────────────────────────────────────
	vk::raii::CommandPool                commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;        // One per frame-in-flight

	// ── Synchronisation ────────────────────────────────────────────────────
	// Vulkan is fully asynchronous — the CPU and GPU run in parallel.
	// Semaphores synchronise GPU ↔ GPU work; fences synchronise CPU ↔ GPU.
	//
	//  presentCompleteSemaphores[frame]  — GPU signals when the swap-chain
	//                                      image is ready to be written.
	//  renderFinishedSemaphores[image]   — GPU signals when rendering into
	//                                      that image is done.
	//  inFlightFences[frame]             — CPU waits on this before reusing
	//                                      the same frame's command buffer.
	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence>     inFlightFences;
	uint32_t                         frameIndex = 0;  // Cycles 0 → MAX_FRAMES_IN_FLIGHT-1

	bool framebufferResized = false;   // Set by GLFW callback when the window is resized

	// ── Device extensions we require ────────────────────────────────────────
	// VK_KHR_swapchain lets the device present images to a window surface.
	std::vector<const char *> requiredDeviceExtension = {
	    vk::KHRSwapchainExtensionName};

	// ─────────────────────────────────────────────────────────────────────
	//  initWindow()
	//
	//  Creates a plain OS window with GLFW.
	//  GLFW_CLIENT_API = GLFW_NO_API → no OpenGL context (Vulkan manages its own).
	//  GLFW_RESIZABLE  = GLFW_TRUE   → we handle resize by recreating the swap chain.
	//
	//  TIP: glfwSetWindowUserPointer stashes a pointer to "this" inside the
	//       GLFW window so our static resize callback can reach the app object.
	// ─────────────────────────────────────────────────────────────────────
	void initWindow()
	{
		glfwInit();   // Must be called before any other GLFW function

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Vulkan, not OpenGL
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);     // Allow resizing

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

		// Store a back-pointer so the static callback can access our members
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	// Static callback invoked by GLFW when the framebuffer (drawable area) is resized.
	// We set a flag; the actual swap-chain recreation happens in drawFrame().
	static void framebufferResizeCallback(GLFWwindow *window, int /*width*/, int /*height*/)
	{
		auto app                = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	// ─────────────────────────────────────────────────────────────────────
	//  initVulkan()
	//
	//  Calls every Vulkan setup step in the correct dependency order.
	//  Each step creates one or a few Vulkan objects that later steps need.
	//
	//  TIP: If you ever need to add a new resource, slot it in here in the
	//       right order and remember: destroy in *reverse* order.
	// ─────────────────────────────────────────────────────────────────────
	void initVulkan()
	{
		createInstance();        // 1.  VkInstance — our handle to the Vulkan loader
		setupDebugMessenger();   // 2.  Hook validation-layer messages (debug only)
		createSurface();         // 3.  VkSurfaceKHR — links the window to Vulkan
		pickPhysicalDevice();    // 4.  Choose a GPU
		createLogicalDevice();   // 5.  VkDevice — logical connection to that GPU
		createSwapChain();       // 6.  Ring of presentable images
		createImageViews();      // 7.  One ImageView per swap-chain image
		createGraphicsPipeline();// 8.  Shaders + fixed-function state
		createCommandPool();     // 9.  Allocator for command buffers
		createCommandBuffers();  // 10. One command buffer per frame-in-flight
		createSyncObjects();     // 11. Semaphores & fences for frame pacing
	}

	// ─────────────────────────────────────────────────────────────────────
	//  mainLoop()
	//
	//  Classic game-loop: poll OS events, then draw a frame.
	//  glfwPollEvents() processes keyboard/mouse/resize without blocking.
	//  device.waitIdle() at the end makes sure the GPU finishes all work
	//  before we start tearing down resources in cleanup().
	// ─────────────────────────────────────────────────────────────────────
	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();   // Process keyboard / mouse / resize events
			drawFrame();        // Record and submit GPU commands for one frame
		}

		device.waitIdle();      // Wait for all GPU work before cleanup
	}

	// ─────────────────────────────────────────────────────────────────────
	//  cleanupSwapChain()
	//
	//  Destroys only the swap-chain and its image views so they can be
	//  recreated with a new size.  Called from recreateSwapChain().
	//
	//  TIP: We don't destroy the pipeline or sync objects here because
	//       they don't depend on the swap-chain dimensions.
	// ─────────────────────────────────────────────────────────────────────
	void cleanupSwapChain()
	{
		swapChainImageViews.clear();   // Destroy all image views (RAII)
		swapChain = nullptr;           // Destroy the swap chain itself
	}

	// ─────────────────────────────────────────────────────────────────────
	//  cleanup()
	//
	//  Only GLFW needs manual teardown — all Vulkan RAII handles destroy
	//  themselves when the HelloTriangleApplication object is destructed.
	// ─────────────────────────────────────────────────────────────────────
	void cleanup()
	{
		glfwDestroyWindow(window);   // Destroy the OS window
		glfwTerminate();             // Shut down GLFW
	}

	// ─────────────────────────────────────────────────────────────────────
	//  recreateSwapChain()
	//
	//  Called when the window is resized or the swap chain becomes stale.
	//
	//  If the framebuffer size is (0,0) — e.g. window minimised — we
	//  block on glfwWaitEvents() until the user restores the window.
	//
	//  device.waitIdle() ensures no in-flight GPU work references the
	//  old swap chain before we destroy it.
	//
	//  TIP: Only destroy what actually depends on the swap-chain size;
	//       the pipeline layout and most sync objects survive intact.
	// ─────────────────────────────────────────────────────────────────────
	void recreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)          // Minimised? Wait.
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		device.waitIdle();   // GPU must be idle before we destroy anything

		cleanupSwapChain();
		createSwapChain();
		createImageViews();
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createInstance()
	//
	//  The VkInstance is our application's handle into Vulkan.  It is the
	//  very first Vulkan object we create and the last one destroyed.
	//
	//  Steps:
	//    1. Describe the app (name, engine, target API version).
	//    2. Enable validation layers (debug only).
	//    3. Gather required instance extensions (from GLFW + debug utils).
	//    4. Verify all layers & extensions exist.
	//    5. Create the instance.
	//
	//  TIP: vk::ApiVersion13 targets Vulkan 1.3.  We need 1.3+ for
	//       dynamicRendering and synchronization2 (core in 1.3).
	// ─────────────────────────────────────────────────────────────────────
	void createInstance()
	{
		// ApplicationInfo is informational — drivers and profiling tools
		// (RenderDoc, NVIDIA Nsight) can display these strings.
		constexpr vk::ApplicationInfo appInfo{.pApplicationName   = "Hello Triangle",
		                                      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		                                      .pEngineName        = "No Engine",
		                                      .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
		                                      .apiVersion         = vk::ApiVersion13};

		// Get the required layers
		std::vector<char const *> requiredLayers;
		if (enableValidationLayers)
		{
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		// Check if the required layers are supported by the Vulkan implementation.
		auto layerProperties    = context.enumerateInstanceLayerProperties();
		auto unsupportedLayerIt = std::ranges::find_if(requiredLayers,
		                                               [&layerProperties](auto const &requiredLayer) {
			                                               return std::ranges::none_of(layerProperties,
			                                                                           [requiredLayer](auto const &layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
		                                               });
		if (unsupportedLayerIt != requiredLayers.end())
		{
			throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
		}

		// Get the required extensions.
		auto requiredExtensions = getRequiredInstanceExtensions();

		// Check if the required extensions are supported by the Vulkan implementation.
		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		auto unsupportedPropertyIt =
		    std::ranges::find_if(requiredExtensions,
		                         [&extensionProperties](auto const &requiredExtension) {
			                         return std::ranges::none_of(extensionProperties,
			                                                     [requiredExtension](auto const &extensionProperty) { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; });
		                         });
		if (unsupportedPropertyIt != requiredExtensions.end())
		{
			throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIt));
		}

		vk::InstanceCreateInfo createInfo{.pApplicationInfo        = &appInfo,
		                                  .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
		                                  .ppEnabledLayerNames     = requiredLayers.data(),
		                                  .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
		                                  .ppEnabledExtensionNames = requiredExtensions.data()};
		instance = vk::raii::Instance(context, createInfo);
	}

	// ─────────────────────────────────────────────────────────────────────
	//  setupDebugMessenger()
	//
	//  Hooks into VK_EXT_debug_utils so validation-layer messages reach
	//  our debugCallback() function.
	//
	//  Severity levels: eVerbose | eWarning | eError
	//  Message types:   eGeneral | ePerformance | eValidation
	//
	//  TIP: Change the severity flags to include eInfo if you want more
	//       detail, but expect a LOT of output.
	// ─────────────────────────────────────────────────────────────────────
	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;   // Nothing to do in release builds

		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
		    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

		vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
		    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral     |
		    vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		    vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

		vk::DebugUtilsMessengerCreateInfoEXT createInfo{
		    .messageSeverity = severityFlags,
		    .messageType     = messageTypeFlags,
		    .pfnUserCallback = &debugCallback   // Our static callback (see bottom of class)
		};
		debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo);
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createSurface()
	//
	//  A VkSurfaceKHR ties Vulkan to the OS window.  GLFW handles the
	//  platform-specific mess (Xlib, Wayland, Win32, etc.) behind a
	//  single call: glfwCreateWindowSurface().
	//
	//  The function returns a raw C VkSurfaceKHR, which we wrap in a
	//  vk::raii::SurfaceKHR so it is destroyed automatically.
	//
	//  TIP: *instance dereferences the RAII handle to get the raw
	//       VkInstance that GLFW's C API expects.
	// ─────────────────────────────────────────────────────────────────────
	void createSurface()
	{
		VkSurfaceKHR _surface;
		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		{
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	// ─────────────────────────────────────────────────────────────────────
	//  pickPhysicalDevice()
	//
	//  Picks the first GPU that supports everything we need:
	//    1. Vulkan 1.3 API version
	//    2. A graphics-capable queue family
	//    3. All our required device extensions (VK_KHR_swapchain)
	//    4. Specific features:
	//         · shaderDrawParameters  — gl_DrawID in shaders
	//         · synchronization2      — modern barrier API (Vulkan 1.3)
	//         · dynamicRendering      — skip VkRenderPass boilerplate
	//         · extendedDynamicState  — set cull-mode, etc. at draw time
	//
	//  TIP: On a multi-GPU system, the order of enumeratePhysicalDevices()
	//       isn't guaranteed.  To always pick a discrete GPU, sort them
	//       by vk::PhysicalDeviceType::eDiscreteGpu first.
	// ─────────────────────────────────────────────────────────────────────
	void pickPhysicalDevice()
	{
		std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
		const auto                            devIter = std::ranges::find_if(
            devices,
            [&](auto const &device) {
                // Check if the device supports the Vulkan 1.3 API version
                bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

                // Check if any of the queue families support graphics operations
                auto queueFamilies = device.getQueueFamilyProperties();
                bool supportsGraphics =
                    std::ranges::any_of(queueFamilies, [](auto const &qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

                // Check if all required device extensions are available
                auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
                bool supportsAllRequiredExtensions =
                    std::ranges::all_of(requiredDeviceExtension,
			                                                       [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
                                            return std::ranges::any_of(availableDeviceExtensions,
				                                                                                  [requiredDeviceExtension](auto const &availableDeviceExtension) { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
                                        });

                auto features                 = device.template getFeatures2<vk::PhysicalDeviceFeatures2,
			                                                                                            vk::PhysicalDeviceVulkan11Features,
			                                                                                            vk::PhysicalDeviceVulkan13Features,
			                                                                                            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
                bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                                features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
                                                features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

                return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
            });
		if (devIter != devices.end())
		{
			physicalDevice = *devIter;
		}
		else
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createLogicalDevice()
	//
	//  A VkDevice is a logical connection to a physical GPU.  We:
	//    1. Find a queue family that supports *both* graphics and present.
	//    2. Enable the features we checked in pickPhysicalDevice().
	//    3. Create the device with one queue.
	//
	//  The "feature chain" uses vk::StructureChain — Vulkan-Hpp's type-safe
	//  wrapper around pNext chains.  Each struct in the chain is linked
	//  automatically; Vulkan reads the full chain to enable features.
	//
	//  TIP: queuePriority (0.0 – 1.0) influences GPU scheduling between
	//       queues.  With a single queue it doesn't matter much.
	// ─────────────────────────────────────────────────────────────────────
	void createLogicalDevice()
	{
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		// Find the first queue family that supports both graphics AND present
		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
		{
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
			    physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
			{
				// found a queue family that supports both graphics and present
				queueIndex = qfpIndex;
				break;
			}
		}
		if (queueIndex == ~0)
		{
			throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
		}

		// query for Vulkan 1.3 features
		vk::StructureChain<vk::PhysicalDeviceFeatures2,
		                   vk::PhysicalDeviceVulkan11Features,
		                   vk::PhysicalDeviceVulkan13Features,
		                   vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
		    featureChain = {
		        {},                                                          // vk::PhysicalDeviceFeatures2
		        {.shaderDrawParameters = true},                              // vk::PhysicalDeviceVulkan11Features
		        {.synchronization2 = true, .dynamicRendering = true},        // vk::PhysicalDeviceVulkan13Features
		        {.extendedDynamicState = true}                               // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
		    };

		// create a Device
		float                     queuePriority = 0.5f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
		vk::DeviceCreateInfo      deviceCreateInfo{.pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		                                           .queueCreateInfoCount    = 1,
		                                           .pQueueCreateInfos       = &deviceQueueCreateInfo,
		                                           .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtension.size()),
		                                           .ppEnabledExtensionNames = requiredDeviceExtension.data()};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		queue  = vk::raii::Queue(device, queueIndex, 0);
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createSwapChain()
	//
	//  The swap chain is a queue of images the GPU renders into while the
	//  display is showing a previously completed image.  Key choices:
	//
	//    minImageCount      — triple-buffering (3) when possible.
	//    imageFormat/space   — BGRA8 sRGB for standard colour reproduction.
	//    presentMode         — Mailbox (low-latency v-sync) if available,
	//                          else FIFO (guaranteed by spec).
	//    clipped = true      — Vulkan can skip pixels hidden behind other
	//                          windows (small perf win).
	//
	//  TIP: eExclusive sharing mode is fastest but only works when one
	//       queue family does both graphics and present (our case).
	// ─────────────────────────────────────────────────────────────────────
	void createSwapChain()
	{
		auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		swapChainExtent          = chooseSwapExtent(surfaceCapabilities);
		swapChainSurfaceFormat   = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));
		vk::SwapchainCreateInfoKHR swapChainCreateInfo{.surface          = *surface,
		                                               .minImageCount    = chooseSwapMinImageCount(surfaceCapabilities),
		                                               .imageFormat      = swapChainSurfaceFormat.format,
		                                               .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
		                                               .imageExtent      = swapChainExtent,
		                                               .imageArrayLayers = 1,
		                                               .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
		                                               .imageSharingMode = vk::SharingMode::eExclusive,
		                                               .preTransform     = surfaceCapabilities.currentTransform,
		                                               .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
		                                               .presentMode      = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(*surface)),
		                                               .clipped          = true};

		swapChain       = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createImageViews()
	//
	//  An ImageView describes how to interpret a VkImage — what format,
	//  which mip level, which array layer, etc.  We need one per
	//  swap-chain image so the GPU knows how to render into them.
	//
	//  subresourceRange = { Color, mip 0, 1 level, layer 0, 1 layer }
	//  → plain 2D colour images, no mip-mapping, no array layers.
	//
	//  TIP: If you later add depth or stencil attachments, you'll create
	//       views with eDepth or eStencil aspect flags.
	// ─────────────────────────────────────────────────────────────────────
	void createImageViews()
	{
		assert(swapChainImageViews.empty());   // Guard: should only be called on a clean state

		vk::ImageViewCreateInfo imageViewCreateInfo{
		    .viewType         = vk::ImageViewType::e2D,
		    .format           = swapChainSurfaceFormat.format,
		    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

		for (auto &image : swapChainImages)
		{
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createGraphicsPipeline()
	//
	//  A Vulkan graphics pipeline bakes together:
	//    · shader stages (vertex + fragment from our Slang SPIR-V)
	//    · vertex input layout (empty — vertices are hard-coded in shader)
	//    · primitive topology (triangle list)
	//    · rasteriser settings (fill mode, cull mode, winding order)
	//    · multisampling (off for now — 1 sample)
	//    · colour blending (disabled — write RGBA directly)
	//    · dynamic state (viewport + scissor set at draw time)
	//    · pipeline layout (push constants / descriptor sets — empty)
	//
	//  Because we use dynamic rendering (Vulkan 1.3), there is no
	//  VkRenderPass.  Instead we chain a PipelineRenderingCreateInfo
	//  that tells the pipeline the colour attachment format.
	//
	//  TIP: *shaderModule dereferences the RAII handle to get the raw
	//       VkShaderModule that vk::PipelineShaderStageCreateInfo needs.
	//
	//  TIP: Both vertex and fragment shaders live in the same .spv file.
	//       Vulkan picks the right entry point via pName ("vertMain" /
	//       "fragMain") — this is how Slang's multi-entry-point model
	//       maps to Vulkan.
	// ─────────────────────────────────────────────────────────────────────
	void createGraphicsPipeline()
	{
		// Read the compiled SPIR-V binary and wrap it in a ShaderModule
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		// Two stages from the same module, distinguished by entry-point name
		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex,   .module = *shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = *shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		vk::PipelineVertexInputStateCreateInfo   vertexInputInfo;
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
		vk::PipelineViewportStateCreateInfo      viewportState{.viewportCount = 1, .scissorCount = 1};

		vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack, .frontFace = vk::FrontFace::eClockwise, .depthBiasEnable = vk::False, .depthBiasSlopeFactor = 1.0f, .lineWidth = 1.0f};

		vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{.blendEnable    = vk::False,
		                                                           .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

		vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

		std::vector dynamicStates = {
		    vk::DynamicState::eViewport,
		    vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 0, .pushConstantRangeCount = 0};

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		    {.stageCount          = 2,
		     .pStages             = shaderStages,
		     .pVertexInputState   = &vertexInputInfo,
		     .pInputAssemblyState = &inputAssembly,
		     .pViewportState      = &viewportState,
		     .pRasterizationState = &rasterizer,
		     .pMultisampleState   = &multisampling,
		     .pColorBlendState    = &colorBlending,
		     .pDynamicState       = &dynamicState,
		     .layout              = *pipelineLayout,
		     .renderPass          = nullptr},
		    {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format}};

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createCommandPool()
	//
	//  A command pool is a memory allocator for command buffers tied to
	//  a specific queue family.
	//
	//  eResetCommandBuffer lets us re-record individual command buffers
	//  every frame (drawFrame() calls commandBuffer.reset()).
	//
	//  TIP: You could also use eTransient if you record + submit once
	//       and never reset.  Drivers may optimise allocation differently.
	// ─────────────────────────────────────────────────────────────────────
	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
		    .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		    .queueFamilyIndex = queueIndex};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createCommandBuffers()
	//
	//  Allocates MAX_FRAMES_IN_FLIGHT primary command buffers.
	//  Primary = can be submitted to a queue directly (as opposed to
	//  secondary command buffers which are called from primary ones).
	//
	//  TIP: We allocate one per frame-in-flight so frame N's recording
	//       doesn't stomp on frame N-1's still-in-use buffer.
	// ─────────────────────────────────────────────────────────────────────
	void createCommandBuffers()
	{
		commandBuffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{
		    .commandPool        = *commandPool,
		    .level              = vk::CommandBufferLevel::ePrimary,
		    .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	// ─────────────────────────────────────────────────────────────────────
	//  recordCommandBuffer()
	//
	//  Records all GPU commands for one frame into the current frame's
	//  command buffer.  The sequence is:
	//
	//    1. Begin command buffer.
	//    2. Image layout transition: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL
	//       (we're about to write to this image).
	//    3. Begin dynamic rendering with a clear colour.
	//    4. Bind the graphics pipeline.
	//    5. Set viewport + scissor (dynamic state).
	//    6. Draw 3 vertices (the triangle — data is in the shader).
	//    7. End rendering.
	//    8. Image layout transition: COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC
	//       (image is ready to be shown on screen).
	//    9. End command buffer.
	//
	//  TIP: With dynamic rendering (Vulkan 1.3) we skip the entire
	//       VkRenderPass / VkFramebuffer boilerplate.  The colour
	//       attachment is specified inline via vk::RenderingInfo.
	// ─────────────────────────────────────────────────────────────────────
	void recordCommandBuffer(uint32_t imageIndex)
	{
		auto &commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});
		// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
		transition_image_layout(
		    imageIndex,
		    vk::ImageLayout::eUndefined,
		    vk::ImageLayout::eColorAttachmentOptimal,
		    {},                                                        // srcAccessMask (no need to wait for previous operations)
		    vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput         // dstStage
		);
		vk::ClearValue              clearColor     = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo = {
		    .imageView   = *swapChainImageViews[imageIndex],
		    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		    .loadOp      = vk::AttachmentLoadOp::eClear,
		    .storeOp     = vk::AttachmentStoreOp::eStore,
		    .clearValue  = clearColor};
		vk::RenderingInfo renderingInfo = {
		    .renderArea           = {.offset = {0, 0}, .extent = swapChainExtent},
		    .layerCount           = 1,
		    .colorAttachmentCount = 1,
		    .pColorAttachments    = &attachmentInfo};
		commandBuffer.beginRendering(renderingInfo);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
		commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
		commandBuffer.draw(3, 1, 0, 0);
		commandBuffer.endRendering();
		// After rendering, transition the swapchain image to PRESENT_SRC
		transition_image_layout(
		    imageIndex,
		    vk::ImageLayout::eColorAttachmentOptimal,
		    vk::ImageLayout::ePresentSrcKHR,
		    vk::AccessFlagBits2::eColorAttachmentWrite,                // srcAccessMask
		    {},                                                        // dstAccessMask
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		    vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
		);
		commandBuffer.end();
	}

	// ─────────────────────────────────────────────────────────────────────
	//  transition_image_layout()
	//
	//  Inserts a pipeline barrier that transitions a swap-chain image
	//  from one layout to another.  This is required because Vulkan
	//  images can be in different internal layouts depending on how they
	//  are used (writing vs presenting vs sampling, etc.).
	//
	//  We use the synchronization2 API (Vulkan 1.3) which is the modern
	//  replacement for the older vkCmdPipelineBarrier.  It is clearer
	//  and more explicit about stage/access dependencies.
	//
	//  TIP: src/dst stage + access masks tell the driver what work must
	//       finish before and after the transition.  Getting these wrong
	//       causes subtle rendering bugs — validation layers will warn.
	// ─────────────────────────────────────────────────────────────────────
	void transition_image_layout(
	    uint32_t                imageIndex,
	    vk::ImageLayout         old_layout,
	    vk::ImageLayout         new_layout,
	    vk::AccessFlags2        src_access_mask,
	    vk::AccessFlags2        dst_access_mask,
	    vk::PipelineStageFlags2 src_stage_mask,
	    vk::PipelineStageFlags2 dst_stage_mask)
	{
		vk::ImageMemoryBarrier2 barrier = {
		    .srcStageMask        = src_stage_mask,
		    .srcAccessMask       = src_access_mask,
		    .dstStageMask        = dst_stage_mask,
		    .dstAccessMask       = dst_access_mask,
		    .oldLayout           = old_layout,
		    .newLayout           = new_layout,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = swapChainImages[imageIndex],
		    .subresourceRange    = {
		           .aspectMask     = vk::ImageAspectFlagBits::eColor,
		           .baseMipLevel   = 0,
		           .levelCount     = 1,
		           .baseArrayLayer = 0,
		           .layerCount     = 1}};
		vk::DependencyInfo dependency_info = {
		    .dependencyFlags         = {},
		    .imageMemoryBarrierCount = 1,
		    .pImageMemoryBarriers    = &barrier};
		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createSyncObjects()
	//
	//  Creates the semaphores and fences that keep CPU and GPU in sync:
	//
	//    · presentCompleteSemaphores — one per frame-in-flight.
	//      Signalled by vkAcquireNextImage when a swap-chain image is
	//      available for rendering.
	//
	//    · renderFinishedSemaphores — one per swap-chain image.
	//      Signalled when rendering into that image is done; the
	//      present engine waits on this before displaying it.
	//
	//    · inFlightFences — one per frame-in-flight.
	//      CPU waits on this fence before reusing the command buffer for
	//      the same frame slot.  Created signalled so the first frame
	//      doesn't block.
	//
	//  TIP: MAX_FRAMES_IN_FLIGHT = 2 means the CPU can prepare frame N+1
	//       while the GPU is still executing frame N.  Increasing this
	//       adds more latency but better throughput.
	// ─────────────────────────────────────────────────────────────────────
	void createSyncObjects()
	{
		assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
			inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
		}
	}

	// ─────────────────────────────────────────────────────────────────────
	//  drawFrame()
	//
	//  The heart of the render loop.  One call = one frame displayed:
	//
	//    1. Wait for the in-flight fence (CPU blocks until the GPU has
	//       finished the PREVIOUS frame using this slot).
	//    2. Acquire the next swap-chain image (GPU signals the
	//       present-complete semaphore when it's ready).
	//    3. Reset the fence + command buffer for this slot.
	//    4. Record all draw commands into the command buffer.
	//    5. Submit the buffer to the GPU queue.
	//       - Wait on presentCompleteSemaphore before writing colour.
	//       - Signal renderFinishedSemaphore when done.
	//       - Signal inFlightFence so the CPU knows this slot is free.
	//    6. Present the finished image to the display.
	//
	//  If the swap chain is stale (window resized, etc.) we recreate it.
	//
	//  TIP: The indices can be confusing — frameIndex cycles through
	//       MAX_FRAMES_IN_FLIGHT, while imageIndex is whichever
	//       swap-chain image was acquired.
	// ─────────────────────────────────────────────────────────────────────
	void drawFrame()
	{
		// inFlightFences / presentCompleteSemaphores / commandBuffers → frameIndex
		// renderFinishedSemaphores → imageIndex
		auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (fenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for fence!");
		}

		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

		// Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined, eErrorOutOfDateKHR can be checked as a result
		// here and does not need to be caught by an exception.
		if (result == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapChain();
			return;
		}
		// On other success codes than eSuccess and eSuboptimalKHR we just throw an exception.
		// On any error code, aquireNextImage already threw an exception.
		else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
		{
			assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// Only reset the fence if we are submitting work
		device.resetFences(*inFlightFences[frameIndex]);

		commandBuffers[frameIndex].reset();
		recordCommandBuffer(imageIndex);

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		const vk::SubmitInfo   submitInfo{.waitSemaphoreCount   = 1,
		                                  .pWaitSemaphores      = &*presentCompleteSemaphores[frameIndex],
		                                  .pWaitDstStageMask    = &waitDestinationStageMask,
		                                  .commandBufferCount   = 1,
		                                  .pCommandBuffers      = &*commandBuffers[frameIndex],
		                                  .signalSemaphoreCount = 1,
		                                  .pSignalSemaphores    = &*renderFinishedSemaphores[imageIndex]};
		queue.submit(submitInfo, *inFlightFences[frameIndex]);

		const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
			                                      .pWaitSemaphores    = &*renderFinishedSemaphores[imageIndex],
			                                      .swapchainCount     = 1,
			                                      .pSwapchains        = &*swapChain,
			                                      .pImageIndices      = &imageIndex};
		result = queue.presentKHR(presentInfoKHR);
		// Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined, eErrorOutOfDateKHR can be checked as a result
		// here and does not need to be caught by an exception.
		if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else
		{
			// There are no other success codes than eSuccess; on any error code, presentKHR already threw an exception.
			assert(result == vk::Result::eSuccess);
		}
		frameIndex   = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	// ─────────────────────────────────────────────────────────────────────
	//  createShaderModule()
	//
	//  Wraps raw SPIR-V bytecode (a vector<char>) in a VkShaderModule.
	//
	//  codeSize is in bytes; pCode expects a uint32_t* pointer because
	//  SPIR-V words are 32-bit.  The reinterpret_cast is safe as long as
	//  the data is aligned — std::vector guarantees that.
	//
	//  [[nodiscard]] warns if you ignore the return value — you always
	//  want to keep the module alive until the pipeline is created.
	// ─────────────────────────────────────────────────────────────────────
	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{
		    .codeSize = code.size() * sizeof(char),
		    .pCode    = reinterpret_cast<const uint32_t *>(code.data())};
		return vk::raii::ShaderModule{device, createInfo};
	}

	// ── Swap-chain helper functions ──────────────────────────────────────
	//
	// These pure/static helpers pick the best swap-chain settings from
	// what the surface + driver advertise.  They are called once in
	// createSwapChain() and again on resize via recreateSwapChain().

	// Prefer 3 images (triple-buffering) but respect the surface limits.
	// maxImageCount == 0 means "no upper limit".
	static uint32_t chooseSwapMinImageCount(const vk::SurfaceCapabilitiesKHR &surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
		{
			minImageCount = surfaceCapabilities.maxImageCount;
		}
		return minImageCount;
	}

	// Prefer BGRA8 sRGB (standard desktop colour space).  Fallback: whatever is first.
	static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
	{
		assert(!availableFormats.empty());
		const auto formatIt = std::ranges::find_if(
		    availableFormats,
		    [](const auto &f) { return f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}

	// Prefer Mailbox (low-latency v-sync); fall back to FIFO (always available).
	// TIP: eFifo caps at monitor refresh rate; eMailbox replaces the queued
	//      frame with the latest one, reducing latency at the cost of GPU work.
	static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes,
		    [](auto pm) { return pm == vk::PresentModeKHR::eFifo; }));

		return std::ranges::any_of(availablePresentModes,
		    [](vk::PresentModeKHR v) { return v == vk::PresentModeKHR::eMailbox; })
		    ? vk::PresentModeKHR::eMailbox
		    : vk::PresentModeKHR::eFifo;
	}

	// Use the surface's current extent (window size in pixels) if the driver
	// reports it; otherwise query GLFW and clamp to the allowed range.
	// 0xFFFFFFFF means "the driver doesn't know — you tell me".
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
	{
		if (capabilities.currentExtent.width != 0xFFFFFFFF)
		{
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {
		    std::clamp<uint32_t>(width,  capabilities.minImageExtent.width,  capabilities.maxImageExtent.width),
		    std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
	}

	// ─────────────────────────────────────────────────────────────────────
	//  getRequiredInstanceExtensions()
	//
	//  Builds the list of instance extensions the app needs:
	//    · GLFW WSI extensions (VK_KHR_surface + platform-specific one)
	//    · VK_EXT_debug_utils (debug builds only — needed for the
	//      debug messenger in setupDebugMessenger())
	//
	//  TIP: glfwGetRequiredInstanceExtensions returns a pointer into
	//       GLFW's internal storage — valid until glfwTerminate().
	// ─────────────────────────────────────────────────────────────────────
	std::vector<const char *> getRequiredInstanceExtensions()
	{
		uint32_t     glfwExtensionCount = 0;
		const char **glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers)
		{
			extensions.push_back(vk::EXTDebugUtilsExtensionName);
		}

		return extensions;
	}

	// ─────────────────────────────────────────────────────────────────────
	//  debugCallback()   [static]
	//
	//  The validation-layer callback.  Must match the raw C function
	//  pointer PFN_vkDebugUtilsMessengerCallbackEXT exactly — that means
	//  raw C Vulkan types, NOT Vulkan-Hpp wrappers.
	//
	//  Returns VK_FALSE to let the Vulkan call proceed.  VK_TRUE would
	//  abort it — only useful for internal layer testing.
	//
	//  TIP: We only print warnings and errors.  If you want eVerbose
	//       or eInfo, add those checks — but expect a wall of text.
	// ─────────────────────────────────────────────────────────────────────
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	    VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
	    VkDebugUtilsMessageTypeFlagsEXT              messageType,
	    const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
	    void * /* pUserData */)
	{
		const auto severity = static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity);
		const auto type     = static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageType);

		if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
		    severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
		{
			std::cerr << "validation layer [" << vk::to_string(type) << "]: "
			          << pCallbackData->pMessage << '\n';
		}

		return VK_FALSE;
	}

	// ─────────────────────────────────────────────────────────────────────
	//  readFile()
	//
	//  Reads a binary file into a vector<char>.  Opens at the end
	//  (ios::ate) to get the size, then seeks back to read everything.
	//
	//  TIP: We open as binary — never text mode — because SPIR-V is
	//       raw bytecode.  Text mode on Windows would mangle \r\n bytes.
	// ─────────────────────────────────────────────────────────────────────
	static std::vector<char> readFile(const std::string &filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file: " + filename);
		}
		std::vector<char> buffer(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		return buffer;
	}
};

// ============================================================================
//  main()
//
//  Minimal entry point — all logic lives in HelloTriangleApplication.
//  A single try/catch catches any Vulkan or runtime error and prints it.
//
//  TIP: If the app crashes silently, enable validation layers (run a
//       Debug build) to get detailed diagnostics.
// ============================================================================
int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}