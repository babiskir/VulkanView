cmake_minimum_required(VERSION 3.29)

project(SimpleEngine VERSION 1.0.0 LANGUAGES CXX C)

# Option to enable/disable Vulkan C++20 module support for this standalone project
option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan in SimpleEngine" OFF)

# Enable C++ module dependency scanning only when modules are enabled
if(ENABLE_CPP20_MODULE)
    set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

# Add CMake module path for custom find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

# Find required packages
find_package (glm REQUIRED)
find_package (Vulkan REQUIRED)
find_package (tinygltf REQUIRED)
find_package (KTX)

# Find or download Vulkan-Hpp headers matching the Vulkan SDK/NDK version
find_package(VulkanHpp REQUIRED)

if(ENABLE_CPP20_MODULE)
    # Set up Vulkan C++ module for this standalone project
    add_library(VulkanCppModule)
    add_library(Vulkan::cppm ALIAS VulkanCppModule)

    target_compile_definitions(VulkanCppModule
            PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )
    target_include_directories(VulkanCppModule
            PUBLIC
            "${Vulkan_INCLUDE_DIR}"
            "${VulkanHpp_INCLUDE_DIRS}"
    )
    target_link_libraries(VulkanCppModule
            PUBLIC
            Vulkan::Vulkan
    )

    set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

    target_sources(VulkanCppModule
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES
            BASE_DIRS
            "${VulkanHpp_CPPM_DIR}"
            FILES
            "${VulkanHpp_CPPM_DIR}/vulkan/vulkan.cppm"
    )

    # MSVC-specific options to improve module support
    if(MSVC)
        target_compile_options(VulkanCppModule PRIVATE
            /std:c++latest
            /permissive-
            /Zc:__cplusplus
            /EHsc
            /Zc:preprocessor
        )
    endif()
else()
    add_library(VulkanCppModule INTERFACE)
    add_library(Vulkan::cppm ALIAS VulkanCppModule)
    target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
    target_compile_definitions(VulkanCppModule
            INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )
    target_include_directories(VulkanCppModule INTERFACE "${VulkanHpp_INCLUDE_DIRS}")
endif()



# Platform-specific settings
if(ANDROID)
    # Android-specific settings
    add_definitions(-DPLATFORM_ANDROID)
    find_package(game-activity REQUIRED CONFIG)
else()
    # Desktop-specific settings
    add_definitions(-DPLATFORM_DESKTOP)
    find_package(glfw3 REQUIRED)
    find_package(OpenAL REQUIRED)
endif()

# Shader compilation
# Find Slang shaders (exclude utility modules that are imported, not compiled standalone)
file(GLOB SLANG_SHADER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang)
list(FILTER SLANG_SHADER_SOURCES EXCLUDE REGEX ".*/(common_types|pbr_utils|lighting_utils|tonemapping_utils)\\.slang$")

# Find slangc executable (optional)
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin ${CMAKE_CURRENT_SOURCE_DIR}/tools)

if(SLANGC_EXECUTABLE)
    # Ensure the output directory for compiled shaders exists
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

    # Compile Slang shaders using slangc
    foreach(SHADER ${SLANG_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        get_filename_component(SHADER_NAME_WE ${SHADER_NAME} NAME_WE)
        string(REGEX REPLACE "\.slang$" "" OUTPUT_NAME ${SHADER_NAME})
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_NAME}.spv
            COMMAND ${SLANGC_EXECUTABLE} ${SHADER} -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_NAME}.spv
            DEPENDS ${SHADER}
            COMMENT "Compiling Slang shader ${SHADER_NAME} with slangc"
        )
        list(APPEND SHADER_SPVS ${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_NAME}.spv)
    endforeach()

    add_custom_target(shaders DEPENDS ${SHADER_SPVS})
else()
    message(STATUS "slangc not found. Skipping shader compilation step.")
    add_custom_target(shaders)
endif()

# Source files
# NOTE: Android builds include this project via `add_subdirectory(...)` from
# `android/app/src/main/cpp/CMakeLists.txt`, so we must not require a desktop `main()`.
set(SOURCES_COMMON
    # Core
    src/core/engine.cpp
    src/core/platform.cpp
    # ECS
    src/ecs/entity.cpp
    src/ecs/component.cpp
    src/ecs/transform_component.cpp
    src/ecs/mesh_component.cpp
    src/ecs/camera_component.cpp
    src/ecs/animation_component.cpp
    # Renderer
    src/renderer/renderer_core.cpp
    src/renderer/renderer_rendering.cpp
    src/renderer/renderer_pipelines.cpp
    src/renderer/renderer_compute.cpp
    src/renderer/renderer_utils.cpp
    src/renderer/renderer_resources.cpp
    src/renderer/renderer_ray_query.cpp
    src/renderer/memory_pool.cpp
    src/renderer/vulkan_device.cpp
    src/renderer/vulkan_dispatch.cpp
    src/renderer/pipeline.cpp
    src/renderer/descriptor_manager.cpp
    # Scene
    src/scene/scene_loading.cpp
    src/scene/model_loader.cpp
    src/scene/resource_manager.cpp
    # Systems
    src/systems/audio_system.cpp
    src/systems/physics_system.cpp
    src/systems/imgui_system.cpp
    # Debug
    src/debug/renderdoc_debug_system.cpp
    # Third-party
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    third_party/mikktspace/mikktspace.c
)

set(SOURCES_DESKTOP
        src/core/main.cpp
)

# Create target
if (ANDROID)
    # Android: build the engine as a library to be linked into the app's `simple_engine_android` SHARED library.
    add_library(SimpleEngine STATIC ${SOURCES_COMMON})
else ()
    # Desktop: build the runnable executable (unchanged behavior vs `HEAD`).
    add_executable(SimpleEngine ${SOURCES_COMMON} ${SOURCES_DESKTOP})
endif ()

add_dependencies(SimpleEngine shaders)
set_target_properties (SimpleEngine PROPERTIES CXX_STANDARD 20)

# Add include directories for the restructured source tree
target_include_directories(SimpleEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ecs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/systems
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scene
    ${CMAKE_CURRENT_SOURCE_DIR}/src/debug
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mikktspace
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Enable required defines for GLM experimental extensions and MSVC math constants
target_compile_definitions(SimpleEngine PRIVATE
    GLM_ENABLE_EXPERIMENTAL
    _USE_MATH_DEFINES
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
)

# Link libraries
# Prefer the Vulkan C++ module target when available (configured at the parent level),
# but fall back to the standard Vulkan library otherwise.
if(TARGET Vulkan::cppm)
    target_link_libraries(SimpleEngine PUBLIC Vulkan::cppm)
else()
    target_link_libraries(SimpleEngine PUBLIC Vulkan::Vulkan)
endif()

target_link_libraries(SimpleEngine PUBLIC
    glm::glm
    tinygltf::tinygltf
    KTX::ktx
)

if (ANDROID)
    target_link_libraries(SimpleEngine PUBLIC game-activity::game-activity OpenSLES android log)
else ()
    target_link_libraries(SimpleEngine PRIVATE glfw OpenAL::OpenAL)
endif()

# Windows/MSVC portability and build settings
if(MSVC)
    # Avoid Windows.h macro pollution and CRT warnings; improve conformance and build perf
    target_compile_definitions(SimpleEngine PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
    target_compile_options(SimpleEngine PRIVATE
        /permissive-
        /Zc:__cplusplus
        /EHsc
        /W3
        /MP
        /bigobj
    )
    # Crash reporter uses Dbghelp; pragma should suffice, but make it explicit for clarity
    target_link_libraries(SimpleEngine PRIVATE Dbghelp)
elseif(WIN32)
    # Non-MSVC Windows toolchains (e.g., MinGW)
    target_compile_definitions(SimpleEngine PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
    # Crash reporter uses Dbghelp for minidump generation
    target_link_libraries(SimpleEngine PRIVATE dbghelp)
endif()

# Copy model and texture files if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
    if (NOT ANDROID)
    add_custom_command(TARGET SimpleEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models
        COMMENT "Copying models to output directory"
    )
endif()
endif ()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/textures)
    if (NOT ANDROID)
    add_custom_command(TARGET SimpleEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/textures ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/textures
        COMMENT "Copying textures to output directory"
    )
endif()
endif ()

# Add packaging configuration
include(CPack)

# Set package properties
set(CPACK_PACKAGE_NAME "SimpleEngine")
set(CPACK_PACKAGE_VENDOR "SimpleEngine Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A simple game engine built with Vulkan")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "SimpleEngine")

# Set platform-specific package generators
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_PACKAGE_NAME "SimpleEngine")
    set(CPACK_NSIS_DISPLAY_NAME "SimpleEngine")
    set(CPACK_NSIS_HELP_LINK "https://github.com/yourusername/SimpleEngine")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/yourusername/SimpleEngine")
    set(CPACK_NSIS_CONTACT "your.email@example.com")
    set(CPACK_NSIS_MODIFY_PATH ON)
elseif(APPLE)
    set(CPACK_GENERATOR "ZIP;DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "SimpleEngine")
    set(CPACK_DMG_FORMAT "UDBZ")
else()
    set(CPACK_GENERATOR "ZIP;DEB")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Your Name <your.email@example.com>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "games")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libvulkan1, libglfw3, libglm-dev, libktx-dev")
endif()

# Include binary and resource directories in the package
if (NOT ANDROID)
install(TARGETS SimpleEngine DESTINATION bin)
if(SLANGC_EXECUTABLE)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders DESTINATION share/SimpleEngine)
endif()

# Install models and textures if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models DESTINATION share/SimpleEngine)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/textures)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/textures DESTINATION share/SimpleEngine)
endif()

# Install README if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md DESTINATION share/SimpleEngine)
endif()
endif ()
