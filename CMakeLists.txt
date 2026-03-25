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



# ==============================================================================
# PhysX 5 (static, CPU-only rigid body physics)
# ==============================================================================
set(PHYSX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/physx")
set(PHYSX_INCLUDE_DIR "${PHYSX_SDK_DIR}/include")

set(PHYSX_LIB_DIR "${PHYSX_SDK_DIR}/lib")
if(UNIX AND NOT APPLE)
    foreach(_subdir "bin/linux.x86_64/release" "bin/linux.aarch64/release" "release" "")
        if(EXISTS "${PHYSX_SDK_DIR}/lib/${_subdir}" AND IS_DIRECTORY "${PHYSX_SDK_DIR}/lib/${_subdir}")
            set(PHYSX_LIB_DIR "${PHYSX_SDK_DIR}/lib/${_subdir}")
            break()
        endif()
    endforeach()
endif()

set(PHYSX_LIB_NAMES
    PhysXExtensions_static PhysXCharacterKinematic_static PhysXCooking_static
    PhysXPvdSDK_static PhysX_static PhysXCommon_static PhysXFoundation_static)
set(PHYSX_LIBRARIES "")
foreach(lib ${PHYSX_LIB_NAMES})
    set(_lp "${PHYSX_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    if(EXISTS "${_lp}")
        list(APPEND PHYSX_LIBRARIES "${_lp}")
    endif()
endforeach()
if(PHYSX_LIBRARIES)
    message(STATUS "PhysX found: ${PHYSX_LIB_DIR}")
else()
    message(WARNING "PhysX libraries not found in ${PHYSX_LIB_DIR}")
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

    # Shaders with multiple same-stage entry points must be compiled per-entry so
    # each resulting SPIR-V has exactly one unambiguous "main" entry point.
    # (Having two Fragment entry points both named "main" in one module is invalid SPIR-V.)

    # pbr.slang — VSMain (vertex) + PSMain (fragment) + GlassPSMain (fragment)
    # Two Fragment entry points cannot share a name in one module.
    set(PBR_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pbr.slang)
    set(PBR_ENTRIES VSMain PSMain GlassPSMain)
    foreach(ENTRY ${PBR_ENTRIES})
        string(TOLOWER ${ENTRY} ENTRY_LOWER)
        set(PBR_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/pbr_${ENTRY_LOWER}.spv)
        add_custom_command(
            OUTPUT ${PBR_SPV}
            COMMAND ${SLANGC_EXECUTABLE} ${PBR_SHADER} -entry ${ENTRY} -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${PBR_SPV}
            DEPENDS ${PBR_SHADER}
            COMMENT "Compiling pbr.slang entry ${ENTRY}"
        )
        list(APPEND SHADER_SPVS ${PBR_SPV})
    endforeach()

    # sky.slang — VSMain (vertex) + PSMain (fragment)
    set(SKY_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sky.slang)
    set(SKY_ENTRIES VSMain PSMain)
    foreach(ENTRY ${SKY_ENTRIES})
        string(TOLOWER ${ENTRY} ENTRY_LOWER)
        set(SKY_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/sky_${ENTRY_LOWER}.spv)
        add_custom_command(
            OUTPUT ${SKY_SPV}
            COMMAND ${SLANGC_EXECUTABLE} ${SKY_SHADER} -entry ${ENTRY} -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${SKY_SPV}
            DEPENDS ${SKY_SHADER}
            COMMENT "Compiling sky.slang entry ${ENTRY}"
        )
        list(APPEND SHADER_SPVS ${SKY_SPV})
    endforeach()

    # csm_depth.slang — VSMain (vertex only, depth-only pass)
    set(CSM_DEPTH_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/shaders/csm_depth.slang)
    set(CSM_DEPTH_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/csm_depth_vsmain.spv)
    add_custom_command(
        OUTPUT ${CSM_DEPTH_SPV}
        COMMAND ${SLANGC_EXECUTABLE} ${CSM_DEPTH_SHADER} -entry VSMain -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${CSM_DEPTH_SPV}
        DEPENDS ${CSM_DEPTH_SHADER}
        COMMENT "Compiling csm_depth.slang entry VSMain"
    )
    list(APPEND SHADER_SPVS ${CSM_DEPTH_SPV})

    # water.slang — VSMain (vertex) + PSMain (fragment)
    set(WATER_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/shaders/water.slang)
    set(WATER_ENTRIES VSMain PSMain)
    foreach(ENTRY ${WATER_ENTRIES})
        string(TOLOWER ${ENTRY} ENTRY_LOWER)
        set(WATER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/water_${ENTRY_LOWER}.spv)
        add_custom_command(
            OUTPUT ${WATER_SPV}
            COMMAND ${SLANGC_EXECUTABLE} ${WATER_SHADER} -entry ${ENTRY} -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${WATER_SPV}
            DEPENDS ${WATER_SHADER}
            COMMENT "Compiling water.slang entry ${ENTRY}"
        )
        list(APPEND SHADER_SPVS ${WATER_SPV})
    endforeach()

    # bloom.slang — VSMain (vertex) + DownsamplePS + UpsamplePS + AddPS (fragment)
    set(BLOOM_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bloom.slang)
    set(BLOOM_ENTRIES VSMain DownsamplePS UpsamplePS AddPS)
    foreach(ENTRY ${BLOOM_ENTRIES})
        string(TOLOWER ${ENTRY} ENTRY_LOWER)
        set(BLOOM_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/bloom_${ENTRY_LOWER}.spv)
        add_custom_command(
            OUTPUT ${BLOOM_SPV}
            COMMAND ${SLANGC_EXECUTABLE} ${BLOOM_SHADER} -entry ${ENTRY} -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${BLOOM_SPV}
            DEPENDS ${BLOOM_SHADER}
            COMMENT "Compiling bloom.slang entry ${ENTRY}"
        )
        list(APPEND SHADER_SPVS ${BLOOM_SPV})
    endforeach()

    # exposure.slang — BuildHistogramCS + ReduceHistogramCS (compute)
    set(EXPOSURE_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/shaders/exposure.slang)
    set(EXPOSURE_ENTRIES BuildHistogramCS ReduceHistogramCS)
    foreach(ENTRY ${EXPOSURE_ENTRIES})
        string(TOLOWER ${ENTRY} ENTRY_LOWER)
        set(EXPOSURE_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/exposure_${ENTRY_LOWER}.spv)
        add_custom_command(
            OUTPUT ${EXPOSURE_SPV}
            COMMAND ${SLANGC_EXECUTABLE} ${EXPOSURE_SHADER} -entry ${ENTRY} -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${EXPOSURE_SPV}
            DEPENDS ${EXPOSURE_SHADER}
            COMMENT "Compiling exposure.slang entry ${ENTRY}"
        )
        list(APPEND SHADER_SPVS ${EXPOSURE_SPV})
    endforeach()

    # Compile all other Slang shaders (skip per-entry shaders handled above)
    foreach(SHADER ${SLANG_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        get_filename_component(SHADER_NAME_WE ${SHADER_NAME} NAME_WE)
        if(SHADER_NAME_WE STREQUAL "physics" OR SHADER_NAME_WE STREQUAL "pbr"
                OR SHADER_NAME_WE STREQUAL "sky" OR SHADER_NAME_WE STREQUAL "csm_depth"
                OR SHADER_NAME_WE STREQUAL "water"
                OR SHADER_NAME_WE STREQUAL "bloom" OR SHADER_NAME_WE STREQUAL "exposure")
            continue()
        endif()
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
    src/core/update_publisher.cpp
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
    src/renderer/vma_impl.cpp
    src/renderer/vulkan_device.cpp
    src/renderer/vulkan_dispatch.cpp
    src/renderer/pipeline.cpp
    src/renderer/descriptor_manager.cpp
    # Scene
    src/scene/scene_loading.cpp
    src/scene/model_loader.cpp
    src/scene/resource_manager.cpp
    src/scene/tessendorf.cpp
    src/scene/primitive_meshes.cpp
    src/scene/physics_scenes.cpp
    src/scene/scene_factory.cpp
    src/scene/scene_ui.cpp
    src/scene/model_manager_scene.cpp
    src/scene/wave_physics.cpp
    src/scene/buoyancy_scene.cpp
    src/scene/spawner_scene.cpp
    # Systems
    src/systems/audio_system.cpp
    src/systems/physics_system.cpp
    src/systems/imgui_system.cpp
    src/systems/water_system.cpp
    # Debug
    src/debug/renderdoc_debug_system.cpp
    # Third-party
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
    imgui/ImGuizmo.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyexr
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vma
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PHYSX_INCLUDE_DIR}
)

# Enable required defines for GLM experimental extensions and MSVC math constants
target_compile_definitions(SimpleEngine PRIVATE
    GLM_ENABLE_EXPERIMENTAL
    _USE_MATH_DEFINES
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    PX_PHYSX_STATIC_LIB
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
    fftw3f  # Tessendorf ocean simulation (single-precision FFT)
    ${PHYSX_LIBRARIES}
    pthread
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

# Copy HDR environment map if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Assets/hdri/farm_field_puresky_4k.exr)
    if (NOT ANDROID)
    add_custom_command(TARGET SimpleEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/Assets/hdri
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/Assets/hdri/farm_field_puresky_4k.exr
            ${CMAKE_CURRENT_BINARY_DIR}/Assets/hdri/farm_field_puresky_4k.exr
        COMMENT "Copying env map EXR to output directory"
    )
    endif()
endif()

# Copy model and texture files if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
    if (NOT ANDROID)
    add_custom_command(TARGET SimpleEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_CURRENT_BINARY_DIR}/models
        COMMENT "Copying models to output directory"
    )
endif()
endif ()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/textures)
    if (NOT ANDROID)
    add_custom_command(TARGET SimpleEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/textures ${CMAKE_CURRENT_BINARY_DIR}/textures
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
