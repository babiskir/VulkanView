cmake_minimum_required(VERSION 3.20)
project(VulkanView VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─────────────────────────────────────────────────────────────────────────────
#  Find required packages
# ─────────────────────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
#  Locate slangc
#
#  slangc is the Slang shader compiler. We look for it in three places
#  (in priority order):
#
#    1. A path the user supplies on the CMake command line:
#         cmake -DSLANGC_EXECUTABLE=/path/to/slangc ..
#
#    2. The Vulkan SDK bin directory — the SDK ships slangc alongside glslc
#       starting from recent versions.
#
#    3. The system PATH (find_program default behaviour).
#
#  CMake prints a fatal error if it cannot locate slangc so the developer gets
#  a clear message rather than a cryptic compile failure later.
# ─────────────────────────────────────────────────────────────────────────────
find_program(SLANGC_EXECUTABLE
    NAMES slangc
    HINTS
        "$ENV{VULKAN_SDK}/bin"   # Vulkan SDK installation
        "$ENV{HOME}/.local/bin"  # User-local installs
    DOC "Path to the slangc Slang shader compiler"
)

if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR
        "slangc not found. Install the Vulkan SDK (which bundles slangc) "
        "or pass -DSLANGC_EXECUTABLE=/path/to/slangc to CMake.")
endif()

message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")

# ─────────────────────────────────────────────────────────────────────────────
#  Source files
# ─────────────────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
)

# Slang shader sources — all .slang files that should be compiled together
# into a single SPIR-V binary (slang.spv) by the shader target below.
set(SHADER_SLANG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang
)

# ─────────────────────────────────────────────────────────────────────────────
#  Main executable
# ─────────────────────────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
)

# Enable Vulkan-Hpp designated initializers (C++20)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ─────────────────────────────────────────────────────────────────────────────
#  add_slang_shader_target(TARGET SOURCES <file1> [file2 …])
#
#  Compiles one or more .slang files into a single SPIR-V 1.4 binary called
#  slang.spv placed in the shaders/ output directory.
#
#  Flags used:
#    -target spirv              — emit SPIR-V (not GLSL, DXIL, etc.)
#    -profile spirv_1_4         — target SPIR-V version 1.4
#    -emit-spirv-directly       — bypass GLSL emission; produce SPIR-V natively
#    -fvk-use-entrypoint-name   — preserve the Slang entry-point names in the
#                                 SPIR-V so the Vulkan driver can find them
#    -entry vertMain            — vertex shader entry point
#    -entry fragMain            — fragment shader entry point
#    -o slang.spv               — output filename (relative to WORKING_DIRECTORY)
#
#  Two add_custom_command calls mirror the tutorial's approach:
#    1. Ensure the shaders/ output directory exists.
#    2. Run slangc whenever any source file is newer than slang.spv.
#
#  The TARGET custom target depends on the .spv output so a simple
#  cmake --build . --target <TARGET> recompiles only when needed.
# ─────────────────────────────────────────────────────────────────────────────
function(add_slang_shader_target TARGET)
    cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})

    if(NOT SHADER_SOURCES)
        message(FATAL_ERROR "add_slang_shader_target: no SOURCES provided for target '${TARGET}'")
    endif()

    set(SHADERS_DIR  ${CMAKE_CURRENT_LIST_DIR}/shaders)
    set(SPV_OUTPUT   ${SHADERS_DIR}/slang.spv)

    # Entry-point flags — one -entry flag per stage
    set(ENTRY_POINTS -entry vertMain -entry fragMain)

    # Step 1: create the output directory if it does not yet exist.
    # Using a stamp file as OUTPUT so CMake can track whether this command
    # has been executed (a bare directory path is unreliable as a CMake output).
    set(DIR_STAMP ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_shaders_dir.stamp)

    add_custom_command(
        OUTPUT  ${DIR_STAMP}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${DIR_STAMP}
        COMMENT "Ensuring shaders output directory exists"
        VERBATIM
    )

    # Step 2: compile Slang sources → slang.spv
    add_custom_command(
        OUTPUT  ${SPV_OUTPUT}
        COMMAND ${SLANGC_EXECUTABLE}
                ${SHADER_SOURCES}
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                ${ENTRY_POINTS}
                -o ${SPV_OUTPUT}
        WORKING_DIRECTORY ${SHADERS_DIR}
        DEPENDS ${DIR_STAMP} ${SHADER_SOURCES}
        COMMENT "Compiling Slang shaders → ${SPV_OUTPUT}"
        VERBATIM
    )

    # Expose a named target so the executable can declare a dependency on it
    add_custom_target(${TARGET} DEPENDS ${SPV_OUTPUT})
endfunction()

# ─────────────────────────────────────────────────────────────────────────────
#  Shader build target
#
#  VulkanViewShaders compiles shaders/shader.slang → shaders/slang.spv.
#  VulkanView lists it as a dependency so the SPIR-V is always up to date
#  before the executable is run.
# ─────────────────────────────────────────────────────────────────────────────
add_slang_shader_target(VulkanViewShaders SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(${PROJECT_NAME} VulkanViewShaders)

# ─────────────────────────────────────────────────────────────────────────────
#  Make shaders accessible from the build directory
#
#  The executable opens "shaders/slang.spv" via a relative path.  When run
#  from the build directory that path must resolve, so we create a symlink:
#
#    <build>/shaders  →  <source>/shaders
#
#  This way the same relative path works whether you run from the source tree
#  or from the build tree, and the compiled .spv is always found.
# ─────────────────────────────────────────────────────────────────────────────
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMENT "Symlinking shaders/ into build directory"
    VERBATIM
)
