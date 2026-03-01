# VulkanView

A Vulkan-based 3D viewer application built with C++20, GLFW, and the Vulkan SDK.

## Prerequisites

- **Vulkan SDK** (1.3+)
- **GLFW** (3.3+)
- **CMake** (3.20+)
- A C++20 compatible compiler (GCC 11+, Clang 14+, MSVC 2022+)

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Running

```bash
./build/VulkanView
```

## Project Structure

```
├── CMakeLists.txt       # Build configuration
├── src/
│   ├── main.cpp         # Entry point
│   ├── Application.h    # Application class declaration
│   └── Application.cpp  # Vulkan initialization & main loop
└── README.md
```
