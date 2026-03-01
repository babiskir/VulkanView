# Dependencies

VulkanView requires the following dependencies. All are resolved via system packages.

| Dependency  | Version   | System Package         | Purpose                        |
|-------------|-----------|------------------------|--------------------------------|
| Vulkan SDK  | 1.3.275+  | `libvulkan-dev`        | Graphics API                   |
| Vulkan-Hpp  | (bundled) | `libvulkan-dev`        | C++ Vulkan bindings            |
| GLFW        | 3.3.10+   | `libglfw3-dev`         | Window creation & input        |
| GLM         | 0.9.9.8+  | `libglm-dev`           | Linear algebra (vectors, mats) |

## Install on Ubuntu/Debian

```bash
sudo apt install libvulkan-dev libglfw3-dev libglm-dev vulkan-validationlayers
```
