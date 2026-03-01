#include "Application.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
    VulkanView::Application app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
