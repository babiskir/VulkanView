/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ---------------------------------------------------------------------------
// Scene-selection entry point.
//
// Usage:
//   ./SimpleEngine                        -- shows ImGui picker at startup
//   ./SimpleEngine --scene bistro         -- loads Bistro directly
//   ./SimpleEngine --scene water          -- loads Water directly
//   ./SimpleEngine --scene physics-hello  -- loads PhysX HelloWorld directly
//
// All scenes are registered via SceneFactory::RegisterAll().
// ---------------------------------------------------------------------------

#include "crash_reporter.h"
#include "engine.h"
#include "renderer.h"
#include "scene_factory.h"
#include "water_system.h"

#include <iostream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int  WINDOW_WIDTH            = 800;
constexpr int  WINDOW_HEIGHT           = 600;
constexpr bool ENABLE_VALIDATION_LAYERS =
#if defined(NDEBUG)
    false;
#else
    true;
#endif

// ---------------------------------------------------------------------------
// Desktop entry point
// ---------------------------------------------------------------------------
#if !defined(PLATFORM_ANDROID)
int main(int argc, char* argv[])
{
    try {
        CrashReporter::GetInstance().Initialize("crashes", "SimpleEngine", "1.0.0");

        // Parse optional --scene <id> to skip the picker
        std::string sceneArg;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--scene" && i + 1 < argc)
                sceneArg = argv[++i];
        }

        Engine engine;
        if (!engine.Initialize("Simple Engine", WINDOW_WIDTH, WINDOW_HEIGHT,
                               ENABLE_VALIDATION_LAYERS))
            throw std::runtime_error("Failed to initialize engine");

        SceneFactory& factory = SceneFactory::Instance();
        factory.RegisterAll();

        if (!sceneArg.empty()) {
            if (!factory.Load(sceneArg, &engine))
                throw std::runtime_error("Unknown scene: " + sceneArg);
            engine.SetSceneLoaded(true);
        } else {
            factory.InstallPicker(&engine);
        }

        engine.Run();

        // Clean up water system if it was active
        if (auto* renderer = engine.GetRenderer()) {
            if (renderer->waterSystem) {
                renderer->waterSystem->Cleanup();
                delete renderer->waterSystem;
                renderer->waterSystem = nullptr;
            }
        }

        CrashReporter::GetInstance().Cleanup();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
        CrashReporter::GetInstance().Cleanup();
        return 1;
    }
}
#endif // !PLATFORM_ANDROID
