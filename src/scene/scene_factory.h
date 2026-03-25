#pragma once

#include <functional>
#include <string>
#include <vector>

class Engine;

/**
 * @brief Registry of loadable scenes.
 *
 * Each scene has a CLI id ("bistro", "water", "physics-hello"), an ImGui label,
 * an optional group name for submenus ("physics"), and a load function.
 *
 * Usage:
 *   // Register all scenes once at startup:
 *   SceneFactory::Instance().RegisterAll();
 *
 *   // Load via CLI arg:
 *   SceneFactory::Instance().Load("bistro", &engine);
 *
 *   // Show ImGui picker:
 *   SceneFactory::Instance().InstallPicker(&engine);
 */
class SceneFactory {
public:
    struct Entry {
        std::string id;     ///< CLI identifier e.g. "bistro", "physics-hello"
        std::string label;  ///< ImGui button text
        std::string group;  ///< "" = root menu; "physics" = physics submenu
        std::function<void(Engine*)> load;
    };

    static SceneFactory& Instance();

    /** Register all built-in scenes. Call once after engine is created. */
    void RegisterAll();

    /** Load a scene by CLI id. Returns false if id is unknown. */
    bool Load(const std::string& id, Engine* engine) const;

    /** Install the ImGui scene-picker overlay on the engine. */
    void InstallPicker(Engine* engine) const;

    const std::vector<Entry>& Entries() const { return entries_; }

private:
    SceneFactory() = default;
    void Register(Entry e);
    std::vector<Entry> entries_;
};
