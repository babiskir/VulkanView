#pragma once
class Engine;

// PhysX snippet demonstrations — each sets up its own camera, physics actors,
// and Vulkan meshes. Call after engine is initialized.
void LoadPhysicsScene_HelloWorld (Engine* engine);  // Stacks of boxes + sphere
void LoadPhysicsScene_Joints     (Engine* engine);  // Joint chains (spherical + fixed + D6)
void LoadPhysicsScene_CCD        (Engine* engine);  // Fast ball through thin wall
void LoadPhysicsScene_Gyroscopic (Engine* engine);  // L-shaped spinning body
void LoadPhysicsScene_Triggers   (Engine* engine);  // Trigger volume detection
