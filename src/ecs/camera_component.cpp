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
#include "camera_component.h"

#include "entity.h"
#include <iostream>

// Most of the CameraComponent class implementation is in the header file
// This file is mainly for any methods that might need additional implementation
//
// This implementation corresponds to the Camera_Transformations chapter in the tutorial:
// @see en/Building_a_Simple_Engine/Camera_Transformations/03_camera_implementation.adoc

// Initializes the camera by updating the view and projection matrices
// @see en/Building_a_Simple_Engine/Camera_Transformations/03_camera_implementation.adoc#camera-initialization
void CameraComponent::Initialize()
{
	UpdateViewMatrix();
	UpdateProjectionMatrix();
}

// Returns the view matrix, updating it if necessary
// @see en/Building_a_Simple_Engine/Camera_Transformations/03_camera_implementation.adoc#accessing-camera-matrices
const glm::mat4 &CameraComponent::GetViewMatrix()
{
	if (viewMatrixDirty)
	{
		UpdateViewMatrix();
	}
	return viewMatrix;
}

// Returns the projection matrix, updating it if necessary
// @see en/Building_a_Simple_Engine/Camera_Transformations/03_camera_implementation.adoc#accessing-camera-matrices
const glm::mat4 &CameraComponent::GetProjectionMatrix()
{
	if (projectionMatrixDirty)
	{
		UpdateProjectionMatrix();
	}
	return projectionMatrix;
}

// Updates the view matrix based on the camera's position and orientation
// @see en/Building_a_Simple_Engine/Camera_Transformations/04_transformation_matrices.adoc#view-matrix
void CameraComponent::UpdateViewMatrix()
{
	auto transformComponent = owner->GetComponent<TransformComponent>();
	const glm::vec3 position = transformComponent
	    ? transformComponent->GetPosition()
	    : glm::vec3(0.0f);

	// Use glm::lookAt with the stored target and up vectors.
	// This is the intended behaviour: SetTarget()/LookAt() configure where the
	// camera points, and UpdateViewMatrix() honours that.  The previous
	// quaternion-from-Euler approach silently ignored the target field.
	viewMatrix = glm::lookAt(position, target, up);
	viewMatrixDirty = false;
}

// Updates the projection matrix based on the camera's projection type and parameters
// @see en/Building_a_Simple_Engine/Camera_Transformations/04_transformation_matrices.adoc#projection-matrix
void CameraComponent::UpdateProjectionMatrix()
{
	if (projectionType == ProjectionType::Perspective)
	{
		projectionMatrix = glm::perspective(glm::radians(fieldOfView), aspectRatio, nearPlane, farPlane);
	}
	else
	{
		float halfWidth  = orthoWidth * 0.5f;
		float halfHeight = orthoHeight * 0.5f;
		projectionMatrix = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, nearPlane, farPlane);
	}
	projectionMatrixDirty = false;
}
