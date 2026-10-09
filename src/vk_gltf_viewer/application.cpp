#include <cassert>
#include <numbers>
#include <ranges>

#include <tracy/Tracy.hpp>

#include <fmt/xchar.h>
#include <fmt/std.h>

#include <vulkan/vk.hpp>
#include <vulkan/pipeline_builder.hpp>
#include <GLFW/glfw3.h> // After Vulkan includes so that it detects it
#include <imgui_impl_glfw.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nvidia/dlss.hpp>
#if defined(VKV_NV_DLSS)
#include <nvsdk_ngx_helpers_vk.h>
#endif
#include <vk_gltf_viewer/scheduler.hpp>
#include <vk_gltf_viewer/application.hpp>
#include <vk_gltf_viewer/assets.hpp>
#include <spirv_manifest.hpp>

#include <visbuffer/visbuffer.h>

#include <fastgltf/tools.hpp>

namespace fg = fastgltf;

void glfwErrorCallback(int errorCode, const char* description) {
	if (errorCode != GLFW_NO_ERROR) {
		fmt::print(stderr, "GLFW error: 0x{:x} {}\n", errorCode, description);
	}
}

void glfwResizeCallback(GLFWwindow* window, int width, int height) {
	ZoneScoped;
	if (width > 0 && height > 0) {
		glm::u32vec2 res(width, height);
		decltype(auto) app = *static_cast<application*>(glfwGetWindowUserPointer(window));
		app.renderer->update_resolution(res);
	}
}

void mark_children_as_dirty(entt::registry& registry, entt::entity entity) {
	ZoneScoped;
	// When a node entitiy's transform is marked dirty, we need to mark all children as dirty, too.
	auto curr = registry.get<node_component_t>(entity).first_child;
	while (curr != entt::null) {
		const auto is_dirty = registry.any_of<dirty_transform_tag_t>(curr);
		if (!is_dirty) {
			// Since this function is setup to run whenever the dirty tag is constructed,
			// this function will implicitly be called in the following line.
			registry.emplace<dirty_transform_tag_t>(curr);
		}
		curr = registry.get<node_component_t>(curr).next;
	}
}

void recompute_transforms(entt::registry& registry, graphics::scene_t& scene) {
	ZoneScoped;
	// Recomputes all dirty transforms by first sorting all entities with the dirty tag, so that
	// we can safely assume that all parent's transforms have been updated before we update children.
	registry.sort<dirty_transform_tag_t>([&registry](const entt::entity lhs, const entt::entity rhs) {
		// TODO: Is this sort *actually* correct? It seems to work generally and it's what skypjack originally wrote
		// for a blog post on this subject, but he noted that he did *not* test its correctness.
		const auto &clhs = registry.get<node_component_t>(lhs);
		const auto &crhs = registry.get<node_component_t>(rhs);
		return crhs.parent == lhs || clhs.next == rhs
			|| (!(clhs.parent == rhs || crhs.next == lhs) && (clhs.parent < crhs.parent || (clhs.parent == crhs.parent && &clhs < &crhs)));
	});

	registry.view<dirty_transform_tag_t, transform_component_t, node_component_t>().each(
		[&](const auto entity, auto& transform, const auto& node) {
		const auto parent_transform = [&]() {
			if (node.parent != entt::null) {
				return registry.get<transform_component_t>(node.parent).transform;
			}
			return glm::mat4(1.f);
		}();

		const auto rot_matrix = glm::mat4_cast(transform.rotation);
		transform.transform = glm::scale(glm::translate(parent_transform, transform.position) * rot_matrix, transform.scale);

		if (auto* mesh = registry.try_get<mesh_component_t>(entity); mesh) {
			scene.update_transform(mesh->instance_index, transform.transform);
		}
	});

	// Finally, remove the dirty tag from every entity.
	registry.clear<dirty_transform_tag_t>();
}

application::application(std::span<std::filesystem::path> gltfs) {
	ZoneScoped;

	// Initialize GLFW
	glfwSetErrorCallback(glfwErrorCallback);
	if (glfwInit() != GLFW_TRUE) {
		throw std::runtime_error("Failed to initialize GLFW");
	}
	deletion_queue.push([]() { glfwTerminate(); });

	// Get the main monitor's video mode
	auto* mainMonitor = glfwGetPrimaryMonitor();
	const auto* videoMode = glfwGetVideoMode(mainMonitor);

	// Create the window
	glfwDefaultWindowHints();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(
		static_cast<int>(static_cast<float>(videoMode->width) * 0.9f),
		static_cast<int>(static_cast<float>(videoMode->height) * 0.9f),
		"vk_viewer", nullptr, nullptr);
	if (window == nullptr)
		throw std::runtime_error("Failed to create window");
	deletion_queue.push([this]() { glfwDestroyWindow(window); });

	glfwSetWindowUserPointer(window, this);
	glfwSetWindowSizeCallback(window, glfwResizeCallback);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	//ImGui_ImplGlfw_InitForVulkan(window, true);
	ImGui_ImplGlfw_InitForOther(window, true);

	renderer = graphics::renderer::create_renderer(window);
	scene = renderer->create_shared_scene();
	camera = std::make_unique<camera_t>();

	registry.on_construct<dirty_transform_tag_t>().connect<&mark_children_as_dirty>();

	for (auto& gltf : gltfs) {
		auto& task = asset_load_tasks.emplace_back(std::make_shared<asset_load_task>(renderer, gltf));
		task_scheduler.AddTaskSetToPipe(task.get());
	}
}

application::~application() noexcept {
	// TODO: is there perhaps some way to detach the tasks, so that we don't lock up
	//       the application during shutting down?
	for (auto& task : asset_load_tasks)
		task_scheduler.WaitforTask(task.get());
}

void application::add_asset_to_scene(asset_load_task& task) {
	ZoneScoped;
	textures.append_range(task.textures);

	auto& asset = *task.asset;

	auto function = [&](std::size_t nodeIndex, fastgltf::math::fmat4x4 node_matrix, entt::entity entity, auto& self) -> void {
		assert(asset.nodes.size() > nodeIndex);
		auto& node = asset.nodes[nodeIndex];
		node_matrix = fastgltf::getTransformMatrix(node, node_matrix);

		const auto& [translation, rotation, scale] = std::get<fastgltf::TRS>(node.transform);
		registry.emplace<transform_component_t>(entity, transform_component_t {
			.position = glm::make_vec3(translation.data()),
			.rotation = glm::fquat::wxyz(rotation.w(), rotation.x(), rotation.y(), rotation.z()),
			.scale = glm::make_vec3(scale.data()),
			.transform = glm::make_mat4x4(node_matrix.data())
		});

		registry.emplace<name_component_t>(entity, node.name.empty() ? "Node" : std::string(node.name));

		std::vector<entt::entity> children;
		children.reserve(node.children.size());
		for (std::size_t i = 0; i < node.children.size(); ++i) {
			const auto child_entity = children.emplace_back(registry.create());
			registry.emplace<node_component_t>(child_entity).parent = entity;
		}

		if (node.meshIndex.has_value()) {
			const auto& primitive_indices = task.meshes[*node.meshIndex].primitive_indices;
			if (primitive_indices.size() == 1) {
				const auto& mesh = task.primitives[primitive_indices.front()];
				const auto& mesh_component = registry.emplace<mesh_component_t>(entity, mesh_component_t {
					.mesh = mesh,
					.instance_index = this->scene->add_mesh_instance(mesh),
				});

				this->scene->update_transform(mesh_component.instance_index, glm::make_mat4x4(node_matrix.data()));
			} else {
				// Instead of having a mesh component store multiple meshes, we'll just create extra nodes
				// so that any node can only ever hold a single mesh.

				children.reserve(children.size() + primitive_indices.size());
				for (const auto& idx : primitive_indices) {
					const auto primitive_node = children.emplace_back(registry.create());
					registry.emplace<node_component_t>(primitive_node).parent = entity;
					registry.emplace<transform_component_t>(primitive_node);
					registry.emplace<name_component_t>(primitive_node, "Node");

					const auto& mesh = task.primitives[idx];
					const auto& mesh_component = registry.emplace<mesh_component_t>(primitive_node, mesh_component_t {
						.mesh = mesh,
						.instance_index = this->scene->add_mesh_instance(mesh),
					});

					this->scene->update_transform(mesh_component.instance_index, glm::make_mat4x4(node_matrix.data()));
				}
			}
		}

		if (!children.empty()) {
			registry.get_or_emplace<node_component_t>(entity).first_child = children[0];

			for (std::size_t i = 0; i < children.size(); ++i) {
				auto& child_node_component = registry.get_or_emplace<node_component_t>(children[i]);
				if (i != 0)
					child_node_component.prev = children[i - 1];
				if (i + 1 < children.size())
					child_node_component.next = children[i + 1];

				// Since we sometimes create additional nodes not found in the glTF hierarchy, this
				// makes sure we don't go out of bounds. This also silently requires that actual
				// glTF nodes are placed at the start of the children vector.
				if (i < node.children.size()) {
					self(node.children[i], node_matrix, children[i], self);
				}
			}
		}
	};

	// Initialise the root node
	if (root_scene_node == entt::null) {
		root_scene_node = registry.create();
		registry.emplace<node_component_t>(root_scene_node);
		registry.emplace<transform_component_t>(root_scene_node);
		registry.emplace<name_component_t>(root_scene_node, "Root");
	}

	// Find the last child of the root node, so that we can append any new nodes to the implicit list
	auto last_root_child = registry.get<node_component_t>(root_scene_node).first_child;
	while (last_root_child != entt::null) {
		const auto next = registry.get<node_component_t>(last_root_child).next;
		if (next == entt::null) {
			break;
		}
		last_root_child = next;
	}

	// Iterate through the glTF scene and create node entities accordingly
	const auto scene_index = 0;
	const auto& scene = asset.scenes[scene_index];
	for (auto& sceneNode : scene.nodeIndices) {
		const auto node_entity = registry.create();

		registry.emplace<node_component_t>(node_entity, node_component_t {
			.parent = root_scene_node,
			.prev = last_root_child,
		});
		if (last_root_child != entt::null)
			registry.get<node_component_t>(last_root_child).next = node_entity;
		else
			registry.get<node_component_t>(root_scene_node).first_child = node_entity;
		last_root_child = node_entity;

		function(sceneNode, fastgltf::math::fmat4x4(), node_entity, function);
	}

	fmt::print("Finished loading asset: {}\n", task.asset_path);
}

void application::run() {
	ZoneScoped;

	std::size_t currentFrame = 0;
	while (!glfwWindowShouldClose(window)) {
		if (renderer->can_render()) {
			ZoneScopedN("glfwPollEvents");
			glfwPollEvents();
		} else {
			// This will wait until we get an event, like the resize event which will recreate the swapchain.
			glfwWaitEvents();
			continue;
		}

		// Check if any asset load task has completed, and get the Asset object
		for (auto& task : asset_load_tasks) {
			if (task->GetIsCompleteWithExceptions()) {
				add_asset_to_scene(*task);
				//vkQueueWaitIdle(device->graphicsQueue);
				//world->addAsset(task);
				task.reset();
			}
		}
		std::erase_if(asset_load_tasks, [](std::shared_ptr<asset_load_task>& value) {
			return !bool(value);
		});

		auto currentTime = glfwGetTime();
		deltaTime = currentTime - lastFrame;
		lastFrame = currentTime;

		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		render_ui();

		recompute_transforms(registry, *scene);

		currentFrame = ++currentFrame % graphics::frame_overlap;

		renderer->prepare_frame(currentFrame);

		camera->update(window, deltaTime, renderer->get_render_resolution());

		renderer->draw(currentFrame, *scene, *camera, static_cast<float>(deltaTime));

		FrameMark;
		firstFrame = false;
	}
}

void application::render_ui() {
	ZoneScoped;
	if (ImGui::Begin("vk_gltf_viewer", nullptr, /*ImGuiWindowFlags_AlwaysAutoResize |*/ ImGuiWindowFlags_NoMove)) {
		if (ImGui::BeginTabBar("#tabbar")) {
			if (ImGui::BeginTabItem("General")) {
				ImGui::SeparatorText("Performance");

				ImGui::Text("Frametime: %.2f ms", deltaTime * 1000.);
				ImGui::Text("FPS: %.2f", 1. / deltaTime);
				ImGui::Text("AFPS: %.2f rad/s", 2. * std::numbers::pi_v<double> / deltaTime); // Angular FPS

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Scene")) {
				if (selected_node != entt::null) {
					auto& transform = registry.get<transform_component_t>(selected_node);
					// TODO: Change internal representation to euler angles to avoid weird UX?
					auto euler_angles = glm::degrees(glm::eulerAngles(transform.rotation)); // pitch as x, yaw as y, roll as z

					bool changed = false;
					changed |= ImGui::DragFloat3("Position##node_translation", glm::value_ptr(transform.position));
					changed |= ImGui::DragFloat3("Rotation##node_rotation", glm::value_ptr(euler_angles));
					changed |= ImGui::DragFloat3("Scale##node_scale", glm::value_ptr(transform.scale));

					if (changed) {
						transform.rotation = glm::fquat(glm::radians(euler_angles));
						registry.emplace<dirty_transform_tag_t>(selected_node);
					}
				}

				render_scene_tree(root_scene_node);

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Graphics")) {
				auto color_space_combo = [&](const std::span<const shaders::color_space_e> color_spaces, const shaders::color_space_e current, const char* name, auto func) {
					const auto current_name = shaders::get_color_space_name(current);
					if (ImGui::BeginCombo(name, current_name.data())) {
						for (const auto& color_space : color_spaces) {
							const bool isSelected = color_space == current;
							const auto color_space_name = shaders::get_color_space_name(color_space);
							if (ImGui::Selectable(color_space_name.data(), isSelected)) {
								func(renderer.get(), color_space);
							}
							if (isSelected)
								ImGui::SetItemDefaultFocus();
						}
						ImGui::EndCombo();
					}
				};

				color_space_combo(renderer->get_internal_color_spaces(), renderer->get_internal_color_space(),
					"Internal color space", std::mem_fn(&graphics::renderer::set_internal_color_space));
				color_space_combo(renderer->get_output_color_spaces(), renderer->get_output_color_space(),
					"Output color space", std::mem_fn(&graphics::renderer::set_output_color_space));

				ImGui::SeparatorText("Resolution scaling");

				const auto modes = renderer->get_scaling_modes();
				const auto selected_mode_name = graphics::get_scaling_mode_name(scaling_mode);
				if (ImGui::BeginCombo("Resolution scaling", selected_mode_name.data())) {
					for (const auto& mode : modes) {
						const bool isSelected = mode == scaling_mode;
						const auto mode_name = graphics::get_scaling_mode_name(mode);
						if (ImGui::Selectable(mode_name.data(), isSelected)) {
							scaling_mode = mode;

							const auto presets = renderer->get_scaling_presets(mode);
							assert(!presets.empty());
							renderer->use_upscaler(scaling_mode, presets.front());
							scaling_mode_preset = 0;
						}
						if (isSelected)
							ImGui::SetItemDefaultFocus();
					}
					ImGui::EndCombo();
				}

				const auto presets = renderer->get_scaling_presets(scaling_mode);
				const auto& selected_preset_name = presets[scaling_mode_preset].name;
				ImGui::BeginDisabled(presets.size() == 1);
				if (ImGui::BeginCombo("Scaling presets", selected_preset_name.data())) {
					for (std::size_t i = 0; const auto& [factor, name] : presets) {
						const bool isSelected = i == scaling_mode_preset;
						if (ImGui::Selectable(name.data(), isSelected)) {
							scaling_mode_preset = i;

							renderer->use_upscaler(scaling_mode, presets[i]);
						}
						if (isSelected)
							ImGui::SetItemDefaultFocus();

						++i;
					}
					ImGui::EndCombo();
				}
				ImGui::EndDisabled();

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Debug")) {
				ImGui::SeparatorText("Camera");
				auto pos = camera->get_position();
				ImGui::Text("Position: <%.2f, %.2f, %.2f>", pos.x, pos.y, pos.z);
				ImGui::Text("Direction: <%.2f, %.2f, %.2f>", camera->direction.x, camera->direction.y, camera->direction.z);
				ImGui::DragFloat("Camera speed multiplier", &camera->speedMultiplier);

				ImGui::SeparatorText("Culling");
				ImGui::Checkbox("Freeze Camera frustum", &camera->freezeCameraFrustum);
				ImGui::Checkbox("Freeze Occlusion matrix", &camera->freezeCullingMatrix);
				//ImGui::Checkbox("Freeze animations", &world->freezeAnimations);

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}
	}
	ImGui::End();

	// ImGui::ShowDemoWindow();

	ImGui::Render();
}

void application::render_scene_tree(entt::entity node) {
	ZoneScoped;
	if (node == entt::null)
		return;

	auto flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;
	if (selected_node == node) {
		flags |= ImGuiTreeNodeFlags_Selected;
	}

	ImGui::PushID(static_cast<int>(static_cast<ENTT_ID_TYPE>(node)));

	const auto sub_flags = flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Bullet;
	const bool is_empty = !registry.any_of<mesh_component_t>(node); // TODO: Update this when a light component or something gets added

	const auto& name = registry.get<name_component_t>(node).name;
	const auto& relationship = registry.get<node_component_t>(node);
	if (relationship.first_child != entt::null) {
		if (ImGui::TreeNodeEx(name.c_str(), flags)) {
			if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
				selected_node = node;
			}

			if (!is_empty) {
				// Draw the mesh information as an extra thing, we don't want to mark it as selected.
				ImGui::TreeNodeEx("Mesh", sub_flags & ~ImGuiTreeNodeFlags_Selected);
				if (ImGui::IsItemClicked()) {
					selected_node = node;
				}
			}

			auto curr = relationship.first_child;
			while (curr != entt::null) {
				render_scene_tree(curr);
				curr = registry.get<node_component_t>(curr).next;
			}
			ImGui::TreePop();
		}
	} else {
		if (ImGui::TreeNodeEx(name.c_str(), is_empty ? sub_flags : flags)) {
			if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
				selected_node = node;
			}

			if (!is_empty) {
				// Draw the mesh information as an extra thing, we don't want to mark it as selected.
				ImGui::TreeNodeEx("Mesh", sub_flags & ~ImGuiTreeNodeFlags_Selected);
				if (ImGui::IsItemClicked()) {
					selected_node = node;
				}
			}

			ImGui::TreePop();
		}
	}

	ImGui::PopID();
}
