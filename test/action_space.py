from rl_mm.actions import ActionLoader

if __name__ == "__main__":
    # Khởi tạo ActionLoader (load config mặc định)
    actions = ActionLoader()

    all_actions = actions.all_actions()
    print(f"Tổng số hành động: {len(all_actions)}")
    index_map = {i: a for i, a in enumerate(all_actions)}

    print(actions.action_by_index(0).name)  # In ra hành động đầu tiên
    print("=== All Actions with index, step and description ===")
    for idx, action in index_map.items():
        step = actions.get_step(action)
        desc = actions.get_description(action)
        print(f"{idx}: {action.name}: step={step}, desc='{desc}'")

    print("\n=== Base Actions ===")
    for idx, action in enumerate(actions.base_actions()):
        print(f"{idx}: {action.name} -> step={actions.get_step(action)}")

    print("\n=== Arm Actions ===")
    for idx, action in enumerate(actions.arm_actions()):
        print(f"{idx}: {action.name} -> step={actions.get_step(action)}")

    print("\n=== Wrist Actions ===")
    for idx, action in enumerate(actions.wrist_actions()):
        print(f"{idx}: {action.name} -> step={actions.get_step(action)}")

    print("\n=== Gripper Actions ===")
    for idx, action in enumerate(actions.gripper_actions()):
        print(f"{idx}: {action.name} -> step={actions.get_step(action)}")
