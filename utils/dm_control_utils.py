import numpy as np

def apply_control_command(cmd, physics, arm_joints, wheel_names, gripper_joints):
    """Apply control command to the physics simulation"""
    if "arm_qpos" in cmd:
        arm_act_ids = np.array([physics.model.name2id(name, 'actuator') for name in arm_joints])
        physics.data.ctrl[arm_act_ids] = np.array(cmd["arm_qpos"])

    if "base_qvel" in cmd:
        wheel_act_ids = np.array([physics.model.name2id(name, 'actuator') for name in wheel_names])
        physics.data.ctrl[wheel_act_ids] = np.array(cmd["base_qvel"])

    if "gripper_qpos" in cmd:
        gripper_act_ids = np.array([physics.model.name2id(name, 'actuator') for name in gripper_joints])
        physics.data.ctrl[gripper_act_ids] = np.array(cmd["gripper_qpos"])

def wrap_joint(val, low, high):
    """
    Wrap val về trong khoảng [low, high]
    """
    range_width = high - low
    while val < low:
        val += range_width
    while val > high:
        val -= range_width
    return val

