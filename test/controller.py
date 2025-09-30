from rl_mm.controllers import ControllerManager
from rl_mm.actions import ActionLoader
from rl_mm.utils.kinematics import Kinematics
from rl_mm.robots import MobileSO101
from dm_control import mjcf

def apply_control_command(cmd):
    """Apply control command to the physics simulation"""
    if "arm_qpos" in cmd:
        for i, joint_name in enumerate(arm_joints):
            act_id = physics.model.name2id(joint_name,'joint')
            target_val = cmd["arm_qpos"][i]
            physics.data.ctrl[act_id] = target_val
            # Lấy control signal đang gửi

    # Base control (velocity)
    if "base_qvel" in cmd:
        for i, wheel_name in enumerate(wheel_names):
            act_id = physics.model.name2id(wheel_name,'joint')
            target_val = cmd["base_qvel"][i]
            physics.data.ctrl[act_id] = target_val

    # Gripper control (position)
    if "gripper_qpos" in cmd:
        for i, joint_name in enumerate(gripper_joints):
            act_id = physics.model.name2id(joint_name,'joint')
            target_val = cmd["gripper_qpos"][i]
            physics.data.ctrl[act_id] = target_val

# Initialize components
robot = MobileSO101()
physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
kinematics = Kinematics(robot, physics)
action_loader = ActionLoader()
wheel_names = [
        'fl_wheel_joint',
        'fr_wheel_joint', 
        'rl_wheel_joint',
        'rr_wheel_joint',
        ]
arm_joints = [
                'shoulder_pan',
                'shoulder_lift', 
                'elbow_flex',
                'wrist_flex',
                'wrist_roll',
            ]
gripper_joints = [
                'gripper',
            ]      
for i in range(90):      
    physics.step()

# Create controller manager
manager = ControllerManager(
    joints_base=wheel_names,
    joints_arm=arm_joints,
    gripper_joints=gripper_joints,
    kinematics=kinematics,
    action_loader=action_loader
)

cmd = manager.step(action_index=6)  # Ví dụ: ARM_FORWARD hoặc MOVE_FORWARD
apply_control_command(cmd)
    
    # Simulation step
physics.step()
physics.forward()
    # Tiếp tục điều khiển trong loop cho đến khi tất cả controllers đạt target
while manager.is_any_moving():
        # Cập nhật tất cả control loops
        cmd = manager.update_control_loops()
        
        if cmd:  # Chỉ apply nếu có command
            apply_control_command(cmd)
        
        # Simulation step
        physics.step()
        physics.forward()

        # In status mỗi 50 steps (optional)
        if physics.time() % 1.0 < 0.02:  # roughly every second
            status = manager.get_status()
            print(f"Status: {status}")
