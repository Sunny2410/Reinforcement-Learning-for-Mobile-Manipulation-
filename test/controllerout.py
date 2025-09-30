import os
import imageio
from rl_mm.controllers import ControllerManager
from rl_mm.actions import ActionLoader
from rl_mm.utils.kinematics import Kinematics
from rl_mm.robots import MobileSO101
from dm_control import mjcf
import numpy as np

def apply_control_command(cmd, physics):
    """Apply control command to the physics simulation"""

    # Arm control
    if "arm_qpos" in cmd:
        arm_act_ids = np.array([physics.model.name2id(name, 'actuator') for name in arm_joints])
        physics.data.ctrl[arm_act_ids] = np.array(cmd["arm_qpos"])

    # Base control
    if "base_qvel" in cmd:
        wheel_act_ids = np.array([physics.model.name2id(name, 'actuator') for name in wheel_names])
        physics.data.ctrl[wheel_act_ids] = np.array(cmd["base_qvel"])

    # Gripper control
    if "gripper_qpos" in cmd:
        gripper_act_ids = np.array([physics.model.name2id(name, 'actuator') for name in gripper_joints])
        physics.data.ctrl[gripper_act_ids] = np.array(cmd["gripper_qpos"])


# ------------------ INIT ------------------
robot = MobileSO101()
physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
kinematics = Kinematics(robot, physics)
action_loader = ActionLoader()

wheel_names = ['scene/fl_wheel_joint', 'scene/fr_wheel_joint', 'scene/rl_wheel_joint', 'scene/rr_wheel_joint']
arm_joints = ['scene/shoulder_pan', 'scene/shoulder_lift', 'scene/elbow_flex', 'scene/wrist_flex', 'scene/wrist_roll']
gripper_joints = ['scene/gripper']

for _ in range(90):
    physics.step()

manager = ControllerManager(
    joints_base=wheel_names,
    joints_arm=arm_joints,
    gripper_joints=gripper_joints,
    kinematics=kinematics,
    action_loader=action_loader
)

# ------------------ VIDEO FRAME ------------------
frames = []
fps = 30
max_time = 10.0  # giây
max_steps = int(fps * max_time)

# Bước đầu: gọi 1 command ví dụ
cmd = manager.step(action_index=6)
apply_control_command(cmd, physics)

physics.step()
physics.forward()
frames.append(physics.render(height=480, width=640, camera_id=-1))

import os
import imageio
import cv2

# ------------------ LOOP SIMULATION ------------------
step_count = 1
final_frame_captured = False
rest_steps = 0
rest_threshold = 5

while step_count < max_steps:
    if manager.is_any_moving():
        cmd = manager.update_control_loops()
        if cmd:
            apply_control_command(cmd, physics)
        rest_steps = 0
    else:
        rest_steps += 1
        # Capture final frame if stopped for enough steps
        if not final_frame_captured and rest_steps >= rest_threshold:
            final_frame = physics.render(height=480, width=640, camera_id=-1)
            # Lưu ảnh PNG riêng
            os.makedirs("rl_mm/test", exist_ok=True)
            cv2.imwrite("rl_mm/test/final_position.png", final_frame)
            print("📷 Final frame saved as PNG.")
            final_frame_captured = True

    physics.step()
    physics.forward()

    # Render frame **mỗi bước** để tạo video mượt
    frame = physics.render(height=480, width=640, camera_id=-1)
    frames.append(frame)

    # Optional: print status mỗi giây
    if physics.time() % 1.0 < 0.02:
        status = manager.get_status()
        print(f"Status: {status}")

    step_count += 1

# ------------------ SAVE VIDEO ------------------
video_path = "rl_mm/test/controller_simulation_10s.mp4"
os.makedirs(os.path.dirname(video_path), exist_ok=True)
imageio.mimwrite(video_path, frames, fps=fps, quality=8)
print(f"🎥 Simulation video saved to: {video_path}")
