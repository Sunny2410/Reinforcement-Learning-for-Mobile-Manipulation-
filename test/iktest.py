import numpy as np
from lerobot.model.kinematics import RobotKinematics  # giả sử bạn lưu code trên robot_kinematics.py

# --- 1. Khởi tạo solver ---
robot = RobotKinematics(
    urdf_path="/home/sunny24/rl_mm/asset/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
)

ee_pose = robot.forward_kinematics([0,0,0,0,0])
print("Initial EE pose:", ee_pose)

# --- 2. Đặt trạng thái khớp hiện tại (độ) ---
current_joints = np.array([-1.96148536e-16, 5.53460030e-01, -7.90426904e-01, 2.36966875e-01, -1.55022441e-15])  # ví dụ các góc khởi tạo
current_joints = np.rad2deg(current_joints)  # Chuyển sang radian
# --- 3. Tính Forward Kinematics để lấy pose hiện tại ---
ee_pose = robot.forward_kinematics(current_joints)
print("Current EE pose:", ee_pose)