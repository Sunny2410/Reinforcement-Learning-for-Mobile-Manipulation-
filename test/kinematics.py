import os
import numpy as np
from dm_control import mjcf
from rl_mm.robots import MobileSO101
from rl_mm.props import Primitive
from rl_mm.arena import StandardArena
from rl_mm.utils import Kinematics  # dùng class mới

def print_joint_info(physics, show_free_joints=False):
    print(f"\n=== Joint Information (Total: {physics.model.njnt}) ===")
    real_joints, free_joints = [], []
    
    for joint_id in range(physics.model.njnt):
        name = physics.model.id2name(joint_id, 'joint')
        jtype = physics.model.jnt_type[joint_id]
        qpos_addr = physics.model.jnt_qposadr[joint_id]

        ndof = 1
        if jtype == 0: ndof = 6
        elif jtype == 1: ndof = 4
        elif jtype in [2, 3]: ndof = 1

        qpos_val = physics.data.qpos[qpos_addr] if ndof == 1 else physics.data.qpos[qpos_addr:qpos_addr+ndof]
        joint_info = {'id': joint_id, 'name': name, 'qpos': qpos_val}
        if name: real_joints.append(joint_info)
        else: free_joints.append(joint_info)
    
    print("📍 REAL JOINTS:")
    for joint in real_joints:
        print(f"  {joint['name']}: {joint['qpos']}")
    if show_free_joints:
        print("🔓 FREE JOINTS:")
        for joint in free_joints:
            print(f"  free_joint: {joint['qpos']}")

def main():
    # --- Setup arena ---
    arena = StandardArena()
    box = Primitive(type="box", size=[0.02,0.02,0.02], pos=[0.5,0,0], rgba=[1,0,0,1])
    arena.attach(box.mjcf_model, pos=[0.5,0,0])

    # --- Add robot ---
    robot = MobileSO101()
    arena.attach(robot.mjcf_model, pos=[0,0,0.0]).add('freejoint')

    # --- Physics ---
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    for _ in range(100):  # Warmup physics
        physics.step()
        physics.forward()

    print_joint_info(physics, show_free_joints=False)

    # --- Kinematics (class mới) ---
    kin = Kinematics(robot, physics)

    # --- Forward Kinematics ---
    fk_before = kin.forward_kinematics()
    print("\n=== FK before IK ===")
    print("EEF world pos:", fk_before["eef_world_pos"], "quat:", fk_before["eef_world_quat"])

    # --- Inverse Kinematics: dịch EEF 5cm theo X ---
    target_pos = fk_before["eef_world_pos"] + np.array([0.05, 0.0, 0.0])
    target_quat = None  # giữ orientation hiện tại

    print("\n=== IK target ===")
    print("Target position:", target_pos)

    # Lấy tên joints từ prefix mới
    joint_names = [
        'scene/shoulder_pan',
        'scene/shoulder_lift',
        'scene/elbow_flex',
        'scene/wrist_flex',
        'scene/wrist_roll'
    ]

    qpos_ik = kin.inverse_kinematics(target_pos, target_quat, joint_names=joint_names)
    print(qpos_ik)
    if qpos_ik is not None:
        print("✅ IK solution found:", qpos_ik)
        physics.forward()
        fk_after = kin.forward_kinematics()
        print("EEF world pos (after IK):", fk_after["eef_world_pos"])
        error = np.linalg.norm(fk_after["eef_world_pos"] - target_pos)
        print(f"Position error: {error:.6f}")
    else:
        print("❌ IK failed.")

    print_joint_info(physics, show_free_joints=False)

if __name__ == "__main__":
    main()
