"""
Arm controller for manipulator.
Implements IK-based PI-controller with integrated safety checks and correct coordinate frame transformations.
"""

import numpy as np
from typing import Dict, List
from scipy.spatial.transform import Rotation # NEW: Import for coordinate transformations
from rl_mm.utils.transform_utils import quat_to_euler, euler_to_quat

class ArmController:
    def __init__(self, joints_arm: List[str], kinematics, Kp: float = 1, Ki: float = 0.001, Kd: float = 0.5, qpos_tol: float = 0.04, floor_safety_offset: float = 0.08):
        """
        Initialize arm controller.

        Args:
            joints_arm: List of arm joint names
            kinematics: Kinematics instance
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            qpos_tol: Joint position tolerance in radians
            floor_safety_offset: Minimum allowed height for the end-effector from Z=0.
        """
        self.joints_arm = joints_arm
        self.kinematics = kinematics
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._prev_error = np.zeros(len(joints_arm))
        self.qpos_tol = qpos_tol

        # Control state
        self._qpos_target = None
        self._is_moving = False
        self._max_step_size = 1  # radians
        self._integral_error = np.zeros(len(joints_arm))

        # --- Safety parameters ---
        self.floor_safety_offset = floor_safety_offset
        try:
            self._joint_ids = [self.kinematics.physics.model.name2id(name, 'joint') for name in self.joints_arm]
            self.joint_limits = self.kinematics.physics.model.jnt_range[self._joint_ids]
        except Exception as e:
            print(f"Warning: Could not get joint limits from model. Safety checks may be affected. Error: {e}")
            self.joint_limits = None


    def set_target(self, action_type: str, step: float) -> bool:
        """
        Set new target delta pose based on step, with safety checks and correct coordinate frame transformations.
        """
        # Initialize target deltas in the LOCAL end-effector frame
        target_pos_delta_local = np.array([0.0, 0.0, 0.0])
        target_rpy_delta = np.array([0.0, 0.0, 0.0])

        # Get current end-effector pose in the WORLD frame
        fk = self.kinematics.forward_kinematics()
        curr_pos_world = fk["eef_world_pos"]
        curr_quat_world = fk["eef_world_quat"] # Quaternion is [w, x, y, z]
        
        # Convert step values
        if "WRIST" in action_type:
            step = np.deg2rad(step)
        else:
            step = step / 100.0 # cm -> m

        # Define ARM deltas in the LOCAL frame
        if action_type == "ARM_FORWARD": target_pos_delta_local[0] += step
        elif action_type == "ARM_BACKWARD": target_pos_delta_local[0] -= step
        elif action_type == "ARM_LEFT": target_pos_delta_local[1] += step
        elif action_type == "ARM_RIGHT": target_pos_delta_local[1] -= step
        elif action_type == "ARM_UP": target_pos_delta_local[2] += step
        elif action_type == "ARM_DOWN": target_pos_delta_local[2] -= step
        # Define WRIST deltas (rotations)
        elif "WRIST" in action_type:
            if action_type == "WRIST_ROLL_LEFT": target_rpy_delta[0] += step
            elif action_type == "WRIST_ROLL_RIGHT": target_rpy_delta[0] -= step
            elif action_type == "WRIST_PITCH_UP": target_rpy_delta[1] += step
            elif action_type == "WRIST_PITCH_DOWN": target_rpy_delta[1] -= step
            elif action_type == "WRIST_YAW_LEFT": target_rpy_delta[2] += step
            elif action_type == "WRIST_YAW_RIGHT": target_rpy_delta[2] -= step

        # --- CORRECTED LOGIC: SAFETY CHECK 1 (Floor Collision) ---
        # 1. Create a rotation object from the current orientation.
        #    Note: Scipy expects quaternion as [x, y, z, w], while MuJoCo gives [w, x, y, z].
        current_rotation = Rotation.from_quat([curr_quat_world[1], curr_quat_world[2], curr_quat_world[3], curr_quat_world[0]])
        
        # 2. Rotate the local delta vector to transform it into the world frame.
        target_pos_delta_world = current_rotation.apply(target_pos_delta_local)
        
        # 3. Calculate the absolute target position in the world frame. Now this is a valid addition.
        absolute_target_pos_world = curr_pos_world + target_pos_delta_world

        if absolute_target_pos_world[2] < self.floor_safety_offset:
            # print(f"⚠️  Arm Safety Check: Target Z ({absolute_target_pos_world[2]:.3f}) is below floor offset ({self.floor_safety_offset}). Action invalid.")
            self._is_moving = False
            return False

        # --- Compute IK ---
        # IMPORTANT: We pass the ORIGINAL LOCAL deltas to your IK function,
        # assuming your `inverse_kinematics` is designed to interpret a delta in the local frame.
        new_qpos_target = self.kinematics.inverse_kinematics(
            target_pos=target_pos_delta_local,
            target_quat=target_rpy_delta,
            joint_names=self.joints_arm
        )

        if new_qpos_target is not None:
            # This logic is kept from your original code.
            new_qpos_target = np.delete(new_qpos_target, 3) 

            # --- SAFETY CHECK 2: Joint Limits (No changes needed here) ---
            if self.joint_limits is not None:
                for i, q_target in enumerate(new_qpos_target):
                    q_min, q_max = self.joint_limits[i]
                    if not (q_min - 1e-4 <= q_target <= q_max + 1e-4):
                        # print(f"⚠️  Arm Safety Check: Target qpos for '{self.joints_arm[i]}' ({q_target:.3f}) exceeds limits [{q_min:.3f}, {q_max:.3f}]. Action invalid.")
                        self._is_moving = False
                        return False
            
            # If all checks pass, set the new target
            self._qpos_target = new_qpos_target
            self._is_moving = True
            self._integral_error[:] = 0.0
            return True
        else:
            # IK failed
            self._is_moving = False
            return False

    def step(self, action_type: str, step: float) -> Dict[str, np.ndarray]:
        """Execute a control step."""
        if not self._is_moving or self._qpos_target is None:
            target_set = self.set_target(action_type, step)
            if not target_set:
                curr_qpos = np.array([self.kinematics.physics.named.data.qpos[j] for j in self.joints_arm]).ravel()
                return {"arm_qpos": curr_qpos}
        return self.update_control_loop()

    def update_control_loop(self) -> Dict[str, np.ndarray]:
        """Update PID control loop toward target."""
        curr_qpos = np.array([self.kinematics.physics.named.data.qpos[j] for j in self.joints_arm]).ravel()
        if not self._is_moving or self._qpos_target is None:
            return {"arm_qpos": curr_qpos}
        qpos_error = self._qpos_target - curr_qpos
        if np.linalg.norm(qpos_error) < self.qpos_tol:
            self._is_moving = False
            self._integral_error[:] = 0.0
            self._prev_error[:] = 0.0
            return {"arm_qpos": self._qpos_target.copy()}
        self._integral_error += qpos_error
        qpos_derivative = qpos_error - self._prev_error
        self._prev_error = qpos_error.copy()
        qpos_cmd_raw = curr_qpos + self.Kp * qpos_error + self.Ki * self._integral_error + self.Kd * qpos_derivative
        qpos_step = qpos_cmd_raw - curr_qpos
        step_norms = np.abs(qpos_step)
        max_step_mask = step_norms > self._max_step_size
        if np.any(max_step_mask):
            qpos_step[max_step_mask] = np.sign(qpos_step[max_step_mask]) * self._max_step_size
        qpos_cmd = curr_qpos + qpos_step
        return {"arm_qpos": qpos_cmd}

    def is_at_target(self) -> bool:
        return not self._is_moving

    def stop(self) -> None:
        self._is_moving = False
        self._qpos_target = None
        self._integral_error[:] = 0.0

def get_joint_qpos_address(physics, joint_name):
    try:
        joint_id = physics.model.name2id(joint_name, 'joint')
        return physics.model.jnt_qposadr[joint_id]
    except:
        return None

