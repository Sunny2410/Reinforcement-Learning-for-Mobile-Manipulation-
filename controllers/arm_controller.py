"""
Arm controller for manipulator.
Implements IK-based PI-controller.
"""

import numpy as np
from typing import Dict, List
from rl_mm.utils.transform_utils import quat_to_euler, euler_to_quat
class ArmController:
    def __init__(self, joints_arm: List[str], kinematics, Kp: float = 1, Ki: float = 0.001, Kd: float = 0.5, qpos_tol: float = 0.04):
        """
        Initialize arm controller.

        Args:
            joints_arm: List of arm joint names
            kinematics: Kinematics instance
            Kp: Proportional gain
            Ki: Integral gain
            qpos_tol: Joint position tolerance in radians
        """
        self.joints_arm = joints_arm
        self.kinematics = kinematics
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._prev_error = np.zeros(len(joints_arm))  # previous error for derivative
        self.qpos_tol = qpos_tol

        # Target pose for end-effector
        self._target_pos = None
        self._target_quat = None
        
        # Control state
        self._qpos_target = None
        self._is_moving = False
        self._max_step_size = 0.1  # radians
        self._integral_error = np.zeros(len(joints_arm))  # integral error

    def set_target(self, action_type: str, step: float) -> bool:
        """Set new target delta pose based on step."""
        # Initialize target deltas
        target_pos = np.array([0.0, 0.0, 0.0])
        target_rpy = np.array([0.0, 0.0, 0.0])

        # Convert step for wrist rotations
        if "WRIST" in action_type:
            step = np.deg2rad(step)
        else:
            step = step / 100.0  # cm -> m

        # ARM deltas
        if action_type == "ARM_FORWARD":
            target_pos[0] += step
        elif action_type == "ARM_BACKWARD":
            target_pos[0] -= step
        elif action_type == "ARM_LEFT":
            target_pos[1] += step
        elif action_type == "ARM_RIGHT":
            target_pos[1] -= step
        elif action_type == "ARM_UP":
            target_pos[2] += step
        elif action_type == "ARM_DOWN":
            target_pos[2] -= step

        # WRIST deltas
        elif "WRIST" in action_type:
            if action_type == "WRIST_ROLL_LEFT":
                target_rpy[0] += step
            elif action_type == "WRIST_ROLL_RIGHT":
                target_rpy[0] -= step
            elif action_type == "WRIST_PITCH_UP":
                target_rpy[1] += step
            elif action_type == "WRIST_PITCH_DOWN":
                target_rpy[1] -= step
            elif action_type == "WRIST_YAW_LEFT":
                target_rpy[2] += step
            elif action_type == "WRIST_YAW_RIGHT":
                target_rpy[2] -= step

        # Assign to class variables
        self._target_pos = target_pos
        self._target_rpy = target_rpy 

        # Compute IK
        new_qpos_target = self.kinematics.inverse_kinematics(
            target_pos=self._target_pos,
            target_quat=self._target_rpy,
            joint_names=self.joints_arm
        )
        if new_qpos_target is not None:
            # self._qpos_target = np.array([
            #     new_qpos_target[get_joint_qpos_address(self.kinematics.physics, j)]
            #     for j in self.joints_arm
            # ])
            self._qpos_target = new_qpos_target[:-1]  # bỏ phần tử cuối cùng            
            self._is_moving = True
            self._integral_error[:] = 0.0  # reset integral when new target
            return True
        else:
            print("IK failed, target not updated")
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
        if not self._is_moving or self._qpos_target is None:
            curr_qpos = np.array([self.kinematics.physics.named.data.qpos[j] for j in self.joints_arm]).ravel()
            return {"arm_qpos": curr_qpos}

        curr_qpos = np.array([self.kinematics.physics.named.data.qpos[j] for j in self.joints_arm]).ravel()
        qpos_error = self._qpos_target - curr_qpos

        if np.linalg.norm(qpos_error) < self.qpos_tol:
            print("Target reached!")
            self._is_moving = False
            self._integral_error[:] = 0.0
            self._prev_error[:] = 0.0
            return {"arm_qpos": self._qpos_target.copy()}

        # Integral term
        self._integral_error += qpos_error

        # Derivative term
        qpos_derivative = qpos_error - self._prev_error
        self._prev_error = qpos_error.copy()

        # PID controller
        qpos_cmd_raw = curr_qpos + self.Kp * qpos_error + self.Ki * self._integral_error + self.Kd * qpos_derivative

        # Limit step size
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
        print("Arm movement stopped")


def get_joint_qpos_address(physics, joint_name):
    try:
        joint_id = physics.model.name2id(joint_name, 'joint')
        return physics.model.jnt_qposadr[joint_id]
    except:
        return None
