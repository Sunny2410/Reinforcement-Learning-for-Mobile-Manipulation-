"""
Base controller for mecanum mobile base.
Implements P-controller for position and orientation control with continuous loop.
"""

import numpy as np
from typing import Dict, List, Optional

class BaseController:
    def __init__(self, joints_base: List[str], kinematics, Kp: float = 500):
        """
        Initialize base controller.

        Args:
            joints_base: List of base joint names (wheels)
            kinematics: Kinematics instance
            Kp: Proportional gain for P-controller
        """
        self.joints_base = joints_base
        self.kinematics = kinematics
        self.Kp = Kp
        
        # Thresholds for position and orientation
        self.pos_threshold = 0.01  # 1cm
        self.yaw_threshold = np.deg2rad(0.1)  # 2 degrees
        
        # Target tracking
        self._target_pos = None  # [x, y]
        self._target_yaw = None  # rad
        self._is_moving = False  # Flag to indicate if base is moving
        self._max_vel_linear = 20  # Maximum linear velocity (m/s)
        self._max_vel_angular = 20  # Maximum angular velocity (rad/s)

    def set_target(self, action_type: str, step: float) -> bool:
        """
        Set new target pose based on action.
        
        Args:
            action_type: Type of action (MOVE_*, TURN_*)
            step: Step size for the action (cm or deg)
            
        Returns:
            bool: True if target was set successfully
        """
        # Get current base pose from kinematics
        fk = self.kinematics.forward_kinematics()
        curr_pos = fk["base_world_pos"][:2]  # only x,y
        curr_quat = fk["base_world_quat"]
        qx, qy, qz, qw = curr_quat

        # Tính yaw (rotation quanh trục Z)
        curr_yaw = np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy*qy + qz*qz)
        )
        # Initialize target if not set
        if self._target_pos is None:
            self._target_pos = curr_pos.copy()
        if self._target_yaw is None:
            self._target_yaw = curr_yaw

        # Convert step to meters/radians
        if "TURN" in action_type:
            step = np.deg2rad(step)  # convert to radians
        else:
            step = step / 100.0  # convert cm to meters

        # Update target based on action type
        if action_type == "MOVE_FORWARD":
            self._target_pos[0] += step * np.cos(curr_yaw)
            self._target_pos[1] += step * np.sin(curr_yaw)
        elif action_type == "MOVE_BACKWARD":
            self._target_pos[0] -= step * np.cos(curr_yaw)
            self._target_pos[1] -= step * np.sin(curr_yaw)
        elif action_type == "MOVE_LEFT":
            self._target_pos[0] -= step * np.sin(curr_yaw)
            self._target_pos[1] += step * np.cos(curr_yaw)
        elif action_type == "MOVE_RIGHT":
            self._target_pos[0] += step * np.sin(curr_yaw)
            self._target_pos[1] -= step * np.cos(curr_yaw)
        elif action_type == "TURN_LEFT":
            self._target_yaw += step
        elif action_type == "TURN_RIGHT":
            self._target_yaw -= step

        # Normalize target yaw to [-pi, pi]
        self._target_yaw = (self._target_yaw + np.pi) % (2 * np.pi) - np.pi
        self._is_moving = True
        return True

    def step(self, action_type: str, step: float) -> Dict[str, np.ndarray]:
        """
        Execute a control step with new action.
        
        Args:
            action_type: Type of action (MOVE_*, TURN_*)
            step: Step size for the action

        Returns:
            dict containing wheel commands
        """
        # Set new target
        self.set_target(action_type, step)
        
        # Execute control step toward target
        return self.update_control_loop()

    def update_control_loop(self) -> Dict[str, np.ndarray]:
        """
        Update control loop without setting new target.
        Use this for continuous control when moving toward existing target.
        
        Returns:
            dict containing wheel commands
        """
        if not self._is_moving or self._target_pos is None or self._target_yaw is None:
            # No active target, stop wheels
            return {"base_qvel": np.zeros(len(self.joints_base))}
        
        # Get current base pose
        fk = self.kinematics.forward_kinematics()
        curr_pos = fk["base_world_pos"][:2]  # only x,y
        curr_quat = fk["base_world_quat"]
        curr_pos1 = fk["eef_world_pos"]
        curr_quat1 = fk["eef_world_quat"]
        print(curr_pos1, curr_quat1)
        # curr_quat = [qx, qy, qz, qw] từ MuJoCo
        qx, qy, qz, qw = curr_quat

        # Tính yaw (rotation quanh trục Z)
        curr_yaw = np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy*qy + qz*qz)
        )
        print(f"Current base pose - pos: {curr_pos}, yaw: {np.rad2deg(curr_yaw):.1f}°")
        # Calculate errors
        pos_error = self._target_pos - curr_pos
        yaw_error = (self._target_yaw - curr_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # Check if we've reached the target
        pos_dist = np.linalg.norm(pos_error)
        yaw_dist = abs(yaw_error)
        
        if pos_dist < self.pos_threshold and yaw_dist < self.yaw_threshold:
            print("Base target reached!")
            self._is_moving = False
            return {"base_qvel": np.zeros(len(self.joints_base))}

        # P-controller
        # Transform errors to base frame for motion planning
        cos_yaw = np.cos(curr_yaw)
        sin_yaw = np.sin(curr_yaw)
        rot_mat = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
        rel_error = rot_mat @ pos_error

        # Calculate velocities with P-controller
        v_x = self.Kp * rel_error[0]  # forward velocity
        v_y = self.Kp * rel_error[1]  # lateral velocity
        omega = self.Kp * yaw_error   # angular velocity

        # Limit velocities
        v_x = np.clip(v_x, -self._max_vel_linear, self._max_vel_linear)
        v_y = np.clip(v_y, -self._max_vel_linear, self._max_vel_linear)
        omega = np.clip(omega, -self._max_vel_angular, self._max_vel_angular)

        # Compute wheel speeds using kinematics
        wheel_cmd = self.kinematics.compute_wheel_speeds(
            target_rel_pos=np.array([v_x, v_y]),
            target_rel_yaw=omega,
            current_rel_pos=np.zeros(2),
            current_rel_yaw=0
        )

        print(f"Base errors - pos: {pos_dist:.3f}m, yaw: {np.rad2deg(yaw_dist):.1f}°")
        print(f"Base velocities - vx: {v_x:.3f}, vy: {v_y:.3f}, omega: {omega:.3f}")

        return {"base_qvel": wheel_cmd}

    def is_at_target(self) -> bool:
        """
        Check if base has reached the target position.
        
        Returns:
            bool: True if at target or not moving, False if still moving toward target
        """
        return not self._is_moving

    def stop(self) -> None:
        """
        Stop base movement and clear target.
        """
        self._is_moving = False
        self._target_pos = None
        self._target_yaw = None
        print("Base movement stopped")