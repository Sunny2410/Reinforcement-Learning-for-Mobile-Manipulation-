"""
Gripper controller for manipulator.
Implements P-controller for gripper width with continuous loop.
"""
import numpy as np
from typing import Dict, List, Optional

class GripperController:
    def __init__(self, gripper_joints: List[str], kinematics, Kp: float = 0.5):
        """
        Initialize gripper controller.
        
        Args:
            gripper_joints: List of gripper joint names
            kinematics: Kinematics instance
            Kp: Proportional gain for P-controller
        """
        self.gripper_joints = gripper_joints
        self.kinematics = kinematics
        self.Kp = Kp
        
        # Target tracking
        self._target_width = None
        self._is_moving = False  # Flag to indicate if gripper is moving
        self.width_tolerance = 0.001  # 1mm tolerance
        self._max_step_size = 1  # Maximum width change per step
        
        # Get joint limits
        self._joint_ranges = [(self.kinematics.physics.named.model.jnt_range[j][0],
                              self.kinematics.physics.named.model.jnt_range[j][1])
                             for j in self.gripper_joints]
        self._min_width = max(r[0] for r in self._joint_ranges)
        self._max_width = min(r[1] for r in self._joint_ranges)

    def set_target(self, action_type: str, step: float) -> bool:
        """
        Set new target gripper width based on action.
        
        Args:
            action_type: Type of action (GRIPPER_OPEN/CLOSE)
            step: Step size for the action (cm)
            
        Returns:
            bool: True if target was set successfully
        """
        # Get current gripper width
        curr_width = np.mean([self.kinematics.physics.named.data.qpos[j]
                             for j in self.gripper_joints])
        
        # Initialize target if not set
        if self._target_width is None:
            self._target_width = curr_width

        # Convert step (assuming it's already in appropriate units)
        step_m = step  # Assuming step is already in meters
        
        print(f"Setting gripper target: {action_type}, step: {step_m}")

        # Update target based on action
        if action_type == "GRIPPER_OPEN":
            self._target_width += step_m
        elif action_type == "GRIPPER_CLOSE":
            self._target_width -= step_m

        # Clamp target width to joint limits
        self._target_width = np.clip(self._target_width, self._min_width, self._max_width)
        
        self._is_moving = True
        print(f"New gripper target width: {self._target_width:.4f}m (limits: {self._min_width:.4f} - {self._max_width:.4f})")
        return True

    def step(self, action_type: str, step: float) -> Dict[str, np.ndarray]:
        """
        Execute a control step with new action.
        
        Args:
            action_type: Type of action (GRIPPER_OPEN/CLOSE)
            step: Step size for the action
            
        Returns:
            dict containing gripper joint commands
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
            dict containing gripper joint commands
        """
        if not self._is_moving or self._target_width is None:
            # No active target, maintain current position
            curr_qpos = np.array([self.kinematics.physics.named.data.qpos[j]
                                 for j in self.gripper_joints])
            return {"gripper_qpos": curr_qpos}

        # Get current joint positions
        curr_qpos = np.array([self.kinematics.physics.named.data.qpos[j]
                             for j in self.gripper_joints])
        curr_width = np.mean(curr_qpos)

        # Check if we've reached the target
        width_error = abs(self._target_width - curr_width)
        
        if width_error < self.width_tolerance:
            print("Gripper target reached!")
            self._is_moving = False
            return {"gripper_qpos": np.full_like(curr_qpos, self._target_width)}

        # Calculate desired joint positions (symmetric gripper)
        qpos_target = np.full_like(curr_qpos, self._target_width)
        
        # P-controller for joint positions
        qpos_error = qpos_target - curr_qpos
        qpos_cmd_raw = curr_qpos + self.Kp * qpos_error
        
        # Limit step size to prevent large jumps
        qpos_step = qpos_cmd_raw - curr_qpos
        step_norms = np.abs(qpos_step)
        max_step_mask = step_norms > self._max_step_size
        
        if np.any(max_step_mask):
            # Scale down steps that are too large
            qpos_step[max_step_mask] = np.sign(qpos_step[max_step_mask]) * self._max_step_size
            
        qpos_cmd = curr_qpos + qpos_step

        print(f"Gripper - current: {curr_width:.4f}m, target: {self._target_width:.4f}m, error: {width_error:.4f}m")

        return {"gripper_qpos": qpos_cmd}

    def is_at_target(self) -> bool:
        """
        Check if gripper has reached the target width.
        
        Returns:
            bool: True if at target or not moving, False if still moving toward target
        """
        return not self._is_moving

    def stop(self) -> None:
        """
        Stop gripper movement and clear target.
        """
        self._is_moving = False
        self._target_width = None
        print("Gripper movement stopped")