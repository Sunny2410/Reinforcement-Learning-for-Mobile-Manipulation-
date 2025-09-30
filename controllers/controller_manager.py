"""
Controller manager for coordinating base, arm, and gripper controllers.
"""
import numpy as np
from typing import Dict, List, Optional
from .base_controller import BaseController
from .arm_controller import ArmController
from .gripper_controller import GripperController

class ControllerManager:
    def __init__(self, joints_base: List[str], joints_arm: List[str],
                 gripper_joints: List[str], kinematics, action_loader):
        """
        Initialize controller manager.
        
        Args:
            joints_base: List of base joint names
            joints_arm: List of arm joint names
            gripper_joints: List of gripper joint names
            kinematics: Kinematics instance
            action_loader: Action loader instance
        """
        self.action_loader = action_loader
        
        # Initialize controllers
        self.base_controller = BaseController(joints_base, kinematics)
        self.arm_controller = ArmController(joints_arm, kinematics)
        self.gripper_controller = GripperController(gripper_joints, kinematics)

        # Track active action
        self.active_action: Optional[str] = None

    def step(self, action_index: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Execute a new action only if no other action is active.
        
        Args:
            action_index: Index of the action to execute
            
        Returns:
            dict containing control commands if new action accepted,
            None if action ignored (because previous action still running)
        """
        # If still moving → ignore new action
        if self.is_any_moving():
            return None  

        # Get action details from loader
        action_type = self.action_loader.action_by_index(action_index)
        action_name = action_type.name
        step = self.action_loader.get_step(action_type)

        # Save active action
        self.active_action = action_name
        print(f"Starting new action: {action_name} with step {step}")
        # Initialize output dict
        output: Dict[str, np.ndarray] = {}

        # Route action to appropriate controller
        if "MOVE" in action_name or "TURN" in action_name:
            output.update(self.base_controller.step(action_name, step))
        elif "ARM" in action_name or "WRIST" in action_name:
            output.update(self.arm_controller.step(action_name, step))
        elif "GRIPPER" in action_name:
            output.update(self.gripper_controller.step(action_name, step))
        else:
            raise ValueError(f"Unknown action type: {action_name}")
            
        return output
    
    def update_control_loops(self) -> Dict[str, np.ndarray]:
        """
        Update all active control loops without setting new targets.
        Call this continuously in your simulation loop to maintain smooth movement.
        
        Returns:
            dict containing all control commands for active controllers
        """
        output: Dict[str, np.ndarray] = {}
        
        # Update arm controller if it's moving
        if not self.arm_controller.is_at_target():
            output.update(self.arm_controller.update_control_loop())
        
        # Update base controller if it's moving
        if not self.base_controller.is_at_target():
            output.update(self.base_controller.update_control_loop())
        
        # Update gripper controller if it's moving
        if not self.gripper_controller.is_at_target():
            output.update(self.gripper_controller.update_control_loop())
        
        # If everything stopped → clear active action
        if not self.is_any_moving():
            self.active_action = None
        
        return output
    
    def is_arm_moving(self) -> bool:
        """Check if arm is currently moving toward a target."""
        return not self.arm_controller.is_at_target()
    
    def is_base_moving(self) -> bool:
        """Check if base is currently moving toward a target."""
        return not self.base_controller.is_at_target()
    
    def is_gripper_moving(self) -> bool:
        """Check if gripper is currently moving toward a target."""
        return not self.gripper_controller.is_at_target()
    
    def is_any_moving(self) -> bool:
        """Check if any controller is currently moving."""
        return (not self.arm_controller.is_at_target() or 
                not self.base_controller.is_at_target() or 
                not self.gripper_controller.is_at_target())
    
    def stop_arm(self) -> None:
        """Stop arm movement."""
        self.arm_controller.stop()
    
    def stop_base(self) -> None:
        """Stop base movement."""
        self.base_controller.stop()
    
    def stop_gripper(self) -> None:
        """Stop gripper movement."""
        self.gripper_controller.stop()
        
    def stop_all(self) -> None:
        """Stop all movements."""
        self.arm_controller.stop()
        self.base_controller.stop()
        self.gripper_controller.stop()
        self.active_action = None
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all controllers."""
        return {
            "arm_moving": not self.arm_controller.is_at_target(),
            "base_moving": not self.base_controller.is_at_target(),
            "gripper_moving": not self.gripper_controller.is_at_target(),
            "any_moving": self.is_any_moving(),
            "active_action": self.active_action
        }

