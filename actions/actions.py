"""
Action definitions for indoor mobile manipulator robot.
This module provides discrete actions that can be used by RL agents
to control the robot's base, arm, wrist, and gripper.
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Union, Optional
import yaml


@dataclass
class ActionStep:
    """Represents a single action's step size and description."""
    step: float
    description: str


class ActionType(Enum):
    """Enumeration of all possible action types."""
    # Base movements
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    
    # Arm Cartesian movements
    ARM_FORWARD = auto()
    ARM_BACKWARD = auto()
    ARM_LEFT = auto()
    ARM_RIGHT = auto()
    ARM_UP = auto()
    ARM_DOWN = auto()
    
    # Wrist movements
    WRIST_ROLL_LEFT = auto()
    WRIST_ROLL_RIGHT = auto()
    WRIST_PITCH_UP = auto()
    WRIST_PITCH_DOWN = auto()
    WRIST_YAW_LEFT = auto()
    WRIST_YAW_RIGHT = auto()
    
    # Gripper actions
    GRIPPER_OPEN = auto()
    GRIPPER_CLOSE = auto()

    @property
    def is_base_movement(self) -> bool:
        """Check if action is a base movement."""
        return self in {
            ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD,
            ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT,
            ActionType.TURN_LEFT, ActionType.TURN_RIGHT
        }

    @property
    def is_arm_movement(self) -> bool:
        """Check if action is an arm movement."""
        return self in {
            ActionType.ARM_FORWARD, ActionType.ARM_BACKWARD,
            ActionType.ARM_LEFT, ActionType.ARM_RIGHT,
            ActionType.ARM_UP, ActionType.ARM_DOWN
        }

    @property
    def is_wrist_movement(self) -> bool:
        """Check if action is a wrist movement."""
        return self in {
            ActionType.WRIST_ROLL_LEFT, ActionType.WRIST_ROLL_RIGHT,
            ActionType.WRIST_PITCH_UP, ActionType.WRIST_PITCH_DOWN,
            ActionType.WRIST_YAW_LEFT, ActionType.WRIST_YAW_RIGHT
        }

    @property
    def is_gripper_action(self) -> bool:
        """Check if action is a gripper action."""
        return self in {ActionType.GRIPPER_OPEN, ActionType.GRIPPER_CLOSE}


class ActionConfig:
    """Manages action configurations loaded from YAML."""
    
    # Mapping từ YAML key sang ActionType
    YAML_TO_ACTION_TYPE = {
        # Base movements
        'move_forward': ActionType.MOVE_FORWARD,
        'move_backward': ActionType.MOVE_BACKWARD,
        'move_left': ActionType.MOVE_LEFT,
        'move_right': ActionType.MOVE_RIGHT,
        'turn_left': ActionType.TURN_LEFT,
        'turn_right': ActionType.TURN_RIGHT,
        # Arm movements
        'arm_forward': ActionType.ARM_FORWARD,
        'arm_backward': ActionType.ARM_BACKWARD,
        'arm_left': ActionType.ARM_LEFT,
        'arm_right': ActionType.ARM_RIGHT,
        'arm_up': ActionType.ARM_UP,
        'arm_down': ActionType.ARM_DOWN,
        # Wrist movements
        'wrist_roll_left': ActionType.WRIST_ROLL_LEFT,
        'wrist_roll_right': ActionType.WRIST_ROLL_RIGHT,
        'wrist_pitch_up': ActionType.WRIST_PITCH_UP,
        'wrist_pitch_down': ActionType.WRIST_PITCH_DOWN,
        'wrist_yaw_left': ActionType.WRIST_YAW_LEFT,
        'wrist_yaw_right': ActionType.WRIST_YAW_RIGHT,
        # Gripper
        'gripper_open': ActionType.GRIPPER_OPEN,
        'gripper_close': ActionType.GRIPPER_CLOSE,
    }
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize action configuration.
        
        Args:
            config_path: Path to the action_config.yaml file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.actions: Dict[ActionType, ActionStep] = self._initialize_actions()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_actions(self) -> Dict[ActionType, ActionStep]:
        """
        Initialize mapping between ActionType and ActionStep.
        Only loads actions that are present in the YAML config.
        """
        action_map = {}
        
        # Load base movements
        if 'base_movements' in self.config:
            for yaml_key, action_type in self.YAML_TO_ACTION_TYPE.items():
                if action_type.is_base_movement and yaml_key in self.config['base_movements']:
                    action_map[action_type] = ActionStep(**self.config['base_movements'][yaml_key])

        # Load arm movements
        if 'arm_cartesian' in self.config:
            for yaml_key, action_type in self.YAML_TO_ACTION_TYPE.items():
                if action_type.is_arm_movement and yaml_key in self.config['arm_cartesian']:
                    action_map[action_type] = ActionStep(**self.config['arm_cartesian'][yaml_key])

        # Load wrist movements
        if 'wrist_movements' in self.config:
            for yaml_key, action_type in self.YAML_TO_ACTION_TYPE.items():
                if action_type.is_wrist_movement and yaml_key in self.config['wrist_movements']:
                    action_map[action_type] = ActionStep(**self.config['wrist_movements'][yaml_key])

        # Load gripper actions
        if 'gripper' in self.config:
            for yaml_key, action_type in self.YAML_TO_ACTION_TYPE.items():
                if action_type.is_gripper_action and yaml_key in self.config['gripper']:
                    action_map[action_type] = ActionStep(**self.config['gripper'][yaml_key])

        return action_map

    def get_step_size(self, action_type: ActionType) -> Optional[float]:
        """
        Get step size for a specific action.
        Returns None if action is not configured.
        """
        action_step = self.actions.get(action_type)
        return action_step.step if action_step else None

    def get_description(self, action_type: ActionType) -> Optional[str]:
        """
        Get description for a specific action.
        Returns None if action is not configured.
        """
        action_step = self.actions.get(action_type)
        return action_step.description if action_step else None

    def get_all_actions(self) -> List[ActionType]:
        """Get list of all available actions loaded from config."""
        return list(self.actions.keys())

    def get_base_actions(self) -> List[ActionType]:
        """Get list of base movement actions that are configured."""
        return [action for action in self.actions.keys() if action.is_base_movement]

    def get_arm_actions(self) -> List[ActionType]:
        """Get list of arm movement actions that are configured."""
        return [action for action in self.actions.keys() if action.is_arm_movement]

    def get_wrist_actions(self) -> List[ActionType]:
        """Get list of wrist movement actions that are configured."""
        return [action for action in self.actions.keys() if action.is_wrist_movement]

    def get_gripper_actions(self) -> List[ActionType]:
        """Get list of gripper actions that are configured."""
        return [action for action in self.actions.keys() if action.is_gripper_action]
    
    def is_action_available(self, action_type: ActionType) -> bool:
        """Check if an action is available in the current configuration."""
        return action_type in self.actions