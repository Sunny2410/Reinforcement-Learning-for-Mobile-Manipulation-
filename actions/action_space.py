import os
from pathlib import Path
from rl_mm.actions.actions import ActionType, ActionStep, ActionConfig

# Đường dẫn mặc định tới file action_config.yaml
_DEFAULT_ACTION_CONFIG = os.path.join(
    os.path.dirname(__file__),
    '../configs/action_config.yaml',
)

class ActionLoader:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = _DEFAULT_ACTION_CONFIG
        self.config = ActionConfig(config_path)
        self.ActionType = ActionType
        self.ActionStep = ActionStep

        # Map index → ActionType
        self.index_map = {i: a for i, a in enumerate(self.config.get_all_actions())}

    def action_by_index(self, idx: int) -> ActionType:
        return self.index_map[idx]

    def get_step(self, action_type: ActionType) -> float:
        return self.config.get_step_size(action_type)

    def get_description(self, action_type: ActionType) -> str:
        return self.config.get_description(action_type)

    def all_actions(self):
        return self.config.get_all_actions()

    def base_actions(self):
        return self.config.get_base_actions()

    def arm_actions(self):
        return self.config.get_arm_actions()

    def wrist_actions(self):
        return self.config.get_wrist_actions()

    def gripper_actions(self):
        return self.config.get_gripper_actions()
