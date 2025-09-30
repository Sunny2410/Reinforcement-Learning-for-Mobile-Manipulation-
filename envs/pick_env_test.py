import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import mjcf
import mujoco.viewer
from rl_mm.controllers import ControllerManager
from rl_mm.actions import ActionLoader
from rl_mm.utils.kinematics import Kinematics
from rl_mm.robots import MobileSO101
from rl_mm.props import Primitive
from rl_mm.arena import StandardArena

class SO101Arm2(gym.Env):
    """Gymnasium environment with StandardArena, robot, prop, and controller"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None, base_coeff=0.1, reach_bonus=10.0, reach_threshold=0.05):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self._render_mode = render_mode

        # ---------------- ARENA ----------------
        self.arena = StandardArena()

        # Add a free box into arena
        self.box = Primitive(type="box", size=[0.02,0.02,0.02], rgba=[1,0,0,1])
        self.arena.attach_free(self.box.mjcf_model, pos=[0.5,0,0.01])

        # Add robot to arena
        self.robot = MobileSO101()
        self.arena.attach_free(self.robot.mjcf_model, pos=[0,0,0]).add
        print("Load robot successful:", self.robot)

        # Build physics from arena MJCF
        self.physics = mjcf.Physics.from_mjcf_model(self.arena.mjcf_model)

        # ---------------- KINEMATICS & CONTROLLER ----------------
        self.kinematics = Kinematics(self.robot, self.physics)
        self.action_loader = ActionLoader()

        self.wheel_names = ['scene/fl_wheel_joint', 'scene/fr_wheel_joint', 'scene/rl_wheel_joint', 'scene/rr_wheel_joint']
        self.arm_joints = ['scene/shoulder_pan', 'scene/shoulder_lift', 'scene/elbow_flex', 'scene/wrist_flex', 'scene/wrist_roll']
        self.gripper_joints = ['scene/gripper']

        self.manager = ControllerManager(
            joints_base=self.wheel_names,
            joints_arm=self.arm_joints,
            gripper_joints=self.gripper_joints,
            kinematics=self.kinematics,
            action_loader=self.action_loader
        )

        # ---------------- OBSERVATION & ACTION SPACE ----------------
        # Observation: [base_pos(3), eef_pos(3), eef_quat(4), box_pos(3)] = 13 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float64
        )
        self.action_space = spaces.Discrete(len(self.action_loader.all_actions()))

        # ---------------- REWARD PARAMETERS ----------------
        self.base_coeff = base_coeff  # Hệ số cho base movement
        self.reach_bonus = reach_bonus  # Bonus khi reach object
        self.reach_threshold = reach_threshold  # Khoảng cách để coi là reach
        
        # Tracking steps
        self.base_steps = 0
        self.arm_steps = 0

        # ---------------- RENDER ----------------
        self._viewer = None
        self._timestep = self.physics.model.opt.timestep
        self._step_start = None
        self.frames = []

    # ---------------- HELPER ----------------
    def _get_obs(self):
        """
        Observation bao gồm:
        - base_pos: vị trí base robot (x, y, z) - 3D
        - eef_pos: vị trí end-effector (x, y, z) - 3D  
        - eef_quat: quaternion của end-effector (w, x, y, z) - 4D
        - box_pos: vị trí của box (x, y, z) - 3D
        Tổng: 13 dimensions
        """
        # 1. Vị trí base robot (lấy từ freejoint hoặc body position)
        fk = self.kinematics.forward_kinematics()
        eef_pos = fk["eef_world_pos"]
        eef_quat = fk["eef_world_quat"]
        base_pos = fk["base_world_pos"]

        # eef_pos: [x, y, z]
        # eef_quat: [w, x, y, z] or [x, y, z, w] - cần check format
        
        # 3. Vị trí của box
        box_body_id = self.physics.model.name2id('unnamed_model/', 'body')
        box_pos = self.physics.data.xpos[box_body_id][:3]  # [x, y, z]
        
        # Concatenate tất cả
        obs = np.concatenate([
            base_pos,      # 3
            eef_pos,       # 3
            eef_quat,      # 4
            box_pos        # 3
        ])
        
        return obs.astype(np.float64)

    def _compute_reward(self, obs, invalid_action=False):
        """
        Reward = -(base_steps * base_coeff + arm_steps) + reach_bonus + invalid_action_penalty

        - base_steps: số steps mà base đã di chuyển
        - arm_steps: số steps mà arm đã di chuyển
        - base_coeff: hệ số penalty cho base movement (thường < 1)
        - reach_bonus: bonus khi eef reach được box
        - invalid_action_penalty: phạt -1 nếu action không thực hiện được
        """
        import numpy as np

        # Extract positions from observation
        eef_pos = obs[3:6]   # End-effector position
        box_pos = obs[10:13]  # Box position

        # Tính khoảng cách từ eef đến box
        distance = np.linalg.norm(eef_pos - box_pos)

        # Base reward: penalty cho movement
        movement_penalty = -(self.base_steps * self.base_coeff + self.arm_steps)

        # Reach bonus: nếu eef đủ gần box
        reach_reward = self.reach_bonus if distance < self.reach_threshold else 0.0

        # Invalid action penalty
        invalid_penalty = -1.0 if invalid_action else 0.0

        # Tổng reward
        total_reward = movement_penalty + reach_reward + invalid_penalty

        # Debug info
        info = {
            'base_steps': self.base_steps,
            'arm_steps': self.arm_steps,
            'distance_to_box': distance,
            'reached': distance < self.reach_threshold,
            'movement_penalty': movement_penalty,
            'reach_reward': reach_reward,
            'invalid_action': invalid_action,
            'invalid_penalty': invalid_penalty
        }

        return total_reward, info

        
        return total_reward, info

    def _apply_command(self, cmd):
        if "arm_qpos" in cmd:
            ids = [self.physics.model.name2id(j, 'actuator') for j in self.arm_joints]
            self.physics.data.ctrl[ids] = np.array(cmd["arm_qpos"])
        if "base_qvel" in cmd:
            ids = [self.physics.model.name2id(j, 'actuator') for j in self.wheel_names]
            self.physics.data.ctrl[ids] = np.array(cmd["base_qvel"])
        if "gripper_qpos" in cmd:
            ids = [self.physics.model.name2id(j, 'actuator') for j in self.gripper_joints]
            self.physics.data.ctrl[ids] = np.array(cmd["gripper_qpos"])

    # ---------------- GYM API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.physics.reset()
        for _ in range(90):
            self.physics.step()
            self.physics.forward()

        # Reset step counters
        self.base_steps = 0
        self.arm_steps = 0

        self.frames = []
        return self._get_obs(), {}

    def step(self, action):
        """
        Step environment:
        - action: index của action
        - n_substeps: số bước physics per step (có thể dùng để tăng tốc simulation)
        """
        # Track loại action để count steps
        action_info = self.action_loader.all_actions()[action]
        print("Selected action:", action_info)
        action_name = action_info.name.lower()

        is_base_action = 'move' or 'turn' in action_name
        is_arm_action = not is_base_action

        print(f"is_base_action: {is_base_action}, is_arm_action: {is_arm_action}")

        # Default: không thực hiện được
        action_executed = False

        # Nếu không có controller nào đang di chuyển, lấy command từ action index
        if not self.manager.is_any_moving():
            cmd = self.manager.step(action)
            if cmd is not None:
                self._apply_command(cmd)
                action_executed = True
        
        # Chạy nhiều physics steps để đẩy nhanh quá trình
        substep_count = 0
        for _ in range(1000):
            cmd = self.manager.update_control_loops()
            if cmd:
                self._apply_command(cmd)
                action_executed = True
            
            self.physics.step()
            self.physics.forward()
            substep_count += 1
            
            # Nếu đã xong movement, break sớm
            if not self.manager.is_any_moving():
                break
        
        # Increment step counters based on action type
        if is_base_action:
            self.base_steps += 1
        if is_arm_action:
            self.arm_steps += 1
            
        self.physics.step()
        self.physics.forward()

        # Render sau khi hoàn thành tất cả substeps
        if self._render_mode == "human":
            self._render_frame()
        elif self._render_mode == "rgb_array":
            frame = self.physics.render(height=480, width=480, camera_id=-1)
            self.frames.append(frame)
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward với phạt action invalid
        reward, reward_info = self._compute_reward(obs, invalid_action=not action_executed)
        
        # Check termination (ví dụ khi reach được object)
        terminated = reward_info['reached']
        
        # Merge info
        info = {
            **reward_info,
            'substeps_executed': substep_count
        }
        truncated = False  
        return obs, reward, terminated, truncated, info

    # ---------------- RENDER ----------------
    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self.physics.model.ptr,
                self.physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self.physics.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()