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
from rl_mm.arena import StandardArena  # <- thêm arena

class SO101Arm(gym.Env):
    """Gymnasium environment with StandardArena, robot, prop, and controller"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
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
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.arm_joints)+len(self.wheel_names),),
            dtype=np.float64
        )
        self.action_space = spaces.Discrete(len(self.action_loader.all_actions()))

        # ---------------- RENDER ----------------
        self._viewer = None
        self._timestep = self.physics.model.opt.timestep
        self._step_start = None
        self.frames = []

    # ---------------- HELPER ----------------
    def _get_obs(self):
        obs = []
        obs += [self.physics.data.qpos[self.physics.model.name2id(j, 'joint')] for j in self.arm_joints]
        obs += [self.physics.data.qpos[self.physics.model.name2id(j, 'joint')] for j in self.wheel_names]
        return np.array(obs, dtype=np.float64)

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

        self.frames = []
        return self._get_obs(), {}

    # def step(self, action):
    #     cmd = self.manager.step(action)
    #     self._apply_command(cmd)
    #     self.physics.step()
    #     self.physics.forward()

    #     if self._render_mode == "human":
    #         self._render_frame()
    #     elif self._render_mode == "rgb_array":
    #         frame = self.physics.render(height=480, width=480, camera_id=-1)
    #         self.frames.append(frame)

    #     obs = self._get_obs()
    #     reward = 0
    #     terminated = False
    #     return obs, reward, terminated, False, {}
    
    def step(self, action, n_substeps=10):

        # Nếu không có controller nào đang di chuyển, lấy command từ action index
        if not self.manager.is_any_moving():
            cmd = self.manager.step(action)
            self._apply_command(cmd)
        
        # Chạy nhiều physics steps để đẩy nhanh quá trình
        for _ in range(100):
            cmd = self.manager.update_control_loops()
            if cmd:
                self._apply_command(cmd)
            
            self.physics.step()
            self.physics.forward()
            
            # Nếu đã xong movement, break sớm
            if not self.manager.is_any_moving():
                break
        
        # Render sau khi hoàn thành tất cả substeps
        if self._render_mode == "human":
            self._render_frame()
        elif self._render_mode == "rgb_array":
            frame = self.physics.render(height=480, width=480, camera_id=-1)
            self.frames.append(frame)
        
        obs = self._get_obs()
        reward = 0
        terminated = False
        return obs, reward, terminated, False, {}

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
