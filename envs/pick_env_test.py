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

    def __init__(self, render_mode=None, base_coeff=0.1, reach_bonus=1.0, reach_threshold=0.5):
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
        Optimized reward function for mobile manipulator.
        
        Key improvements:
        - Consistent distance-based rewards (no conflicting terms)
        - Per-step movement costs (not cumulative)
        - Better scaling and normalization
        - Smoother success transition
        """

        # ----- Positions -----
        eef_pos = np.asarray(obs[3:6], dtype=float)
        box_pos = np.asarray(obs[10:13], dtype=float)
        
        diff = eef_pos - box_pos
        dx, dy, dz = diff[0], diff[1], diff[2]
        
        # Current distances
        dist_xy = float(np.linalg.norm(diff[:2]))
        dist_3d = float(np.linalg.norm(diff))
        
        # ----- Delta-based shaping -----
        prev_obs = getattr(self, "prev_obs", None)
        if prev_obs is not None:
            prev_eef = np.asarray(prev_obs[3:6], dtype=float)
            prev_box = np.asarray(prev_obs[10:13], dtype=float)
            prev_xy = float(np.linalg.norm(prev_eef[:2] - prev_box[:2]))
            prev_3d = float(np.linalg.norm(prev_eef - prev_box))
            
            # Positive if moving closer
            delta_xy = prev_xy - dist_xy
            delta_3d = prev_3d - dist_3d
        else:
            delta_xy = 0.0
            delta_3d = 0.0
        
        # ----- Hyperparameters -----
        # Distance shaping weights
        w_delta_xy = getattr(self, "w_delta_xy", 2.0)      # reduced from 5.0
        w_delta_z = getattr(self, "w_delta_z", 1.0)        # separate Z component
        w_sparse_dist = getattr(self, "w_sparse_dist", -0.1)  # small absolute penalty
        
        # Movement costs (per-step, not cumulative)
        cost_base_action = getattr(self, "cost_base_action", 0.01)
        cost_arm_action = getattr(self, "cost_arm_action", 0.005)
        
        # Success thresholds
        reach_threshold = getattr(self, "reach_threshold", 0.5)
        success_threshold = getattr(self, "success_threshold", 0.05)
        
        # Bonuses
        reach_bonus_scale = getattr(self, "reach_bonus_scale", 1.0)
        success_bonus = getattr(self, "success_bonus", 10.0)
        invalid_penalty = getattr(self, "invalid_penalty", 1.0)
        
        # ----- 1. Distance-based shaping (main guidance) -----
        if prev_obs is not None:
            # Reward progress in XY (base responsibility)
            xy_term = w_delta_xy * delta_xy
            
            # Reward progress in Z (arm responsibility)
            prev_dz = float(prev_eef[2] - prev_box[2])
            delta_dz = abs(prev_dz) - abs(dz)  # positive if closer
            z_term = w_delta_z * delta_dz
            
            distance_shaping = xy_term + z_term
        else:
            # Fallback: sparse penalty based on current distance
            distance_shaping = w_sparse_dist * dist_3d
        
        # ----- 2. Movement costs (per-step only) -----
        # Check if actions were taken THIS step (you need to track this)
        base_action_taken = getattr(self, "base_action_taken", False)
        arm_action_taken = getattr(self, "arm_action_taken", False)
        
        movement_cost = 0.0
        if base_action_taken:
            movement_cost -= cost_base_action
        if arm_action_taken:
            movement_cost -= cost_arm_action
        
        # ----- 3. Reach bonus (smooth transition) -----
        if dist_3d < reach_threshold:
            # Quadratic falloff: max bonus at dist=0, zero at reach_threshold
            reach_progress = 1.0 - (dist_3d / reach_threshold)
            reach_bonus = reach_bonus_scale * (reach_progress ** 2)
        else:
            reach_bonus = 0.0
        
        # ----- 4. Success bonus -----
        success = dist_3d < success_threshold
        success_term = success_bonus if success else 0.0
        
        # ----- 5. Invalid action penalty -----
        invalid_term = -invalid_penalty if invalid_action else 0.0
        
        # ----- Total reward -----
        total_reward = (
            distance_shaping +
            movement_cost +
            reach_bonus +
            success_term +
            invalid_term
        )
        
        # ----- Info dict -----
        info = {
            "dist_3d": float(dist_3d),
            "dist_xy": float(dist_xy),
            "dist_z": float(abs(dz)),
            "delta_xy": float(delta_xy),
            "delta_3d": float(delta_3d),
            "distance_shaping": float(distance_shaping),
            "movement_cost": float(movement_cost),
            "reach_bonus": float(reach_bonus),
            "success": int(success),
            "success_term": float(success_term),
            "invalid_term": float(invalid_term),
            "total_reward": float(total_reward),
            "base_action": bool(base_action_taken),
            "arm_action": bool(arm_action_taken),
        }
        
        # ----- Store current obs for next step -----
        self.prev_obs = np.array(obs, dtype=float)
        
        return total_reward, info


# ----- Helper: You need to track actions in your step() method -----
def step(self, action):
    """
    Example of how to track actions for reward calculation.
    Add this to your environment's step() method.
    """
    # Decode action (adjust based on your action space)
    base_action = action[0]  # e.g., 0=no move, 1=forward, 2=left, etc.
    arm_action = action[1:]  # joint velocities or positions
    
    # Track which systems were used
    self.base_action_taken = (base_action != 0)  # adjust for your encoding
    self.arm_action_taken = np.any(np.abs(arm_action) > 1e-6)
    
    # ... rest of your step logic ...
    
    # Calculate reward
    reward, info = self._compute_reward(obs, invalid_action)
    
    return obs, reward, done, info


    
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
                # self._apply_command(cmd)
                action_executed = False
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
        print(action_executed)
        # Compute reward với phạt action invalid
        reward, reward_info = self._compute_reward(obs, invalid_action=not action_executed)
        print("Reward info:", reward_info,reward)
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