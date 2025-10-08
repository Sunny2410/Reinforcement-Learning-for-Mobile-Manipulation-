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
from rl_mm.observations.observation_processor import ObservationProcessor
from rl_mm.randomization import DomainRandomizer

class SO101Arm3(gym.Env):
    """
    Gymnasium environment with a comprehensively optimized reward structure.
    - Includes potential-based shaping for orientation and height alignment.
    - Balanced parameters for robust learning.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self._render_mode = render_mode
        
        # --- Domain Randomization ---
        self.randomizer = DomainRandomizer(
            distance_range=(0.3, 1.0), angle_range=(-30, 30), height_range=(0.01, 0.05)
        )
        
        # --- Arena Setup ---
        self.arena = StandardArena()
        
        # --- Robot Setup ---
        robot_pos, robot_quat = self.randomizer.randomize_robot_pose(spawn_area=(-1.5, 1.5, -1.5, 1.5))
        self.robot = MobileSO101()
        self.arena.attach_free(self.robot.mjcf_model, pos=robot_pos, quat=robot_quat)
        
        # --- Object Setup ---
        object_pos, object_quat = self.randomizer.randomize_object_pose(robot_pos, robot_quat)
        self.box = Primitive(type="box", size=[0.02,0.02,0.02], rgba=[1,0,0,1], mass=0.03)
        self.arena.attach_free(self.box.mjcf_model, pos=object_pos, quat=object_quat)
        
        # --- Physics and Controllers ---
        self.physics = mjcf.Physics.from_mjcf_model(self.arena.mjcf_model)
        self.kinematics = Kinematics(self.robot, self.physics)
        self.action_loader = ActionLoader()
        self.wheel_names = ['scene/fl_wheel_joint', 'scene/fr_wheel_joint', 'scene/rl_wheel_joint', 'scene/rr_wheel_joint']
        self.arm_joints = ['scene/shoulder_pan', 'scene/shoulder_lift', 'scene/elbow_flex', 'scene/wrist_flex', 'scene/wrist_roll']
        self.gripper_joints = ['scene/gripper']
        self.manager = ControllerManager(
            joints_base=self.wheel_names, joints_arm=self.arm_joints, gripper_joints=self.gripper_joints,
            kinematics=self.kinematics, action_loader=self.action_loader
        )

        # --- Observation & Action Space ---
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            'image': spaces.Box(low=0.0, high=1.0, shape=(224, 224, 3), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(len(self.action_loader.all_actions()))
        self.obs_processor = ObservationProcessor(image_size=(224, 224), use_grayscale=False, stack_frames=1)
        self.camera_name = "scene/front_cam"

        # ==============================================================================
        # ---------------- REWARD PARAMETERS (OPTIMIZED & BALANCED) ----------------
        # ==============================================================================
        
        # --- Shaping weights for potential function ---
        self.w_approach = 1.5         # Weight for distance-based approach
        self.w_lift = 3.0             # Weight for lifting potential
        self.w_orientation = 0.5      # NEW: Weight for gripper orientation
        self.w_height_align = 0.75    # NEW: Weight for Z-height alignment
        
        # --- Distance & Height Thresholds ---
        self.target_height_above_box = 0.05 # 5cm above the box is ideal for grasping
        self.lift_height = 0.15       # 15cm - lift threshold for bonus
        self.success_height = 0.10    # 10cm - A clear success height
        self.xy_align_dist = 0.05     # 5cm - XY alignment threshold
        
        # --- Milestone Bonuses (Hierarchical) ---
        self.bonus_xy_aligned = 0.5   # Small bonus for getting close
        self.bonus_grasp = 2.0        # Significant bonus for a successful grasp
        self.bonus_lift = 3.0         # Large bonus for lifting the object
        self.bonus_success = 10.0     # HUGE bonus for completing the task
        
        # --- Penalties (Clear Consequences) ---
        self.penalty_invalid = 0.05   # Minimal penalty to not hinder exploration
        self.penalty_drop = -5.0      # LARGE penalty for dropping the object
        self.penalty_time = 0.001     # Tiny per-step cost for efficiency
        
        # --- Episode settings & Tracking variables ---
        self.max_episode_steps = 1000
        self.current_step = 0
        self.stage_xy_aligned = False
        self.stage_grasped = False
        self.stage_lifted = False
        self.prev_potential = 0.0
        self.prev_grasped = False
        self.gripper_is_open = True
        self.max_eef_reach = 2.0
        
        # --- Render ---
        self._viewer = None
        self._timestep = self.physics.model.opt.timestep
        self._step_start = None

    def _get_obs(self):
        fk = self.kinematics.forward_kinematics()
        base_pos, eef_pos, eef_quat = fk["base_world_pos"], fk["eef_world_pos"], fk["eef_world_quat"]
        eef_pos_relative = eef_pos - base_pos
        eef_pos_normalized = np.clip(eef_pos_relative / self.max_eef_reach, -1.0, 1.0)
        quat_norm = np.linalg.norm(eef_quat)
        if quat_norm > 1e-6: eef_quat /= quat_norm
        state = np.concatenate([eef_pos_normalized, eef_quat]).astype(np.float32)
        try:
            raw_image = self.physics.render(height=480, width=640, camera_id=self.physics.model.name2id(self.camera_name, 'camera'))
            processed_image = self.obs_processor.process_camera_observation(raw_image)
        except Exception:
            processed_image = np.zeros((224, 224, 3), dtype=np.float32)
        return {'state': state, 'image': processed_image}
    
    def _get_gripper_opening(self):
        return 1.0 if self.gripper_is_open else 0.0
    
    def _compute_potential(self, eef_pos, eef_quat, box_pos, is_grasped):
        """
        🎯 OPTIMIZED Potential function with orientation and height alignment.
        This function calculates a dense reward signal to guide the agent.
        """
        
        if not is_grasped:
            # --- STAGE 1-2: APPROACHING ---
            
            # 1. Distance Potential (XY and Z): Encourages getting closer.
            dist_3d = np.linalg.norm(eef_pos - box_pos)
            approach_potential = -self.w_approach * dist_3d
            
            # 2. Height Alignment Potential (Gaussian reward): Encourages ideal height for grasping.
            height_error = abs((eef_pos[2] - box_pos[2]) - self.target_height_above_box)
            height_align_potential = self.w_height_align * np.exp(-100 * (height_error**2))
            
            # 3. Orientation Potential: Encourages gripper to point downwards.
            w, x, y, z = eef_quat
            # This calculates the gripper's Z-axis vector in the world frame
            gripper_approach_vec = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x**2 + y**2)])
            dot_product = np.dot(gripper_approach_vec, np.array([0., 0., -1.]))
            orientation_potential = self.w_orientation * np.exp(5 * (dot_product - 1))
            
            total_potential = approach_potential + height_align_potential + orientation_potential
            
        else:
            # --- STAGE 3-4: LIFTING ---
            height_gain = max(0.0, box_pos[2] - 0.02) # Height above ground
            total_potential = self.w_lift * (np.exp(height_gain * 10) - 1)
        
        return float(total_potential)

    def _check_grasp_state(self):
        return self.manager.gripper_controller.is_grasping()

    def _compute_reward(self, obs, invalid_action=False):
        """
        🎯 OPTIMIZED Dense Reward for Pick Task.
        Combines potential shaping, milestone bonuses, and penalties.
        """
        if invalid_action:
            info = { 'total': float(-self.penalty_invalid - self.penalty_time), 'is_grasped': int(self.prev_grasped), 'reached': 0, 'invalid': 1, 'stage_xy_aligned': int(self.stage_xy_aligned), 'stage_grasped': int(self.stage_grasped), 'stage_lifted': int(self.stage_lifted), 'potential_shaping': 0.0, 'stage_bonus': 0.0, 'drop_penalty': 0.0 }
            return float(-self.penalty_invalid - self.penalty_time), info
        
        # --- 1. Extract State Information ---
        state = obs['state']
        base_pos = self.kinematics.forward_kinematics()["base_world_pos"]
        eef_pos = base_pos + np.asarray(state[0:3], dtype=float) * self.max_eef_reach
        eef_quat = np.asarray(state[3:7], dtype=float)
        box_pos = self._get_box_pos()
        is_grasped = self._check_grasp_state()
        
        # --- 2. Potential-Based Shaping (Main Signal) ---
        current_potential = self._compute_potential(eef_pos, eef_quat, box_pos, is_grasped)
        potential_shaping = 0.99 * current_potential - self.prev_potential
        
        # --- 3. Stage Milestone Bonuses ---
        stage_bonus = 0.0
        dist_xy = np.linalg.norm(eef_pos[:2] - box_pos[:2])
        if not self.stage_xy_aligned and dist_xy < self.xy_align_dist:
            stage_bonus += self.bonus_xy_aligned; self.stage_xy_aligned = True
        if not self.prev_grasped and is_grasped:
            stage_bonus += self.bonus_grasp; self.stage_grasped = True
        box_height = float(box_pos[2])
        if not self.stage_lifted and is_grasped and box_height > self.lift_height:
            stage_bonus += self.bonus_lift; self.stage_lifted = True
        
        success = is_grasped and box_height >= self.success_height
        if success:
            stage_bonus += self.bonus_success
        
        # --- 4. Drop Penalty ---
        drop_penalty = 0.0
        if self.prev_grasped and not is_grasped:
            drop_penalty = self.penalty_drop
            self.stage_grasped = False; self.stage_lifted = False
        
        # --- 5. Total Reward & Info ---
        total_reward = float(potential_shaping + stage_bonus + drop_penalty - self.penalty_time)
        info = {
            'potential_shaping': float(potential_shaping), 'stage_bonus': float(stage_bonus), 'drop_penalty': float(drop_penalty),
            'total': total_reward, 'is_grasped': int(is_grasped), 'reached': int(success), 'invalid': 0,
            'stage_xy_aligned': int(self.stage_xy_aligned), 'stage_grasped': int(self.stage_grasped), 'stage_lifted': int(self.stage_lifted),
        }
        
        self.prev_potential = current_potential
        self.prev_grasped = is_grasped
        return total_reward, info

    def _get_box_pos(self):
        return np.asarray(self.physics.data.xpos[self.physics.model.name2id('unnamed_model/', 'body')][:3], dtype=float)

    def _apply_command(self, cmd):
        if "arm_qpos" in cmd: self.physics.data.ctrl[[self.physics.model.name2id(j, 'actuator') for j in self.arm_joints]] = np.array(cmd["arm_qpos"])
        if "base_qvel" in cmd: self.physics.data.ctrl[[self.physics.model.name2id(j, 'actuator') for j in self.wheel_names]] = np.array(cmd["base_qvel"])
        if "gripper_qpos" in cmd: self.physics.data.ctrl[[self.physics.model.name2id(j, 'actuator') for j in self.gripper_joints]] = np.array(cmd["gripper_qpos"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        robot_pos, robot_quat = self.randomizer.randomize_robot_pose(spawn_area=(-1.5, 1.5, -1.5, 1.5))
        object_pos, object_quat = self.randomizer.randomize_object_pose(robot_pos, robot_quat)
        robot_joint_id = self.physics.model.name2id("scene/", "joint")
        self.physics.data.qpos[self.physics.model.jnt_qposadr[robot_joint_id]:self.physics.model.jnt_qposadr[robot_joint_id]+7] = np.hstack([robot_pos, robot_quat])
        box_joint_id = self.physics.model.name2id("unnamed_model/", "joint")
        self.physics.data.qpos[self.physics.model.jnt_qposadr[box_joint_id]:self.physics.model.jnt_qposadr[box_joint_id]+7] = np.hstack([object_pos, object_quat])
        self.physics.data.qvel[:] = 0; self.physics.data.qacc[:] = 0
        self.physics.forward()
        self.kinematics.reset_mink_configuration()

        self.current_step, self.stage_xy_aligned, self.stage_grasped, self.stage_lifted, self.prev_grasped, self.gripper_is_open = 0, False, False, False, False, True
        
        obs = self._get_obs()
        base_pos = self.kinematics.forward_kinematics()["base_world_pos"]
        eef_pos = base_pos + np.asarray(obs['state'][0:3], dtype=float) * self.max_eef_reach
        eef_quat = np.asarray(obs['state'][3:7], dtype=float)
        self.prev_potential = self._compute_potential(eef_pos, eef_quat, self._get_box_pos(), False)
        
        return obs, {}

    def step(self, action):
        self.current_step += 1
        action_name = self.action_loader.all_actions()[action].name.lower()
        if 'gripper_open' in action_name: self.gripper_is_open = True
        elif 'gripper_close' in action_name: self.gripper_is_open = False
            
        action_executed = False
        if not self.manager.is_any_moving():
            if self.manager.step(action) is not None: action_executed = True

        for _ in range(1000):
            cmd = self.manager.update_control_loops()
            if cmd: self._apply_command(cmd); action_executed = True
            self.physics.step(); self.physics.forward()
            if not self.manager.is_any_moving(): break

        self.physics.step(); self.physics.forward()

        if self._render_mode == "human": self._render_frame()
        
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(obs, invalid_action=not action_executed)
        terminated = reward_info['reached']
        truncated = self.current_step >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, reward_info

    def render(self):
        if self._render_mode == "human": self._render_frame()
        else: return self.physics.render(height=480, width=640, camera_id=self.physics.model.name2id(self.camera_name, 'camera'))

    def _render_frame(self):
        if self._viewer is None: self._viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
        if self._step_start is None: self._step_start = time.time()
        self._viewer.sync()
        time_until_next_step = self._timestep - (time.time() - self._step_start)
        if time_until_next_step > 0: time.sleep(time_until_next_step)
        self._step_start = time.time()

    def close(self):
        if self._viewer is not None: self._viewer.close()
