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
    """Gymnasium environment with StandardArena, robot, prop, and controller"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None, base_coeff=0.1, reach_bonus=1.0, reach_threshold=0.5):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self._render_mode = render_mode
        
        # ---------------- DOMAIN RANDOMIZER ----------------
        self.randomizer = DomainRandomizer(
            distance_range=(0.3, 1.0),  # Object distance from robot
            angle_range=(-30, 30),       # FOV ±30 degrees
            height_range=(0.01, 0.05)    # Object height
        )
        
        # ---------------- ARENA WITH RANDOMIZATION ----------------
        self.arena = StandardArena()  # Randomize floor and wall colors
        
        # ---------------- RANDOMIZE ROBOT POSE ----------------
        robot_pos, robot_quat = self.randomizer.randomize_robot_pose(
            spawn_area=(-1.5, 1.5, -1.5, 1.5)  # Within arena bounds
        )
        
        # Add robot to arena with randomized pose
        self.robot = MobileSO101()
        self.arena.attach_free(
            self.robot.mjcf_model, 
            pos=robot_pos, 
            quat=robot_quat
        )
        print(f"Robot spawned at: pos={robot_pos}, quat={robot_quat}")
        
        # ---------------- RANDOMIZE OBJECT POSE ----------------
        # Spawn object in front of robot within FOV
        object_pos, object_quat = self.randomizer.randomize_object_pose(
            robot_pos, 
            robot_quat
        )
        
        # Create primitive box with randomization
        self.box = Primitive(type="box", size=[0.02,0.02,0.02], rgba=[1,0,0,1],mass=0.03,randomize=False)
        
        # Attach box with randomized pose
        self.arena.attach_free(
            self.box.mjcf_model, 
            pos=object_pos,
            quat=object_quat
        )
        print(f"Box spawned at: pos={object_pos}, quat={object_quat}")
        
        # ---------------- BUILD PHYSICS ----------------
        self.physics = mjcf.Physics.from_mjcf_model(self.arena.mjcf_model)
        
        # ---------------- KINEMATICS & CONTROLLER ----------------
        self.kinematics = Kinematics(self.robot, self.physics)
        self.action_loader = ActionLoader()
        self.wheel_names = [
            'scene/fl_wheel_joint', 'scene/fr_wheel_joint', 
            'scene/rl_wheel_joint', 'scene/rr_wheel_joint'
        ]
        self.arm_joints = [
            'scene/shoulder_pan', 'scene/shoulder_lift', 
            'scene/elbow_flex', 'scene/wrist_flex', 'scene/wrist_roll'
        ]
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
        self.observation_space = spaces.Dict({
            'state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float64
            ),
            'image': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(224, 224, 3),      # RGB image
                dtype=np.float32
            )
        })
        self.action_space = spaces.Discrete(len(self.action_loader.all_actions()))
        self.obs_processor = ObservationProcessor(
                            image_size=(224, 224),      
                            use_grayscale=False,
                            stack_frames=1           
                        )
        self.camera_name = "scene/front_cam"

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

    def _get_obs(self):
        """
        Observation với proper normalization.
        State: 7D normalized vector
        Image: (224, 224, 3) processed camera
        """
        fk = self.kinematics.forward_kinematics()
        
        base_pos = fk["base_world_pos"]
        eef_pos = fk["eef_world_pos"]
        eef_quat = fk["eef_world_quat"]  # (w, x, y, z)
        
        # Relative position
        eef_pos_relative = eef_pos - base_pos
        self.max_eef_reach = 2
        # Normalize position to [-1, 1]
        eef_pos_normalized = np.clip(
            eef_pos_relative / self.max_eef_reach,
            -1.0, 1.0
        )
        
        # Ensure quaternion is normalized
        quat_norm = np.linalg.norm(eef_quat)
        if quat_norm > 1e-6:
            eef_quat = eef_quat / quat_norm
        
        state = np.concatenate([
            eef_pos_normalized,  # 3D ∈ [-1, 1]
            eef_quat,            # 4D ∈ [-1, 1] (unit quaternion)
        ]).astype(np.float32)
        
        # Camera
        try:
            camera_id = self.physics.model.name2id(self.camera_name, 'camera')
            raw_image = self.physics.render(height=480, width=640, camera_id=camera_id)
            processed_image = self.obs_processor.process_camera_observation(raw_image)
        except Exception as e:
            print(f"Warning: Cannot get camera: {e}")
            processed_image = np.zeros((224, 224, 3), dtype=np.float32)
        
        return {
            'state': state,
            'image': processed_image
        }
    
    def _compute_reward(self, obs, invalid_action=False):
        """
        Reward function với PRIVILEGED box position.
        
        Note: Box position KHÔNG có trong obs['state'], nhưng dùng cho reward.
        Đây là asymmetric information (teacher-student paradigm).
        """
        
        # ----- Early return for invalid actions -----
        invalid_penalty = getattr(self, "invalid_penalty", 10.0)
        if invalid_action:
            self.prev_obs = np.array(obs['state'], dtype=float)
            self.prev_box_pos = self._get_box_pos()  # Store for next step
            
            return -invalid_penalty, {
                "dist_3d": 0.0,
                "dist_xy": 0.0,
                "dist_z": 0.0,
                "delta_xy": 0.0,
                "delta_z": 0.0,
                "xy_shaping": 0.0,
                "z_shaping": 0.0,
                "xy_bonus": 0.0,
                "reach_bonus": 0.0,
                "reached": 0,
                "success_bonus": 0.0,
                "movement_cost": 0.0,
                "total_reward": float(-invalid_penalty),
                "invalid": 1,
            }
        
        state = obs['state']
        
        # ----- Positions -----
        # State format: [eef_pos_relative (3), eef_quat (4)] = 7D
        # eef_pos_relative = state[0:3]
        # eef_quat = state[3:7]
        
        # ✨ Get absolute positions for reward calculation
        fk = self.kinematics.forward_kinematics()
        base_pos = fk["base_world_pos"]
        eef_pos_relative = np.asarray(state[0:3], dtype=float)
        eef_pos = base_pos + eef_pos_relative  # Convert back to world frame
        
        # ✨ Get box position (PRIVILEGED - không có trong obs!)
        box_pos = self._get_box_pos()
        
        diff = eef_pos - box_pos
        dx, dy, dz = diff[0], diff[1], diff[2]
        
        # Current distances
        dist_xy = float(np.linalg.norm(diff[:2]))
        dist_z = float(abs(dz))
        dist_3d = float(np.linalg.norm(diff))
        
        # ----- Delta-based shaping -----
        prev_obs = getattr(self, "prev_obs", None)
        prev_box_pos = getattr(self, "prev_box_pos", None)
        
        if prev_obs is not None and prev_box_pos is not None:
            prev_eef_relative = np.asarray(prev_obs[0:3], dtype=float)
            prev_base_pos = getattr(self, "prev_base_pos", base_pos)
            prev_eef = prev_base_pos + prev_eef_relative
            
            prev_diff = prev_eef - prev_box_pos
            prev_xy = float(np.linalg.norm(prev_diff[:2]))
            prev_z = float(abs(prev_diff[2]))
            
            delta_xy = prev_xy - dist_xy
            delta_z = prev_z - dist_z
        else:
            delta_xy = 0.0
            delta_z = 0.0
        
        # ----- Hyperparameters -----
        w_xy = getattr(self, "w_xy", 5.0)
        w_z = getattr(self, "w_z", 3.0)
        cost_base = getattr(self, "cost_base", 0.02)
        cost_arm = getattr(self, "cost_arm", 0.01)
        xy_close_threshold = getattr(self, "xy_close_threshold", 0.3)
        reach_threshold = getattr(self, "reach_threshold", 0.15)
        success_threshold = getattr(self, "success_threshold", 0.08)
        xy_close_bonus = getattr(self, "xy_close_bonus", 2.0)
        reach_bonus_val = getattr(self, "reach_bonus_val", 5.0)
        success_bonus = getattr(self, "success_bonus", 20.0)
        
        # ----- 1. XY Shaping -----
        if prev_obs is not None and delta_xy != 0.0:
            scale = np.exp(-dist_xy)
            xy_shaping = w_xy * delta_xy * (1.0 + scale)
        else:
            xy_shaping = 0.0
        
        # ----- 2. Z Shaping -----
        if prev_obs is not None and delta_z != 0.0:
            if dist_xy < 0.5:
                scale = np.exp(-dist_z)
                z_shaping = w_z * delta_z * (1.0 + scale)
            else:
                z_shaping = -0.5 * abs(delta_z)
        else:
            z_shaping = 0.0
        
        # ----- 3. Movement costs -----
        base_action_taken = getattr(self, "base_action_taken", False)
        arm_action_taken = getattr(self, "arm_action_taken", False)
        
        movement_cost = 0.0
        if base_action_taken:
            if dist_xy < xy_close_threshold:
                movement_cost -= cost_base * 3.0
            else:
                movement_cost -= cost_base
        
        if arm_action_taken:
            if dist_xy > 0.5:
                movement_cost -= cost_arm * 2.0
            else:
                movement_cost -= cost_arm
        
        # ----- 4. Milestone bonuses -----
        if dist_xy < xy_close_threshold:
            xy_bonus = xy_close_bonus * (1.0 - dist_xy / xy_close_threshold)
        else:
            xy_bonus = 0.0
        
        if dist_3d < reach_threshold:
            reach_progress = 1.0 - (dist_3d / reach_threshold)
            reach_bonus = reach_bonus_val * (reach_progress ** 2)
        else:
            reach_bonus = 0.0
        
        # ----- 5. Success -----
        success = dist_3d < success_threshold
        success_term = success_bonus if success else 0.0
        
        # ----- Total reward -----
        total_reward = float(
            xy_shaping +
            z_shaping +
            xy_bonus +
            reach_bonus +
            movement_cost +
            success_term
        )
        
        # ----- Info dict -----
        info = {
            "dist_3d": float(dist_3d),
            "dist_xy": float(dist_xy),
            "dist_z": float(dist_z),
            "delta_xy": float(delta_xy),
            "delta_z": float(delta_z),
            "xy_shaping": float(xy_shaping),
            "z_shaping": float(z_shaping),
            "xy_bonus": float(xy_bonus),
            "reach_bonus": float(reach_bonus),
            "reached": bool(success),
            "success_bonus": float(success_term),
            "movement_cost": float(movement_cost),
            "total_reward": float(total_reward),
            "invalid": 0,
        }
        
        # ----- Store for next step -----
        self.prev_obs = np.array(obs['state'], dtype=float)
        self.prev_box_pos = box_pos
        self.prev_base_pos = base_pos
        
        return total_reward, info


    def _get_box_pos(self):
        """Helper to get box position (privileged info)."""
        box_body_id = self.physics.model.name2id('unnamed_model/', 'body')
        return np.asarray(self.physics.data.xpos[box_body_id][:3], dtype=float)


# ----- Recommended hyperparameters -----
    """
    HYPERPARAMETERS (set as class attributes or pass to __init__):

    # Invalid penalty (CRITICAL!)
    invalid_penalty = 10.0  # Large enough to matter

    # Shaping weights
    w_xy = 5.0              # XY approach (base movement)
    w_z = 3.0               # Z approach (arm movement)

    # Movement costs
    cost_base = 0.02        # Small per-step cost
    cost_arm = 0.01

    # Distance thresholds
    xy_close_threshold = 0.3    # 30cm - base considered "close"
    reach_threshold = 0.15      # 15cm - arm can reach
    success_threshold = 0.08    # 8cm - success

    # Milestone bonuses
    xy_close_bonus = 2.0        # Bonus for getting XY close
    reach_bonus_val = 5.0       # Bonus for final approach
    success_bonus = 20.0        # Large reward for success

    TRAINING CONFIG:
    - total_timesteps: 500,000 (for 1m distance)
    - n_steps: 2048
    - batch_size: 64
    - learning_rate: 3e-4
    - gamma: 0.99
    - gae_lambda: 0.95
    """

# ----- Helper: You need to track actions in your step() method -----

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

        # Randomize poses
        robot_pos, robot_quat = self.randomizer.randomize_robot_pose(
            spawn_area=(-1.5, 1.5, -1.5, 1.5)
        )
        object_pos, object_quat = self.randomizer.randomize_object_pose(
            robot_pos, robot_quat
        )

        # Robot
        try:
            robot_joint_id = self.physics.model.name2id("scene/", "joint")
            start = self.physics.model.jnt_qposadr[robot_joint_id]
            self.physics.data.qpos[start:start+3] = robot_pos
            self.physics.data.qpos[start+3:start+7] = robot_quat
        except Exception:
            print("❌ Cannot find robot joint!")

        # Box
        try:
            box_joint_id = self.physics.model.name2id("unnamed_model/", "joint")
            start = self.physics.model.jnt_qposadr[box_joint_id]
            self.physics.data.qpos[start:start+3] = object_pos
            self.physics.data.qpos[start+3:start+7] = object_quat
        except Exception:
            print("❌ Cannot find box joint!")

        # Reset velocities
        self.physics.data.qvel[:] = 0
        self.physics.data.qacc[:] = 0

        # Forward
        self.physics.forward()

        # # Reset tracking vars
        # for _ in range(90):
        #     self.physics.step()
        #     self.physics.forward()

        # Reset step counters
        self.base_steps = 0
        self.arm_steps = 0
        self.kinematics.reset_mink_configuration()
        self.frames = []

        observation = self._get_obs()
        info = {}

        # print(f"✅ Robot moved to {robot_pos}, quat={robot_quat}")
        # print(f"✅ Object moved to {object_pos}, quat={object_quat}")

        return observation, info


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

        # print(f"is_base_action: {is_base_action}, is_arm_action: {is_arm_action}")

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

        # # Render sau khi hoàn thành tất cả substeps
        # if self._render_mode == "human":
        #     self._render_frame()
        # elif self._render_mode == "rgb_array":
        #     frame = self.physics.render(height=480, width=480, camera_id=-1)
        #     self.frames.append(frame)
        
        # Get observation
        obs = self._get_obs()
        # Compute reward với phạt action invalid
        reward, reward_info = self._compute_reward(obs, invalid_action=not action_executed)
        # print("Reward info:", reward_info,reward)
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