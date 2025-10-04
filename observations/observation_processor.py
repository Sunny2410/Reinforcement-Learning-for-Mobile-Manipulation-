"""
Observation processing utilities for image and sensor data
Tạo thư mục: observations/observation_processor.py
"""
import numpy as np
from typing import Dict, Tuple, Optional
import cv2


class ObservationProcessor:
    """Xử lý observations từ environment"""
    
    def __init__(self, 
                 image_size=(84, 84),
                 use_grayscale=False,
                 stack_frames=4):
        """
        Args:
            image_size: (height, width) kích thước output image
            use_grayscale: có chuyển sang grayscale không
            stack_frames: số frame stack lại (cho temporal info)
        """
        self.image_size = image_size
        self.use_grayscale = use_grayscale
        self.stack_frames = stack_frames
        
        # Frame buffer cho stacking
        self.frame_buffer = []
        
    def process_camera_observation(self, raw_image):
        """
        Xử lý ảnh từ camera
        
        Args:
            raw_image: numpy array [H, W, C]
            
        Returns:
            processed_image: numpy array với size đã resize
        """
        # Resize
        image = cv2.resize(raw_image, self.image_size, 
                          interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if self.use_grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def stack_observations(self, observation):
        """
        Stack observations cho temporal information
        
        Args:
            observation: single frame observation
            
        Returns:
            stacked_obs: stacked frames
        """
        self.frame_buffer.append(observation)
        
        # Giữ đủ số frame
        if len(self.frame_buffer) > self.stack_frames:
            self.frame_buffer.pop(0)
        
        # Pad nếu chưa đủ frames (đầu episode)
        while len(self.frame_buffer) < self.stack_frames:
            self.frame_buffer.append(observation)
        
        # Stack along channel dimension
        stacked = np.concatenate(self.frame_buffer, axis=-1)
        return stacked
    
    def reset_buffer(self):
        """Reset frame buffer khi start episode mới"""
        self.frame_buffer = []
    
    def process_proprioceptive_state(self, physics, robot):
        """
        Xử lý proprioceptive observations (joint positions, velocities, etc.)
        
        Args:
            physics: mujoco physics object
            robot: robot object
            
        Returns:
            state_dict: dictionary chứa các state
        """
        state_dict = {}
        
        # Joint positions and velocities
        if hasattr(robot, 'arm_joints'):
            arm_qpos = [physics.bind(joint).qpos[0] for joint in robot.arm_joints]
            arm_qvel = [physics.bind(joint).qvel[0] for joint in robot.arm_joints]
            state_dict['arm_qpos'] = np.array(arm_qpos)
            state_dict['arm_qvel'] = np.array(arm_qvel)
        
        if hasattr(robot, 'gripper_joints'):
            gripper_qpos = [physics.bind(joint).qpos[0] for joint in robot.gripper_joints]
            state_dict['gripper_qpos'] = np.array(gripper_qpos)
        
        # Base position and orientation (nếu có freejoint)
        if hasattr(robot, 'freejoint'):
            base_pos = physics.bind(robot.freejoint).qpos[:3]
            base_quat = physics.bind(robot.freejoint).qpos[3:7]
            state_dict['base_pos'] = base_pos
            state_dict['base_quat'] = base_quat
        
        return state_dict
    
    def normalize_state(self, state_dict, joint_ranges):
        """
        Normalize state về [-1, 1] hoặc [0, 1]
        
        Args:
            state_dict: dictionary của states
            joint_ranges: dict với min/max values cho mỗi state
            
        Returns:
            normalized_state_dict
        """
        normalized = {}
        
        for key, value in state_dict.items():
            if key in joint_ranges:
                range_min, range_max = joint_ranges[key]
                # Normalize to [-1, 1]
                normalized[key] = 2 * (value - range_min) / (range_max - range_min) - 1
            else:
                normalized[key] = value
        
        return normalized
    
    def create_observation_space_dict(self):
        """
        Tạo observation space dictionary cho gym environment
        
        Returns:
            observation_space: Dict space
        """
        from gymnasium import spaces
        
        obs_space = {}
        
        # Image observation
        if self.use_grayscale:
            channels = self.stack_frames
        else:
            channels = 3 * self.stack_frames
        
        obs_space['image'] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(*self.image_size, channels),
            dtype=np.float32
        )
        
        # Proprioceptive observations (example sizes)
        obs_space['arm_qpos'] = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        obs_space['arm_qvel'] = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        obs_space['gripper_qpos'] = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        return spaces.Dict(obs_space)


class CameraConfig:
    """Configuration cho camera setup"""
    
    @staticmethod
    def get_camera_configs():
        """
        Trả về các camera configurations
        
        Returns:
            dict với camera names và configs
        """
        return {
            'wrist_camera': {
                'pos': [0, 0, 0],  # relative to wrist
                'quat': [1, 0, 0, 0],
                'fovy': 60,
                'resolution': (480, 640)
            },
            'third_person': {
                'pos': [-1.5, 0, 1.5],
                'quat': [0.92, 0.38, 0, 0],
                'fovy': 45,
                'resolution': (480, 640)
            },
            'front_view': {
                'pos': [2, 0, 1],
                'quat': [0.85, 0.5, 0, 0],
                'fovy': 50,
                'resolution': (480, 640)
            }
        }
    
    @staticmethod
    def setup_camera(physics, camera_name, config):
        """
        Setup camera trong mujoco physics
        
        Args:
            physics: mujoco physics
            camera_name: tên camera
            config: camera config dict
        """
        # Add camera to model if not exists
        # Implementation depends on your setup
        pass