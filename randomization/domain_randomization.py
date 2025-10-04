"""
Domain Randomization utilities for spawning robot and objects
Tạo thư mục: randomization/domain_randomization.py
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


class DomainRandomizer:
    """Handles domain randomization for robot and object spawning"""
    
    def __init__(self, 
                 distance_range=(0.3, 1.0),
                 angle_range=(-30, 30),
                 height_range=(0.01, 0.05)):
        """
        Args:
            distance_range: (min, max) khoảng cách từ robot (meters)
            angle_range: (min, max) góc theo hướng nhìn robot (degrees)
            height_range: (min, max) độ cao spawn object (meters)
        """
        self.distance_range = distance_range
        self.angle_range = angle_range
        self.height_range = height_range
        
    def randomize_robot_pose(self, spawn_area=(-1.5, 1.5, -1.5, 1.5)):
        """
        Random vị trí và hướng của robot trong arena
        
        Args:
            spawn_area: (x_min, x_max, y_min, y_max) khu vực spawn
            
        Returns:
            pos: [x, y, z] vị trí robot
            quat: [w, x, y, z] quaternion hướng robot
        """
        x = np.random.uniform(spawn_area[0], spawn_area[1])
        y = np.random.uniform(spawn_area[2], spawn_area[3])
        z = 0  # Robot luôn ở mặt đất
        
        # Random góc quay quanh trục z (yaw)
        yaw = np.random.uniform(-np.pi, np.pi)
        quat = self._euler_to_quat(0, 0, yaw)
        
        return [x, y, z], quat
    
    def randomize_object_pose(self, robot_pos, robot_quat):
        """
        Random vị trí object trước mặt robot
        
        Args:
            robot_pos: [x, y, z] vị trí robot
            robot_quat: [w, x, y, z] quaternion hướng robot
            
        Returns:
            pos: [x, y, z] vị trí object
            quat: [w, x, y, z] quaternion hướng object
        """
        # Lấy hướng nhìn của robot (forward direction)
        robot_forward = self._get_forward_direction(robot_quat)
        
        # Random khoảng cách và góc lệch
        distance = np.random.uniform(*self.distance_range)
        angle_offset = np.random.uniform(*self.angle_range)
        angle_rad = np.deg2rad(angle_offset)
        
        # Tính vị trí object
        # Rotate forward direction bởi angle_offset
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation matrix quanh trục z
        dx = distance * (robot_forward[0] * cos_a - robot_forward[1] * sin_a)
        dy = distance * (robot_forward[0] * sin_a + robot_forward[1] * cos_a)
        
        # Random height
        height = np.random.uniform(*self.height_range)
        
        obj_pos = [
            robot_pos[0] + dx,
            robot_pos[1] + dy,
            height
        ]
        
        # Random orientation cho object
        yaw = np.random.uniform(-np.pi, np.pi)
        obj_quat = self._euler_to_quat(0, 0, yaw)
        
        return obj_pos, obj_quat
    
    def _get_forward_direction(self, quat):
        """
        Lấy vector hướng nhìn từ quaternion
        
        Args:
            quat: [w, x, y, z]
            
        Returns:
            forward: [x, y, z] normalized forward vector
        """
        # Convert quaternion to rotation matrix
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
        
        # Forward direction trong robot frame là [1, 0, 0]
        forward = rot.apply([1, 0, 0])
        
        return forward
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion [w, x, y, z]
        
        Args:
            roll, pitch, yaw: góc quay (radians)
            
        Returns:
            quat: [w, x, y, z]
        """
        rot = R.from_euler('xyz', [roll, pitch, yaw])
        quat_xyzw = rot.as_quat()  # [x, y, z, w]
        return [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w, x, y, z]
    
    def randomize_object_properties(self):
        """
        Random các thuộc tính vật lý của object (optional)
        
        Returns:
            dict với các thuộc tính như size, color, mass, friction
        """
        size_scale = np.random.uniform(0.8, 1.2)
        base_size = 0.02
        
        return {
            'size': [base_size * size_scale] * 3,
            'rgba': [
                np.random.uniform(0.5, 1.0),  # R
                np.random.uniform(0.0, 0.5),  # G
                np.random.uniform(0.0, 0.5),  # B
                1.0  # A
            ],
            'mass': np.random.uniform(0.05, 0.2),
            'friction': [np.random.uniform(0.5, 1.5)] * 3
        }


# Utility functions để dùng trực tiếp
def get_random_spawn_poses(distance_range=(0.3, 1.0), angle_range=(-30, 30)):
    """
    Convenience function để lấy random poses cho robot và object
    
    Returns:
        robot_pos, robot_quat, object_pos, object_quat
    """
    randomizer = DomainRandomizer(distance_range, angle_range)
    
    robot_pos, robot_quat = randomizer.randomize_robot_pose()
    object_pos, object_quat = randomizer.randomize_object_pose(robot_pos, robot_quat)
    
    return robot_pos, robot_quat, object_pos, object_quat