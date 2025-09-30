from dm_control import mjcf
from rl_mm.utils.transform_utils import (
    mat2quat
)
import numpy as np

class MobileManipulator():
    def __init__(self, xml_path, eef_site_name, base_site_name=None, joints_arm=None, joints_base=None, name: str = None):
        self._mjcf_root = mjcf.from_path(xml_path)
        if name:
            self._mjcf_root.model = name
            self.name = name

        # Nếu không truyền vào thì lấy toàn bộ joint
        if joints_arm is None:
            self._joints_arm = self._mjcf_root.find_all('joint')
        else:
            self._joints_arm = [self._mjcf_root.find('joint', j) for j in joints_arm]

        if joints_base is None:
            self._joints_base = []
        else:
            self._joints_base = [self._mjcf_root.find('joint', j) for j in joints_base]

        # End-effector site
        self._eef_site = self._mjcf_root.find('site', eef_site_name)
        if base_site_name is not None:
            self._base_site = self._mjcf_root.find('site', base_site_name)
        else:
            self._base_site = None
            
    @property
    def joints_arm(self):
        """List of joint elements belonging to the arm."""
        return self._joints_arm

    @property
    def joints_base(self):
        """List of joint elements belonging to the base."""
        return self._joints_base
        
    @property
    def eef_site(self):
        """Wrist site of the arm (attachment point for the hand)."""
        return self._eef_site
        
    @property
    def base_site(self):
        """Wrist site of the arm (attachment point for the hand)."""
        return self._base_site
    
    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root
    
    def attach_tool(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        frame = self._attachment_site.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame
    
    # def get_eef_pose(self, physics):
    #     ee_pos = physics.bind(self._eef_site).xpos
    #     ee_quat = mat2quat(physics.bind(self._eef_site).xmat.reshape(3, 3))
    #     ee_pose = np.concatenate((ee_pos, ee_quat))
    #     return ee_pose
    
    # def get_base_pose(self, physics):
    #     """
    #     Lấy pose của base (position + quaternion) trong world frame
    #     """
    #     if self._base_site is None:
    #         return None  # robot không có base site
        
    #     base_pos = physics.bind(self._base_site).xpos
    #     base_quat = mat2quat(physics.bind(self._base_site).xmat.reshape(3, 3))
    #     base_pose = np.concatenate((base_pos, base_quat))
    #     return base_pose
    
    # def get_eef_relative_pose(self, physics):
    #     ee_pos = physics.bind(self._eef_site).xpos
    #     ee_rot = physics.bind(self._eef_site).xmat.reshape(3, 3)

    #     if self._base_site is not None:
    #         base_pos = physics.bind(self._base_site).xpos
    #         base_rot = physics.bind(self._base_site).xmat.reshape(3, 3)

    #         ee_rel_pos = base_rot.T @ (ee_pos - base_pos)
    #         ee_rel_rot = base_rot.T @ ee_rot
    #         ee_rel_quat = mat2quat(ee_rel_rot)
    #         ee_rel_pose = np.concatenate((ee_rel_pos, ee_rel_quat))
    #         return ee_rel_pose
    #     else:
    #         return np.concatenate((ee_pos, mat2quat(ee_rot)))
