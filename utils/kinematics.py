"""
Kinematics utilities for mobile manipulator robot.
Includes base (mecanum) kinematics, forward kinematics, and inverse kinematics.
"""

# import numpy as np
# from dm_control import mjcf
# from dm_control.utils import inverse_kinematics as ik
# from typing import Dict, List, Optional
# from rl_mm.utils.transform_utils import mat2quat
# import mink

# class Kinematics:
#     def __init__(self, robot, physics: Optional[mjcf.Physics] = None, use_mink: bool = True):
#         """
#         Initialize kinematics utility.
        
#         Args:
#             robot: MobileSO101 robot instance
#             physics: MuJoCo physics instance from dm_control (optional)
#             use_mink: If True, use Mink IK by default
#         """
#         import mujoco
#         import mink

#         self.robot = robot
#         self.physics = physics or mjcf.Physics.from_mjcf_model(robot.mjcf_model)
#         self.use_mink = use_mink

#         self._prefix = "scene/"
#         self.eef_site_name = self._prefix + (robot.eef_site if isinstance(robot.eef_site, str) else robot.eef_site.name)
#         self.base_site_name = self._prefix + (robot.base_site if isinstance(robot.base_site, str) else robot.base_site.name)

#         # ---------------------------
#         # Mink configuration initialized immediately
#         # ---------------------------
#         self._mink_configuration = None
#         if self.use_mink:
#             try:
#                 # Load MuJoCo model directly from fixed path
#                 mj_model = mujoco.MjModel.from_xml_path("/home/sunny24/rl_mm/asset/SO101/so101_new_calib.xml")
#                 self._mink_configuration = mink.Configuration(mj_model)
#                 # Initialize joint positions to zeros
#             except Exception as e:
#                 print(f"[Mink init failed] {str(e)}")
#                 self._mink_configuration = None


#     # ---------------------------
#     # Mecanum base kinematics
#     # ---------------------------
#     def compute_wheel_speeds(
#         self, 
#         target_rel_pos: np.ndarray, 
#         target_rel_yaw: float,
#         current_rel_pos: np.ndarray, 
#         current_rel_yaw: float,
#         k: float = 1.0
#     ) -> np.ndarray:
#         # ... same as your existing method ...
#         pos_error = target_rel_pos - current_rel_pos
#         yaw_error = target_rel_yaw - current_rel_yaw
#         POS_THRESHOLD = 0.01
#         YAW_THRESHOLD = np.deg2rad(2)
#         if (np.linalg.norm(pos_error) < POS_THRESHOLD and abs(yaw_error) < YAW_THRESHOLD):
#             return np.zeros(4)
#         dx, dy = k * pos_error
#         omega = k * yaw_error
#         return np.array([dx - dy - omega, dx + dy + omega, dx + dy - omega, dx - dy + omega])

#     # ---------------------------
#     # Forward Kinematics
#     # ---------------------------
#     def forward_kinematics(self) -> Dict[str, np.ndarray]:
#         self.physics.forward()
#         eef_pose = self.get_pose_from_site(self.eef_site_name)
#         base_pose = None
#         if self.base_site_name is not None:
#             base_pose = self.get_pose_from_site(self.base_site_name)
#         return {
#             "eef_world_pos": eef_pose[:3],
#             "eef_world_quat": eef_pose[3:],
#             "base_world_pos": None if base_pose is None else base_pose[:3],
#             "base_world_quat": None if base_pose is None else base_pose[3:],
#         }

#     def get_pose_from_site(self, site_name: str):
#         site_id = self.physics.model.name2id(site_name, "site")
#         pos = self.physics.data.site_xpos[site_id].copy()
#         quat = mat2quat(self.physics.data.site_xmat[site_id].reshape(3, 3))
#         return np.concatenate((pos, quat))

#     def get_eef_pose(self, eef_site_name):
#         return self.get_pose_from_site(eef_site_name)

#     def get_base_pose(self, base_site_name):
#         if base_site_name is None:
#             return None
#         return self.get_pose_from_site(base_site_name)

#     # ---------------------------
#     # Inverse Kinematics
#     # ---------------------------
#     def inverse_kinematics(
#         self,
#         target_pos: np.ndarray,
#         target_quat: Optional[np.ndarray] = None,
#         joint_names: Optional[List[str]] = None,
#         tol: float = 3e-3,
#         max_steps: int = 100,
#         use_mink: Optional[bool] = None
#     ) -> Optional[np.ndarray]:
#         """
#         Compute inverse kinematics for target end-effector pose.
        
#         Args:
#             target_pos: Target position [x, y, z] in world frame
#             target_quat: Optional target orientation as quaternion [x, y, z, w]
#             joint_names: List of joint names to use for IK (default: robot.joints_arm)
#             tol: Tolerance for IK convergence
#             max_steps: Maximum number of iterations
#             use_mink: Whether to use Mink IK (default: class default)
            
#         Returns:
#             np.ndarray: Joint positions if solution found, None otherwise
#         """
#         if use_mink is None:
#             use_mink = self.use_mink

#         if joint_names is None:
#             if hasattr(self.robot, "joints_arm"):
#                 joint_names = [self._prefix + j.name for j in self.robot.joints_arm]
#             else:
#                 raise AttributeError("Robot does not define arm joints.")

#         if use_mink:
#             config = self._mink_configuration
#             if config is None:
#                 print("[Mink IK] Mink configuration not available, fallback to dm_control IK")
#                 use_mink = False
#             else:
#                 # Create IK task
#                 # Tạo IK task
#                 task = mink.FrameTask(
#                     "gripperframe",  # frame robot
#                     "site",          # target type
#                     position_cost=1.0,
#                     orientation_cost=1
#                 )

#                 # Lấy transform hiện tại của gripper
#                 transform_init = config.get_transform_frame_to_world("gripperframe", "site")

#                 # Tạo rotation SO3 từ roll-pitch-yaw (radian)
#                 if target_quat is not None:  # target_rpy = (roll, pitch, yaw)
#                     rotation_so3 = mink.SO3.from_rpy_radians(
#                         roll=target_quat[0],
#                         pitch=target_quat[1],
#                         yaw=target_quat[2]
#                     )
#                 else:
#                     rotation_so3 = mink.SO3.identity()  # không xoay nếu không có target
#                 # Tạo SE3 target với rotation + translation
#                 target_se3 = mink.SE3.from_rotation_and_translation(rotation_so3, target_pos)

#                 # Kết hợp với transform hiện tại để có target cuối cùng trong world frame
#                 transform_target_to_world = transform_init @ target_se3

#                 # Gán target cho task
#                 task.set_target(transform_target_to_world)

#                 # Differential IK loop
#                 success = False
#                 for step in range(max_steps):
#                     v = mink.solve_ik(config, [task], dt=0.01, solver="daqp")
#                     if v is None or np.any(np.isnan(v)):
#                         break
#                     config.integrate_inplace(v, dt=0.01)
#                     error = np.linalg.norm(task.compute_error(config))
#                     err_vec = task.compute_error(config)   # 6D vector: [dx, dy, dz, dRx, dRy, dRz]
#                     pos_err = np.linalg.norm(err_vec[:3])  # magnitude of position error
#                     rot_err = np.linalg.norm(err_vec[3:])  # magnitude of rotation error

#                     if error < tol:
#                         success = True
#                         break
#                 if success:
#                     return config.q
#                 else:
#                     print("[Mink IK] Did not converge, fallback to dm_control IK")
#                     return None
#         if not use_mink:
#             # fallback to dm_control IK
#             try:
#                 # Initial guess from current qpos
#                 init_pos = np.zeros(len(joint_names))
#                 for i, name in enumerate(joint_names):
#                     init_pos[i] = self.physics.named.data.qpos[name]

#                 result = ik.qpos_from_site_pose(
#                     physics=self.physics,
#                     site_name=self.eef_site_name,
#                     target_pos=target_pos,
#                     target_quat=target_quat,
#                     joint_names=joint_names,
#                     tol=tol,
#                     max_steps=max_steps,
#                     inplace=False
#                 )
#                 if result.success:
#                     return result.qpos
#                 else:
#                     return None
#             except Exception as e:
#                 print(f"[dm_control IK failed] {str(e)}")
#                 return None



""""
Kinematics utilities for mobile manipulator robot.
Includes base (mecanum) kinematics, forward kinematics, and inverse kinematics.
"""

import numpy as np
from dm_control import mjcf
from dm_control.utils import inverse_kinematics as ik
from typing import Dict, List, Optional
from rl_mm.utils.transform_utils import mat2quat
import mink

class Kinematics:
    def __init__(self, robot, physics: Optional[mjcf.Physics] = None, use_mink: bool = True):
        """
        Initialize kinematics utility.
        
        Args:
            robot: MobileSO101 robot instance
            physics: MuJoCo physics instance from dm_control (optional)
            use_mink: If True, use Mink IK by default
        """
        import mujoco
        import mink

        self.robot = robot
        self.physics = physics or mjcf.Physics.from_mjcf_model(robot.mjcf_model)
        self.use_mink = use_mink

        self._prefix = "scene/"
        self.eef_site_name = self._prefix + (robot.eef_site if isinstance(robot.eef_site, str) else robot.eef_site.name)
        self.base_site_name = self._prefix + (robot.base_site if isinstance(robot.base_site, str) else robot.base_site.name)

        # ---------------------------
        # Mink configuration initialized immediately
        # ---------------------------
        self._mink_configuration = None
        if self.use_mink:
            try:
                # Load MuJoCo model directly from fixed path
                mj_model = mujoco.MjModel.from_xml_path("/home/sunny24/rl_mm/asset/SO101/so101_new_calib.xml")
                self._mink_configuration = mink.Configuration(mj_model)
                # Initialize joint positions to zeros
            except Exception as e:
                print(f"[Mink init failed] {str(e)}")
                self._mink_configuration = None


    # ---------------------------
    # Mecanum base kinematics
    # ---------------------------
    def compute_wheel_speeds_position(
        self,
        target_rel_pos: np.ndarray,
        current_rel_pos: np.ndarray,
        k: float = 1.0
    ) -> np.ndarray:
        """
        Tính wheel speeds chỉ cho di chuyển theo x,y (không xoay).
        """
        pos_error = target_rel_pos - current_rel_pos
        POS_THRESHOLD = 0.01

        if np.linalg.norm(pos_error) < POS_THRESHOLD:
            return np.zeros(4)

        dx, dy = k * pos_error
        return np.array([
            dx - dy,   # wheel 1
            dx - dy,   # wheel 2
            dx - dy,   # wheel 3
            dx - dy    # wheel 4
        ])


    def compute_wheel_speeds_yaw(
        self,
        target_rel_yaw: float,
        current_rel_yaw: float,
        k: float = 1.0
    ) -> np.ndarray:
        """
        Tính wheel speeds chỉ cho xoay quanh yaw.
        """
        yaw_error = target_rel_yaw - current_rel_yaw
        YAW_THRESHOLD = np.deg2rad(2)

        if abs(yaw_error) < YAW_THRESHOLD:
            return np.zeros(4)

        omega = k * yaw_error
        return np.array([
            -omega,  # wheel 1
            omega,  # wheel 2
            -omega,  # wheel 3
            omega   # wheel 4
        ])


    def compute_wheel_speeds(
        self,
        target_rel_pos: np.ndarray,
        target_rel_yaw: float,
        current_rel_pos: np.ndarray,
        current_rel_yaw: float,
        k: float = 1.0
    ) -> np.ndarray:
        """
        Wrapper: gộp cả position + yaw (nếu cần).
        """
        pos_cmd = self.compute_wheel_speeds_position(target_rel_pos, current_rel_pos, k)
        yaw_cmd = self.compute_wheel_speeds_yaw(target_rel_yaw, current_rel_yaw, k)
        return pos_cmd + yaw_cmd

    # ---------------------------
    # Forward Kinematics
    # ---------------------------
    def forward_kinematics(self) -> Dict[str, np.ndarray]:
        self.physics.forward()
        eef_pose = self.get_pose_from_site(self.eef_site_name)
        base_pose = None
        if self.base_site_name is not None:
            base_pose = self.get_pose_from_site(self.base_site_name)
        return {
            "eef_world_pos": eef_pose[:3],
            "eef_world_quat": eef_pose[3:],
            "base_world_pos": None if base_pose is None else base_pose[:3],
            "base_world_quat": None if base_pose is None else base_pose[3:],
        }

    def get_pose_from_site(self, site_name: str):
        site_id = self.physics.model.name2id(site_name, "site")
        pos = self.physics.data.site_xpos[site_id].copy()
        quat = mat2quat(self.physics.data.site_xmat[site_id].reshape(3, 3))
        return np.concatenate((pos, quat))

    def get_eef_pose(self, eef_site_name):
        return self.get_pose_from_site(eef_site_name)

    def get_base_pose(self, base_site_name):
        if base_site_name is None:
            return None
        return self.get_pose_from_site(base_site_name)

    # ---------------------------
    # Inverse Kinematics
    # ---------------------------
    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        joint_names: Optional[List[str]] = None,
        tol: float = 3e-3,
        max_steps: int = 100,
        use_mink: Optional[bool] = None
    ) -> Optional[np.ndarray]:
        """
        Compute inverse kinematics for target end-effector pose.
        
        Args:
            target_pos: Target position [x, y, z] in world frame
            target_quat: Optional target orientation as quaternion [x, y, z, w]
            joint_names: List of joint names to use for IK (default: robot.joints_arm)
            tol: Tolerance for IK convergence
            max_steps: Maximum number of iterations
            use_mink: Whether to use Mink IK (default: class default)
            
        Returns:
            np.ndarray: Joint positions if solution found, None otherwise
        """
        if use_mink is None:
            use_mink = self.use_mink

        if joint_names is None:
            if hasattr(self.robot, "joints_arm"):
                joint_names = [self._prefix + j.name for j in self.robot.joints_arm]
            else:
                raise AttributeError("Robot does not define arm joints.")

        if use_mink:
            config = self._mink_configuration
            if config is None:
                print("[Mink IK] Mink configuration not available, fallback to dm_control IK")
                use_mink = False
            else:
                # Create IK task
                # Tạo IK task
                task = mink.FrameTask(
                    "gripperframe",  # frame robot
                    "site",          # target type
                    position_cost=1.0,
                    orientation_cost=1
                )

                # Lấy transform hiện tại của gripper
                transform_init = config.get_transform_frame_to_world("gripperframe", "site")

                # Tạo rotation SO3 từ roll-pitch-yaw (radian)
                if target_quat is not None:  # target_rpy = (roll, pitch, yaw)
                    rotation_so3 = mink.SO3.from_rpy_radians(
                        roll=target_quat[0],
                        pitch=target_quat[1],
                        yaw=target_quat[2]
                    )
                else:
                    rotation_so3 = mink.SO3.identity()  # không xoay nếu không có target
                # Tạo SE3 target với rotation + translation
                target_se3 = mink.SE3.from_rotation_and_translation(rotation_so3, target_pos)

                # Kết hợp với transform hiện tại để có target cuối cùng trong world frame
                transform_target_to_world = transform_init @ target_se3

                # Gán target cho task
                task.set_target(transform_target_to_world)

                # Differential IK loop
                success = False
                for step in range(max_steps):
                    v = mink.solve_ik(config, [task], dt=0.01, solver="daqp")
                    if v is None or np.any(np.isnan(v)):
                        break
                    config.integrate_inplace(v, dt=0.01)
                    error = np.linalg.norm(task.compute_error(config))
                    if error < tol:
                        success = True
                        break
                if success:
                    return config.q
                else:
                    print("[Mink IK] Did not converge, fallback to dm_control IK")
                    return None
        if not use_mink:
            # fallback to dm_control IK
            try:
                # Initial guess from current qpos
                init_pos = np.zeros(len(joint_names))
                for i, name in enumerate(joint_names):
                    init_pos[i] = self.physics.named.data.qpos[name]

                result = ik.qpos_from_site_pose(
                    physics=self.physics,
                    site_name=self.eef_site_name,
                    target_pos=target_pos,
                    target_quat=target_quat,
                    joint_names=joint_names,
                    tol=tol,
                    max_steps=max_steps,
                    inplace=False
                )
                if result.success:
                    return result.qpos
                else:
                    return None
            except Exception as e:
                print(f"[dm_control IK failed] {str(e)}")
                return None
