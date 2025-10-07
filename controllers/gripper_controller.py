import numpy as np
from typing import Dict, List, Optional

class GripperController:
    def __init__(self, gripper_joints: List[str], kinematics, Kp: float = 0.5, 
                 force_threshold: float = 5.0, no_contact_timeout: int = 50):
        """
        Gripper controller with force feedback.
        """
        self.gripper_joints = gripper_joints
        self.kinematics = kinematics
        self.Kp = Kp
        self.force_threshold = force_threshold
        self.no_contact_timeout = no_contact_timeout

        self._target_width = None
        self._is_moving = False
        self._object_grasped = False
        self._no_contact_counter = 0
        self.width_tolerance = 0.001
        self._max_step_size = 1.0  # meters per step

        self._joint_ranges = [(self.kinematics.physics.named.model.jnt_range[j][0],
                               self.kinematics.physics.named.model.jnt_range[j][1])
                              for j in self.gripper_joints]
        self._min_width = 0
        self._max_width = min(r[1] for r in self._joint_ranges)

    def set_target(self, action_type: str, step: float) -> bool:
        """Set target position for gripper"""
        curr_width = np.mean([self.kinematics.physics.named.data.qpos[j]
                            for j in self.gripper_joints])
        if self._target_width is None:
            self._target_width = curr_width

        if action_type == "GRIPPER_OPEN":
            # Mở thẳng tới max width
            self._target_width = self._max_width
            self._object_grasped = False  # Reset khi mở
            self._no_contact_counter = 0
        elif action_type == "GRIPPER_CLOSE":
            # Giữ step để đóng
            self._target_width -= step

        # Clip vẫn cần để an toàn
        self._target_width = np.clip(self._target_width, self._min_width, self._max_width)
        self._is_moving = True
        return True

    def step(self, action_type: str, step: float) -> Dict[str, np.ndarray]:
        """Execute one control step"""
        self.set_target(action_type, step)
        return self.update_control_loop()

    def _get_gripper_forces(self) -> np.ndarray:
        """Get contact forces on gripper bodies"""
        total_force = 0.0
        try:
            ncon = self.kinematics.physics.data.ncon
            for i in range(ncon):
                contact = self.kinematics.physics.data.contact[i]
                geom1_name = self.kinematics.physics.model.id2name(contact.geom1, 'geom')
                geom2_name = self.kinematics.physics.model.id2name(contact.geom2, 'geom')
                for joint_name in self.gripper_joints:
                    if (geom1_name and joint_name.split('_')[0] in geom1_name) or \
                       (geom2_name and joint_name.split('_')[0] in geom2_name):
                        for j in range(self.kinematics.physics.data.nefc):
                            force = abs(self.kinematics.physics.data.efc_force[j])
                            if force > 0.01:
                                total_force += force
        except Exception as e:
            print(f"[Gripper] Error reading contact forces: {e}")

        if total_force < 0.01:
            try:
                joint_forces = [abs(self.kinematics.physics.named.data.qfrc_constraint[j])
                                for j in self.gripper_joints]
                total_force = np.sum(joint_forces)
            except:
                pass
        return np.array([total_force])

    def update_control_loop(self) -> Dict[str, np.ndarray]:
        """Main control loop with force feedback"""
        if not self._is_moving or self._target_width is None:
            curr_qpos = np.array([self.kinematics.physics.named.data.qpos[j] 
                                  for j in self.gripper_joints])
            return {"gripper_qpos": curr_qpos}

        curr_qpos = np.array([self.kinematics.physics.named.data.qpos[j] 
                              for j in self.gripper_joints])
        curr_width = np.mean(curr_qpos)

        # --- Force feedback ---
        forces = self._get_gripper_forces()
        max_force = forces[0] if len(forces) > 0 else 0.0
        # if max_force > 0.01 or self._no_contact_counter % 10 == 0:
        #     print(f"[Gripper] Force: {max_force:.4f}N, Width: {curr_width:.4f}, Target: {self._target_width:.4f}")

        # Xác định đang đóng hay mở
        closing = self._target_width < curr_width
        opening = self._target_width > curr_width

        # Chỉ áp dụng force feedback khi đang đóng
        if closing and max_force >= self.force_threshold:
            self._is_moving = False
            self._target_width = curr_width
            self._object_grasped = True
            self._no_contact_counter = 0
            # print(f"[Gripper] Object grasped! Force: {max_force:.2f}N, Width: {curr_width:.4f}")
            return {"gripper_qpos": np.full_like(curr_qpos, curr_width)}

        # Nếu đang đóng nhưng không có contact
        if closing:
            self._no_contact_counter += 1
            if self._no_contact_counter >= self.no_contact_timeout and not self._object_grasped:
                # print(f"[Gripper] No object detected, moving to min position")
                self._target_width = self._min_width
                self._no_contact_counter = 0
        else:
            self._no_contact_counter = 0

        # Width error check
        width_error = abs(self._target_width - curr_width)
        if width_error < self.width_tolerance:
            self._is_moving = False
            return {"gripper_qpos": np.full_like(curr_qpos, self._target_width)}

        # P-controller
        qpos_target = np.full_like(curr_qpos, self._target_width)
        qpos_error = qpos_target - curr_qpos
        qpos_cmd_raw = curr_qpos + self.Kp * qpos_error

        # Limit step size
        qpos_step = qpos_cmd_raw - curr_qpos
        qpos_step = np.clip(qpos_step, -self._max_step_size, self._max_step_size)
        qpos_cmd = curr_qpos + qpos_step

        return {"gripper_qpos": qpos_cmd}

    def is_at_target(self) -> bool:
        return not self._is_moving

    def is_grasping(self) -> bool:
        return self._object_grasped

    def get_grasp_width(self) -> Optional[float]:
        return self._target_width if self._object_grasped else None

    def stop(self) -> None:
        self._is_moving = False
        self._target_width = None
        self._object_grasped = False
        self._no_contact_counter = 0

    def reset(self) -> None:
        self.stop()
