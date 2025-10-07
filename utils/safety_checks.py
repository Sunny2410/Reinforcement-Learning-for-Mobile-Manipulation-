import numpy as np
from typing import Dict, List, Tuple

class SafetyChecker:
    """
    Safety checker for mobile manipulator:
    - Joint limits violation
    - Ground collision (arm links hitting floor)
    - Base collision (arm hitting robot base)
    - Self-collision (arm links colliding with each other)
    - Environment collision (arm hitting walls/obstacles, EXCEPT grasped object)
    """
    
    def __init__(self, physics, kinematics, arm_joints: List[str], gripper_joints: List[str]):
        self.physics = physics
        self.kinematics = kinematics
        self.arm_joints = arm_joints
        self.gripper_joints = gripper_joints
        
        # ========== THRESHOLDS ==========
        self.ground_height = 0.01  # 5cm above ground
        self.base_radius = 0.25    # 25cm radius from base center
        self.base_height_threshold = 0.3  # 30cm above base
        self.joint_limit_buffer = np.radians(5)  # 5 degrees buffer
        
        # ========== PENALTY WEIGHTS ==========
        self.penalty_joint_limit = 10.0
        self.penalty_ground = 20.0
        self.penalty_base_collision = 15.0
        self.penalty_self_collision = 25.0
        self.penalty_env_collision = 30.0  # Collision với tường/obstacles
        
        # ========== GRASPING ==========
        self.grasping_force_threshold = 5.0
        
        # Arm link geoms (để check collision với ground/base)
        self.arm_geom_keywords = [
            'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'moving_jaw_so101_v1'
        ]
        
    def check_all_violations(self, grasping_force: float = 0.0) -> Dict:
        """
        Check all safety violations and print debug info.
        """
        penalties = {
            'joint_limits': 0.0,
            'ground_collision': 0.0,
            'base_collision': 0.0,
            'self_collision': 0.0,
            'env_collision': 0.0,
            'total': 0.0,
            'info': {}
        }

        is_grasping = grasping_force > self.grasping_force_threshold

        # 1. Joint limits
        joint_penalty, joint_info = self._check_arm_limits()
        penalties['joint_limits'] = joint_penalty
        penalties['info']['joint_violations'] = joint_info
        print("\n[DEBUG] Joint Limits Violations:", joint_info)

        # 2. Ground collision
        if not is_grasping:
            ground_penalty, ground_info = self._check_ground_collision()
            penalties['ground_collision'] = ground_penalty
            penalties['info']['ground_collisions'] = ground_info
            print("[DEBUG] Ground Collisions:", ground_info)
        else:
            penalties['info']['grasping_active'] = True
            print("[DEBUG] Grasping active, skipping ground collision.")

        # 3. Base collision
        if not is_grasping:
            base_penalty, base_info = self._check_base_collision()
            penalties['base_collision'] = base_penalty
            penalties['info']['base_collisions'] = base_info
            print("[DEBUG] Base Collisions:", base_info)
        
        # 4. Self-collision
        self_penalty, self_info = self._check_self_collision()
        if is_grasping:
            self_penalty *= 0.5
        penalties['self_collision'] = self_penalty
        penalties['info']['self_collisions'] = self_info
        print("[DEBUG] Self Collisions:", self_info)

        # 5. Environment collision
        env_penalty, env_info = self._check_environment_collision(is_grasping)
        penalties['env_collision'] = env_penalty
        penalties['info']['env_collisions'] = env_info
        print("[DEBUG] Environment Collisions:", env_info)

        # Total
        penalties['total'] = sum([
            penalties['joint_limits'],
            penalties['ground_collision'],
            penalties['base_collision'],
            penalties['self_collision'],
            penalties['env_collision']
        ])
        print("[DEBUG] Total Penalty:", penalties['total'])

        # Extra info
        safety_info = self.get_safety_info()
        print("[DEBUG] Safety Info:", safety_info)

        return penalties

    
    def _check_arm_limits(self) -> Tuple[float, List[str]]:
        """Check joint limits"""
        penalty = 0.0
        violations = []
        
        for joint_name in self.arm_joints:
            try:
                joint_id = self.physics.model.name2id(joint_name, 'joint')
                qpos_addr = self.physics.model.jnt_qposadr[joint_id]
                pos = self.physics.data.qpos[qpos_addr]
                limits = self.physics.model.jnt_range[joint_id]
                
                # Lower limit
                if pos < (limits[0] + self.joint_limit_buffer):
                    violation = (limits[0] + self.joint_limit_buffer) - pos
                    penalty += violation * self.penalty_joint_limit
                    violations.append(
                        f"{joint_name}: {np.degrees(pos):.1f}° < {np.degrees(limits[0]):.1f}°"
                    )
                
                # Upper limit
                elif pos > (limits[1] - self.joint_limit_buffer):
                    violation = pos - (limits[1] - self.joint_limit_buffer)
                    penalty += violation * self.penalty_joint_limit
                    violations.append(
                        f"{joint_name}: {np.degrees(pos):.1f}° > {np.degrees(limits[1]):.1f}°"
                    )
                    
            except (KeyError, IndexError) as e:
                continue
        
        return penalty, violations
    
    def _check_ground_collision(self) -> Tuple[float, List[str]]:
        """
        Check if ANY arm link collides with ground.
        Returns penalty and list of violating links.
        """
        penalty = 0.0
        violations = []
        
        try:
            ncon = self.physics.data.ncon
            
            for i in range(ncon):
                contact = self.physics.data.contact[i]
                geom1_name = self.physics.model.id2name(contact.geom1, 'geom') or ""
                geom2_name = self.physics.model.id2name(contact.geom2, 'geom') or ""
                
                # Check if one geom is ground (floor) and other is arm link
                is_ground_contact = False
                arm_geom = None
                
                if 'floor' in geom1_name.lower() or 'ground' in geom1_name.lower():
                    # geom1 is ground, check if geom2 is arm
                    if self._is_arm_geom(geom2_name):
                        is_ground_contact = True
                        arm_geom = geom2_name
                
                elif 'floor' in geom2_name.lower() or 'ground' in geom2_name.lower():
                    # geom2 is ground, check if geom1 is arm
                    if self._is_arm_geom(geom1_name):
                        is_ground_contact = True
                        arm_geom = geom1_name
                
                if is_ground_contact:
                    penetration = abs(contact.dist) if contact.dist < 0 else 0
                    violation_cm = penetration * 100
                    penalty += violation_cm * self.penalty_ground
                    violations.append(f"{arm_geom} hitting ground")
            
        except Exception as e:
            print(f"[SafetyChecker] Error checking ground collision: {e}")
        
        return penalty, violations
    
    def _check_base_collision(self) -> Tuple[float, List[str]]:
        """
        Check if arm links collide with robot base.
        Returns penalty and list of violations.
        """
        penalty = 0.0
        violations = []
        
        try:
            ncon = self.physics.data.ncon
            
            for i in range(ncon):
                contact = self.physics.data.contact[i]
                geom1_name = self.physics.model.id2name(contact.geom1, 'geom') or ""
                geom2_name = self.physics.model.id2name(contact.geom2, 'geom') or ""
                
                # Check if one is base and other is arm
                is_base_collision = False
                arm_geom = None
                
                if self._is_base_geom(geom1_name) and self._is_arm_geom(geom2_name):
                    is_base_collision = True
                    arm_geom = geom2_name
                elif self._is_base_geom(geom2_name) and self._is_arm_geom(geom1_name):
                    is_base_collision = True
                    arm_geom = geom1_name
                
                if is_base_collision:
                    penetration = abs(contact.dist) if contact.dist < 0 else 0
                    violation_cm = penetration * 100
                    penalty += violation_cm * self.penalty_base_collision
                    violations.append(f"{arm_geom} hitting base")
            
        except Exception as e:
            print(f"[SafetyChecker] Error checking base collision: {e}")
        
        return penalty, violations
    
    def _check_self_collision(self) -> Tuple[float, List[str]]:
        """
        Check arm links colliding with each other.
        EXCLUDES:
        - Adjacent links (by design)
        - Gripper internal collision (fingers closing)
        """
        penalty = 0.0
        collisions = []
        
        try:
            ncon = self.physics.data.ncon
            
            for i in range(ncon):
                contact = self.physics.data.contact[i]
                geom1_name = self.physics.model.id2name(contact.geom1, 'geom') or ""
                geom2_name = self.physics.model.id2name(contact.geom2, 'geom') or ""
                
                # Both must be robot parts (scene/)
                if not ('scene/' in geom1_name and 'scene/' in geom2_name):
                    continue
                
                # Skip gripper internal collision
                if self._is_gripper_internal_collision(geom1_name, geom2_name):
                    continue
                
                # Skip adjacent links
                if self._is_adjacent_link(geom1_name, geom2_name):
                    continue
                
                # Check if both are arm links
                if self._is_arm_geom(geom1_name) and self._is_arm_geom(geom2_name):
                    penetration = abs(contact.dist) if contact.dist < 0 else 0
                    if penetration > 0:
                        penalty += self.penalty_self_collision
                        collisions.append(f"{geom1_name} <-> {geom2_name}")
            
        except Exception as e:
            print(f"[SafetyChecker] Error checking self-collision: {e}")
        
        return penalty, collisions
    
    def _check_environment_collision(self, is_grasping: bool) -> Tuple[float, List[str]]:
        """
        Check arm collision with environment (walls, obstacles).
        EXCLUDES:
        - Robot itself (scene/)
        - Grasped object (unnamed_model/) when is_grasping=True
        """
        penalty = 0.0
        collisions = []
        
        try:
            ncon = self.physics.data.ncon
            
            for i in range(ncon):
                contact = self.physics.data.contact[i]
                geom1_name = self.physics.model.id2name(contact.geom1, 'geom') or ""
                geom2_name = self.physics.model.id2name(contact.geom2, 'geom') or ""
                
                # Determine which is arm and which is environment
                arm_geom = None
                env_geom = None
                
                if self._is_arm_geom(geom1_name):
                    arm_geom = geom1_name
                    env_geom = geom2_name
                elif self._is_arm_geom(geom2_name):
                    arm_geom = geom2_name
                    env_geom = geom1_name
                else:
                    # Neither is arm → skip
                    continue
                
                # Skip if env_geom is robot itself
                if 'scene/' in env_geom:
                    continue
                
                # CRITICAL: Skip if grasping and env_geom is the grasped object
                if is_grasping and 'unnamed_model/' in env_geom:
                    continue
                
                # This is arm hitting environment → penalize
                penetration = abs(contact.dist) if contact.dist < 0 else 0
                if penetration > 0:
                    penalty += self.penalty_env_collision
                    collisions.append(f"{arm_geom} hitting {env_geom}")
            
        except Exception as e:
            print(f"[SafetyChecker] Error checking environment collision: {e}")
        
        return penalty, collisions
    
    # ========== HELPER METHODS ==========
    
    def _is_arm_geom(self, geom_name: str) -> bool:
        """Check if geom belongs to arm links"""
        if not geom_name or 'scene/' not in geom_name:
            return False
        
        # Exclude base, wheels, gripper
        exclude = ['base', 'wheel', 'gripper', 'finger', 'palm']
        if any(ex in geom_name.lower() for ex in exclude):
            return False
        
        # Check for arm keywords
        return any(kw in geom_name.lower() for kw in self.arm_geom_keywords)
    
    def _is_base_geom(self, geom_name: str) -> bool:
        """Check if geom belongs to robot base"""
        if not geom_name or 'scene/' not in geom_name:
            return False
        return 'base' in geom_name.lower() or 'chassis' in geom_name.lower()
    
    def _is_gripper_internal_collision(self, geom1: str, geom2: str) -> bool:
        """Check if collision is internal gripper (fingers closing)"""
        gripper_parts = ['gripper', 'finger', 'palm', 'left_finger', 'right_finger']
        
        has_gripper1 = any(part in geom1.lower() for part in gripper_parts)
        has_gripper2 = any(part in geom2.lower() for part in gripper_parts)
        
        return has_gripper1 and has_gripper2
    
    def _is_adjacent_link(self, geom1: str, geom2: str) -> bool:
        """Check if two geoms are from adjacent links (allowed by design)"""
        adjacent_pairs = [
            ('shoulder_pan', 'shoulder_lift'),
            ('shoulder_lift', 'elbow_flex'),
            ('elbow_flex', 'forearm'),
            ('forearm', 'wrist_flex'),
            ('wrist_flex', 'wrist_roll'),
            ('wrist_roll', 'gripper'),
        ]
        
        for link1, link2 in adjacent_pairs:
            if ((link1 in geom1.lower() and link2 in geom2.lower()) or
                (link2 in geom1.lower() and link1 in geom2.lower())):
                return True
        
        return False
    
    def get_safety_info(self) -> Dict:
        """Get current safety status for logging"""
        info = {}
        
        try:
            fk = self.kinematics.forward_kinematics()
            eef_pos = fk["eef_world_pos"]
            base_pos = fk["base_world_pos"]
            
            info['eef_height'] = float(eef_pos[2])
            info['ground_clearance'] = float(eef_pos[2])
            
            if base_pos is not None:
                dist_from_base = float(np.linalg.norm(eef_pos[:2] - base_pos[:2]))
                info['dist_from_base_xy'] = dist_from_base
                info['inside_base_zone'] = dist_from_base < self.base_radius
            
            # Joint positions
            joint_positions = {}
            for joint_name in self.arm_joints:
                try:
                    joint_id = self.physics.model.name2id(joint_name, 'joint')
                    qpos_addr = self.physics.model.jnt_qposadr[joint_id]
                    pos = self.physics.data.qpos[qpos_addr]
                    joint_positions[joint_name] = float(np.degrees(pos))
                except:
                    pass
            info['joint_positions_deg'] = joint_positions
            
        except Exception as e:
            print(f"[SafetyChecker] Error getting safety info: {e}")
        
        return info