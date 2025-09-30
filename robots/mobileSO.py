import os
from rl_mm.robots.MobileManipulator import MobileManipulator

_MOBILESO101_XML = os.path.join(
    os.path.dirname(__file__),
    '../asset/robotfull.xml',
)

# Các joint thuộc base (các bánh xe)
_BASE_JOINTS = (
    'fl_wheel_joint',
    'fr_wheel_joint',
    'rl_wheel_joint',
    'rr_wheel_joint',
)

# Các joint thuộc arm (cánh tay robot + gripper)
_ARM_JOINTS = (
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
)

# End-effector site
_EEF_SITE = 'gripperframe'

_BASE_SITE = 'baseframe'

class MobileSO101(MobileManipulator):
    def __init__(self, name: str = None):
        super().__init__(
            xml_path=_MOBILESO101_XML,
            eef_site_name=_EEF_SITE,
            base_site_name=_BASE_SITE,
            joints_arm=_ARM_JOINTS,
            joints_base=_BASE_JOINTS,
            name=name
        )



