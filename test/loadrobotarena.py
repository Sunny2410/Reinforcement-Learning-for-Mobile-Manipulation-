from rl_mm.robots import MobileSO101
from dm_control import mjcf
import imageio
from rl_mm.arena import StandardArena
from rl_mm.props import Primitive
from rl_mm.utils.transform_utils import mat2quat
import os, builtins, mujoco
from dm_control.utils import io as dm_io

# =========================
# Absolute path fix cho XML
# =========================
XML_ABS_PATH = os.path.abspath("rl_mm/asset/SO101/so101_new_calib.xml")
print("XML absolute path:", XML_ABS_PATH)

# -------------------------
# 1. Patch builtins.open
# -------------------------
_original_open = builtins.open
_original_normpath = os.path.normpath

def fix_path(path: str) -> str:
    if not isinstance(path, str):
        return path
    if "/rl_mm/" in path:  # chuẩn hóa khi dm_control đưa absolute
        path = "rl_mm/" + path.split("/rl_mm/")[-1]
    return _original_normpath(path)

def open_patched(file, *args, **kwargs):
    return _original_open(fix_path(file), *args, **kwargs)

builtins.open = open_patched

# -------------------------
# 2. Patch dm_control.GetResource
# -------------------------
_original_getresource = dm_io.GetResource

def getresource_patched(path, *args, **kwargs):
    return _original_getresource(fix_path(path), *args, **kwargs)

dm_io.GetResource = getresource_patched

# -------------------------
# 3. Patch mujoco.MjModel.from_xml_path
# -------------------------
_old_from_xml = mujoco.MjModel.from_xml_path

def from_xml_path_patched(path, *args, **kwargs):
    if "so101_new_calib.xml" in path:
        path = XML_ABS_PATH
    return _old_from_xml(path, *args, **kwargs)

mujoco.MjModel.from_xml_path = from_xml_path_patched

print("✅ Patched: open() + dm_control.GetResource + mujoco.MjModel.from_xml_path")

if __name__ == "__main__":
    # --- Tạo arena ---
    arena = StandardArena()
    # Flexcomp example
    flex_config = {
        "count": [8,8,8],
        "spacing": [0.1,0.1,0.1],
        "pos": [0,0,1],
        "radius": 0.01,
        "rgba": [0.68,0.53,0.38,1],
        "mass": 1,
        "condim": 3,
        "solref": [0.01,1],
        "solimp": [0.95,0.99,0.0001],
        "damping": 1,
        "young": 6e6,
        "poisson": 0.2,
        "thickness": 8e-3,
        "elastic2d": "bend",
        "elastic_damp": 1e-5,
        "dim" : 2,
    }

    flex_block = Primitive(type="flexcomp", composite_config=flex_config)


    # --- Tạo box (primitive) ---
    box_pos = [0.5, 0, 0.01]  # đặt trên sàn
    box_quat = [1, 0, 0, 0]
    arena.attach_free(flex_block.mjcf_model, pos=box_pos, quat=box_quat)

    # --- Load robot ---
    robot = MobileSO101(name="test_robot")
    print("Load robot successful:", robot)
    for joint in robot.joints_arm:
        print("Joint name:", joint.name)
    if getattr(robot, "baseframe", None) is not None: 
        print("Baseframe exists")

    # Attach robot vào arena
    arena.attach(robot.mjcf_model, pos=[0,0,0], quat=[1,0,0,0])
    
    # --- Tạo physics từ arena ---
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    physics.forward()

    # # --- Lấy body id và pose của box ---
    # box_body_id = physics.model.body_name2id(box.geom.name)
    # box_pos_world = physics.data.xpos[box_body_id].copy()
    # # Lấy rotation matrix rồi chuyển sang quaternion
    # box_quat_world = mat2quat(physics.data.xmat[box_body_id].copy().reshape(3,3))

    # print("Attach box successful:", box)
    # print("Box position (world):", box_pos_world)
    # print("Box orientation (quaternion, world):", box_quat_world)

    # --- In ra tất cả site ---
    print(f"Number of sites: {physics.model.nsite}")
    for i in range(physics.model.nsite):
        name = physics.model.id2name(i, 'site')
        pos = physics.data.site_xpos[i].copy()
        quat = mat2quat(physics.data.site_xmat[i].copy().reshape(3,3))
        print(f"Site {i}: {name}, position: {pos}, quaternion: {quat}")
    
    # --- Render ảnh ---
    img = physics.render(height=480, width=640, camera_id=-1)
    os.makedirs("rl_mm/test", exist_ok=True)
    imageio.imwrite("rl_mm/test/arena_robot.png", img)
    print("Saved image to rl_mm/test/arena_robot.png")
