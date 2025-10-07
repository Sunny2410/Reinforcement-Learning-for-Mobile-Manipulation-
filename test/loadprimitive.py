from rl_mm.props import Primitive
from rl_mm.robots import MobileSO101
from rl_mm.arena import StandardArena
from dm_control import mjcf
import imageio
import os
import numpy as np
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

from rl_mm.props import Primitive
from rl_mm.robots import MobileSO101
from rl_mm.arena import StandardArena
from dm_control import mjcf
import imageio
import os
import numpy as np

if __name__ == "__main__":
    # 1️⃣ Tạo môi trường cơ bản
    arena = StandardArena()

    # 2️⃣ Thêm robot có thể di chuyển (freejoint)
    robot = MobileSO101(name="test_robot")
    arena.attach_free(robot.mjcf_model, pos=[0, 0, 0.05])

    # 3️⃣ Thêm hộp có thể di chuyển (freejoint)
    box = Primitive(type="box", size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1],mass=0.03)
    arena.attach_free(box.mjcf_model, pos=[0.4, 0, 0.05])

    # 4️⃣ Build physics
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    # # 5️⃣ Di chuyển robot và box trong runtime
    # # -------------------------------------------------
    # # Lấy số lượng qpos hiện tại
    # nq = physics.model.nq
    # print(f"\n🔧 Tổng số qpos: {nq}")

    # # Mỗi freejoint có 7 giá trị: [x, y, z, qw, qx, qy, qz]
    # # Giả sử thứ tự là: robot (7 giá trị đầu) → box (7 giá trị tiếp)
    # # Robot: di chuyển tới [0.6, -0.2, 0.05] và quay nhẹ quanh z
    # robot_pos = [0.6, -0.2, 0.05]
    # robot_quat = [0.9239, 0, 0, 0.3827]  # quay 45 độ quanh z

    # # Box: di chuyển tới [0.2, 0.3, 0.05]
    # box_pos = [0.2, 0.3, 0.05]
    # box_quat = [1, 0, 0, 0]

    # # Gán lại qpos
    # physics.data.qpos[0:3] = robot_pos
    # physics.data.qpos[3:7] = robot_quat
    # physics.data.qpos[7:10] = box_pos
    # physics.data.qpos[10:14] = box_quat
    # physics.forward()
    # # -------------------------------------------------

    # # 6️⃣ Render ảnh
    # img = physics.render(height=480, width=480, camera_id=-1)

    # # 7️⃣ Lưu ảnh
    # save_path = "rl_mm/test/env.png"
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # imageio.imwrite(save_path, img)
    # print(f"✅ Environment image saved at: {save_path}")

    # # 8️⃣ Lấy danh sách body và vị trí
    # body_ids = range(physics.model.nbody)
    # body_names = [physics.model.id2name(i, "body") for i in body_ids]
    # body_positions = physics.data.xpos
    # body_quats = physics.data.xquat

    # print("\n📦 Danh sách các body trong mô hình:")
    # for i, name in enumerate(body_names):
    #     pos = np.round(body_positions[i], 3)
    #     quat = np.round(body_quats[i], 3)
    #     print(f"  {i:02d}. {name:25s} → pos={pos}, quat={quat}")

    # # 9️⃣ In riêng body của robot
    # print("\n🤖 Các body thuộc robot test_robot:")
    # for i, name in enumerate(body_names):
    #     if name and name.startswith("test_robot"):
    #         pos = np.round(body_positions[i], 3)
    #         print(f"  {i:02d}. {name:25s} → pos={pos}")

    # # 🔟 Kiểm tra robot có joint hay không
    # try:
    #     root_body_id = physics.model.name2id("test_robot/", "body")
    #     has_joint = np.any(physics.model.body_jntnum[root_body_id] > 0)
    # except Exception:
    #     has_joint = False

    # if has_joint:
    #     print("✅ Robot có joint → có thể di chuyển trong runtime.")
    # else:
    #     print("❌ Robot không có joint → cố định, không di chuyển được.")
    xml_text = arena.mjcf_model.to_xml_string()
    print("=== MJCF XML ===")
    print(xml_text)

    # Lưu ra file nếu muốn
    with open("arena_dump.xml", "w") as f:
        f.write(xml_text)
    print("✅ XML dumped to arena_dump.xml")
    