import gymnasium
import rl_mm
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


env = gymnasium.make("rl_mm/SO101-v1", render_mode="human")
obs, info = env.reset(seed=42)

print("Nhập số action rồi nhấn Enter. Nhập q để thoát.")
idx = None

try:
    while True:
        # Nếu chưa có action đang chạy, đọc input mới
        if idx is None:
            key = input("Action index: ").strip()
            if key.lower() == 'q':
                break
            if key.isdigit():
                tmp_idx = int(key) - 1  # nếu muốn nhập 1..N thay vì 0..N-1
                if 0 <= tmp_idx < env.action_space.n:
                    idx = tmp_idx
                else:
                    print(f"Action {tmp_idx+1} không hợp lệ (1-{env.action_space.n})")
                    continue
            else:
                print("Nhập số từ 1 đến", env.action_space.n, "hoặc q để thoát")
                continue

        # Nếu có action, gọi step liên tục cho đến khi controller xong
        obs, reward, terminated, truncated, info = env.step(idx)
        env.render()
        
        if not env.manager.is_any_moving():  # controller xong
            idx = None

        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()
