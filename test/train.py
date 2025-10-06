import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from PIL import Image  # để lưu ảnh


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


# -----------------------------
# 1. Setup environment
# -----------------------------
env_id = "rl_mm/SO101-v1"
test_env = gym.make(env_id, render_mode="rgb_array")  # <-- đổi từ human sang rgb_array
obs, info = test_env.reset(seed=42)

# Lấy env gốc nếu cần truy cập manager / robot
raw_env = test_env.unwrapped

# -----------------------------
# 2. Load trained PPO model
# -----------------------------
model_path = "ppo_subproc_so101"  # đường dẫn model
model = PPO.load(
    model_path,
    custom_objects={
        "lr_schedule": lambda _: 3e-4,   # override learning rate schedule
        "clip_range": lambda _: 0.2      # override clip_range
    }
)

# -----------------------------
# 3. Test loop + save frames
# -----------------------------
n_steps = 20
for step in range(n_steps):
    # Dự đoán action từ model
    action, _states = model.predict(obs, deterministic=True)
    
    # Convert action array -> int nếu cần
    if isinstance(action, np.ndarray):
        action = int(action.item())

    print(f"Step {step} - Action chosen:", action)

    # Thực hiện action
    obs, reward, terminated, truncated, info = test_env.step(action)

    # Render ra array (RGB) và lưu ảnh
    frame = test_env.render()  # trả về numpy array
    img = Image.fromarray(frame)
    img.save(f"frames/frame_{step:04d}.png")  # lưu theo thứ tự

    # Reset nếu episode kết thúc
    if terminated or truncated:
        obs, info = test_env.reset()

# -----------------------------
# 4. Close environment
# -----------------------------
test_env.close()
