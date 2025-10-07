import gymnasium
import rl_mm
import os, builtins, mujoco
import numpy as np
import imageio
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

# -------------------------
# 4. Run Environment
# -------------------------
env = gymnasium.make("rl_mm/SO101-v2", render_mode="rgb_array")
obs, info = env.reset(seed=42)
print("▶️ Env reset lần đầu")

# Thư mục lưu ảnh
os.makedirs("debug_images", exist_ok=True)

print("Nhập số action rồi nhấn Enter. Nhập q để thoát.")
idx = None
step_count = 0
episode_count = 1

try:
    while True:
        # Nhập action từ bàn phím
        if idx is None:
            key = input(f"[Episode {episode_count} | Step {step_count}] Action index: ").strip()
            if key.lower() == 'q':
                print("⏹ Thoát.")
                break
            if key.isdigit():
                tmp_idx = int(key) - 1
                if 0 <= tmp_idx < env.action_space.n:
                    idx = tmp_idx
                else:
                    print(f"⚠️ Action {tmp_idx+1} không hợp lệ (1-{env.action_space.n})")
                    continue
            else:
                print("⚠️ Nhập số từ 1 đến", env.action_space.n, "hoặc q để thoát")
                continue

        # Step
        obs, reward, terminated, truncated, info = env.step(idx)
        # === In obs chi tiết ===
        if isinstance(obs, dict):
            print("🔍 Obs keys:", list(obs.keys()))
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    print(f"\n📊 {k}: shape={v.shape}, dtype={v.dtype}")
                    # Nếu mảng nhỏ thì in hết, nếu lớn thì in 1 vài phần tử
                    flat = v.flatten()
                    if flat.size <= 50:
                        print(v)
                    else:
                        print(f"  min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}")
                        print("  sample:", flat[:10], "...")
                else:
                    print(f"\n📄 {k}: {v}")
        else:
            print("🔍 Obs:", obs)


        step_count += 1
        print(f"Step {step_count} | Action {idx} | Reward {reward:.3f} | Terminated={terminated} | Truncated={truncated}")

        # ======= 🖼️ In/Lưu ảnh quan sát =========
# ======= 🖼️ In/Lưu ảnh quan sát =========
        if isinstance(obs, dict):
            img_key = None
            for k in obs.keys():
                if 'image' in k or 'rgb' in k:
                    img_key = k
                    break

            if img_key is not None:
                try:
                    img = obs[img_key]
                    # Nếu float [0,1], scale lên 0–255
                    if np.issubdtype(img.dtype, np.floating):
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    # Đảm bảo 3 channel
                    if img.ndim == 2:
                        img = np.stack([img]*3, axis=-1)
                    elif img.shape[-1] != 3:
                        img = img[..., :3]

                    filename = f"debug_images/ep{episode_count:02d}_step{step_count:04d}.png"
                    imageio.imwrite(filename, img)
                    print(f"🖼️ Saved image from '{img_key}' → {filename}")
                except Exception as e:
                    print(f"⚠️ Cannot save image from '{img_key}': {e}")

        # =========================================

        env.render()

        if not env.manager.is_any_moving():
            idx = None

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"🔄 Episode {episode_count} kết thúc ({reason}). Reset env...")
            obs, info = env.reset()
            episode_count += 1
            step_count = 0

finally:
    env.close()
    print("✅ Env closed")
