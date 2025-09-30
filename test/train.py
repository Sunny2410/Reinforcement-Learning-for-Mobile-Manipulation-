import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from PIL import Image  # để lưu ảnh

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
