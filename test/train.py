import gymnasium as gym
import rl_mm
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# -----------------------------
# 1. Create environment
# -----------------------------
env_id = "rl_mm/SO101-v1"
env = gym.make(env_id)

# -----------------------------
# 2. Define PPO model
# -----------------------------
model = PPO(
    policy="MlpPolicy",
    env=env,
    batch_size=64,
    n_steps=2048,
    gamma=0.99,
    learning_rate=3e-4,
    verbose=0   # tắt log mặc định, để mình điều khiển bằng tqdm
)

# -----------------------------
# 3. Callback dùng tqdm
# -----------------------------
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training progress")

    def _on_step(self) -> bool:
        # cập nhật thanh tiến độ theo số timesteps đã chạy
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        self.pbar.n = self.total_timesteps
        self.pbar.refresh()
        self.pbar.close()

# -----------------------------
# 4. Train the model
# -----------------------------
total_timesteps = 1000000
callback = ProgressBarCallback(total_timesteps)
model.learn(total_timesteps=total_timesteps, callback=callback)

# -----------------------------
# 5. Save the model
# -----------------------------
model.save("ppo_tqdm_so101")

# -----------------------------
# 6. Test the trained model
# -----------------------------
test_env = gym.make(env_id, render_mode="human")
obs, info = test_env.reset(seed=42)

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print("Test action chosen:", action)  # In ra action test
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()
