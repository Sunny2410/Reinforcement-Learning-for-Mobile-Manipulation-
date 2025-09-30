import gymnasium as gym
import rl_mm
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

# -----------------------------
# 1. Create environment
# -----------------------------
env_id = "rl_mm/SO101-v0"
env = gym.make(env_id)

# -----------------------------
# 2. Define PPO model with HER replay buffer
# -----------------------------
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3 import PPO

from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",
    env=env,
    batch_size=64,
    n_steps=2048,
    gamma=0.99,
    learning_rate=3e-4,
    verbose=1
)


# -----------------------------
# 3. Train the model
# -----------------------------
total_timesteps = 1000000
model.learn(total_timesteps=total_timesteps)

# -----------------------------
# 4. Save the model
# -----------------------------
model.save("ppo_her_replaybuffer_so101")

# -----------------------------
# 5. Test the trained model
# -----------------------------
test_env = gym.make(env_id, render_mode="rgb_array")
obs, info = test_env.reset(seed=42)

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()
