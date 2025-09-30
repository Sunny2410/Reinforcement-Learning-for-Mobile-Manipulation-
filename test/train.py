import gymnasium
import rl_mm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 1. Tạo vectorized environment để train nhanh hơn
env_id = "rl_mm/SO101-v0"
num_envs = 4
env = make_vec_env(env_id, n_envs=num_envs)

# 2. Tạo model PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2
)

# 3. Train model
total_timesteps = 100_000  # chỉnh số timesteps theo nhu cầu
model.learn(total_timesteps=total_timesteps)

# 4. Lưu model
model.save("ppo_so101")

# 5. Test model
test_env = gymnasium.make(env_id, render_mode="human")
obs, info = test_env.reset(seed=42)

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    test_env.render()
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()
