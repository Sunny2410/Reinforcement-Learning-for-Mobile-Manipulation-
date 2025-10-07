"""
Ví dụ training với Vision Encoders - SIMPLIFIED VERSION
Tạo file: training/train_with_vision.py
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import torch
import numpy as np

# Import your environment (should be registered in Gym)
# from rl_mm.envs import SO101Arm2  

# Import feature extractors
from .feature_extractors import (
    VisionStateExtractor,
    VisionOnlyExtractor,
    StateOnlyExtractor
)


# ============================================================
# ENVIRONMENT FACTORY
# ============================================================
def make_env(rank, seed=0, env_id="rl_mm/SO101-v2"):
    """Create a Gym environment instance for vectorized training"""
    def _init():
        env = gym.make(env_id, render_mode=None)
        env = Monitor(env)
        # Set seeds
        env.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        return env
    return _init


# ============================================================
# STRATEGY 1: Vision + State (DINOv2) - RECOMMENDED
# ============================================================
def train_multimodal_dinov2(num_envs: int = 4, total_timesteps: int = 200_000):
    print("\n" + "="*50)
    print("TRAINING: DINOv2 (frozen) + State")
    print("="*50 + "\n")

    env_fns = [make_env(rank=i, seed=42) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns)
    eval_env = DummyVecEnv([make_env(rank=999, seed=123)])

    policy_kwargs = dict(
        features_extractor_class=VisionStateExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            vision_encoder='dinov2',
            vision_encoder_kwargs={
                'model_name': 'small',
                'freeze': True
            },
            state_hidden_dim=64,
            normalize_state=True
        ),
        net_arch=[256, 256],
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/dinov2_multimodal/",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/dinov2_multimodal/",
        name_prefix="ppo_dinov2"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/dinov2_multimodal/best/",
        log_path="./logs/dinov2_multimodal/eval/",
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save("./models/dinov2_multimodal/final_model")
    env.close()
    eval_env.close()
    print("✅ Training completed and model saved!")
    return model


# ============================================================
# STRATEGY 2: Vision Only (DINOv2)
# ============================================================
def train_vision_only(num_envs: int = 4):
    print("\n" + "="*50)
    print("TRAINING: Vision Only (DINOv2)")
    print("="*50 + "\n")

    from gymnasium import ObservationWrapper

    class ImageOnlyWrapper(ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = env.observation_space['image']

        def observation(self, obs):
            return obs['image']

    def make_env_img(rank=0):
        env = gym.make("rl_mm/SO101-v2", render_mode=None)
        env = ImageOnlyWrapper(env)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env_img for _ in range(num_envs)])
    eval_env = DummyVecEnv([make_env_img])

    policy_kwargs = dict(
        features_extractor_class=VisionOnlyExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            vision_encoder='dinov2',
            vision_encoder_kwargs={'model_name': 'small', 'freeze': True}
        ),
        net_arch=[256, 256],
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./logs/vision_only/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/vision_only/",
        name_prefix="ppo_vision"
    )

    model.learn(total_timesteps=500_000, callback=[checkpoint_callback])
    model.save("./models/vision_only/final_model")
    return model


# ============================================================
# STRATEGY 3: State Only (Baseline)
# ============================================================
def train_state_only(num_envs: int = 4):
    print("\n" + "="*50)
    print("TRAINING: State Only (Baseline)")
    print("="*50 + "\n")

    from gymnasium import ObservationWrapper

    class StateOnlyWrapper(ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = env.observation_space['state']

        def observation(self, obs):
            return obs['state']

    def make_env_state(rank=0):
        env = gym.make("rl_mm/SO101-v2", render_mode=None)
        env = StateOnlyWrapper(env)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env_state for _ in range(num_envs)])
    eval_env = DummyVecEnv([make_env_state])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./logs/state_only/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/state_only/",
        name_prefix="ppo_state"
    )

    model.learn(total_timesteps=200_000, callback=[checkpoint_callback])
    model.save("./models/state_only/final_model")
    return model


# ============================================================
# TESTING TRAINED MODEL
# ============================================================
def test_model(model_path, n_episodes=10, render=True):
    print(f"\nTesting model: {model_path}")
    model = PPO.load(model_path)
    env = gym.make("rl_mm/SO101-v2", render_mode='human' if render else None)

    rewards = []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
            if info.get('reached', 0):
                successes += 1
        rewards.append(total_r)
        print(f"Episode {ep+1}: Reward={total_r:.2f}")

    print(f"\nMean reward: {np.mean(rewards):.2f}, Success rate: {successes/n_episodes*100:.1f}%")
    env.close()


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='multimodal',
                        choices=['multimodal', 'vision_only', 'state_only'],
                        help='Training strategy')
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--total_timesteps', type=int, default=200_000)
    parser.add_argument('--test', type=str, default=None)
    args = parser.parse_args()

    if args.test:
        test_model(args.test)
    else:
        if args.strategy == 'multimodal':
            train_multimodal_dinov2(args.num_envs, args.total_timesteps)
        elif args.strategy == 'vision_only':
            train_vision_only(args.num_envs)
        elif args.strategy == 'state_only':
            train_state_only(args.num_envs)
