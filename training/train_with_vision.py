"""
Ví dụ training với Vision Encoders
Tạo file: training/train_with_vision.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

# Import your environment
from rl_mm.envs import SO101Arm2  # Adjust import path

# Import feature extractors
from training.feature_extractors import (
    VisionStateExtractor,
    VisionOnlyExtractor,
    StateOnlyExtractor
)


# ============================================================
# STRATEGY 1: Vision + State (DINOv2) - RECOMMENDED
# ============================================================
def train_multimodal_dinov2():
    """
    Chiến lược tốt nhất: DINOv2 (frozen) + State
    - Fast training (50k-200k steps)
    - Sample efficient
    - Good generalization
    """
    print("\n" + "="*70)
    print("TRAINING: DINOv2 (frozen) + State")
    print("="*70 + "\n")
    
    # Create environment
    def make_env():
        env = SO101Arm2(render_mode=None)
        env = Monitor(env)
        return env
    
    # Vectorized environment (4 parallel envs)
    env = DummyVecEnv([make_env for _ in range(4)])
    
    # Eval environment
    eval_env = DummyVecEnv([make_env])
    
    # Policy kwargs với DINOv2
    policy_kwargs = dict(
        features_extractor_class=VisionStateExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            vision_encoder='dinov2',
            vision_encoder_kwargs={
                'model_name': 'small',  # 384 dim, fast
                'freeze': True          # Frozen pre-trained weights
            },
            state_hidden_dim=64,
            normalize_state=True
        ),
        net_arch=[256, 256],  # Policy/Value head architecture
    )
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
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
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/dinov2_multimodal/",
        name_prefix="ppo_dinov2"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/dinov2_multimodal/best/",
        log_path="./logs/dinov2_multimodal/eval/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=200000,  # 200k steps (~2-4 hours)
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    model.save("./models/dinov2_multimodal/final_model")
    print("✓ Training completed!")
    
    return model


# ============================================================
# STRATEGY 2: Vision Only (DINOv2) - Pure Vision-based
# ============================================================
def train_vision_only():
    """
    Vision-based RL: Chỉ dùng image
    - Cần nhiều steps hơn (500k-1M)
    - Harder to train
    - End-to-end vision control
    """
    print("\n" + "="*70)
    print("TRAINING: Vision Only (DINOv2)")
    print("="*70 + "\n")
    
    def make_env():
        from gymnasium import ObservationWrapper
        
        # Wrapper to extract only image
        class ImageOnlyWrapper(ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = env.observation_space['image']
            
            def observation(self, obs):
                return obs['image']
        
        env = SO101Arm2(render_mode=None)
        env = ImageOnlyWrapper(env)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env for _ in range(4)])
    eval_env = DummyVecEnv([make_env])
    
    policy_kwargs = dict(
        features_extractor_class=VisionOnlyExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            vision_encoder='dinov2',
            vision_encoder_kwargs={
                'model_name': 'small',
                'freeze': True
            }
        ),
        net_arch=[256, 256],
    )
    
    model = PPO(
        "CnnPolicy",  # Use CnnPolicy for vision-only
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./logs/vision_only/",
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/vision_only/",
        name_prefix="ppo_vision"
    )
    
    model.learn(
        total_timesteps=500000,  # Need more steps
        callback=[checkpoint_callback]
    )
    
    model.save("./models/vision_only/final_model")
    return model


# ============================================================
# STRATEGY 3: State Only (Baseline) - No vision
# ============================================================
def train_state_only():
    """
    Baseline: Chỉ dùng proprioceptive state
    - Fastest training
    - Good for simple tasks
    - No visual understanding
    """
    print("\n" + "="*70)
    print("TRAINING: State Only (Baseline)")
    print("="*70 + "\n")
    
    from gymnasium import ObservationWrapper
    
    class StateOnlyWrapper(ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = env.observation_space['state']
        
        def observation(self, obs):
            return obs['state']
    
    def make_env():
        env = SO101Arm2(render_mode=None)
        env = StateOnlyWrapper(env)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env for _ in range(4)])
    eval_env = DummyVecEnv([make_env])
    
    model = PPO(
        "MlpPolicy",  # Simple MLP
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./logs/state_only/",
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/state_only/",
        name_prefix="ppo_state"
    )
    
    model.learn(
        total_timesteps=200000,
        callback=[checkpoint_callback]
    )
    
    model.save("./models/state_only/final_model")
    return model


# ============================================================
# STRATEGY 4: Fine-tune (Unfreeze DINOv2)
# ============================================================
def train_finetune_dinov2():
    """
    Advanced: Fine-tune DINOv2
    - Train với frozen DINOv2 trước (100k steps)
    - Unfreeze và fine-tune (100k steps nữa)
    - Learning rate nhỏ hơn cho vision encoder
    """
    print("\n" + "="*70)
    print("TRAINING: Fine-tune DINOv2")
    print("="*70 + "\n")
    
    def make_env():
        env = SO101Arm2(render_mode=None)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env for _ in range(4)])
    
    # PHASE 1: Train với frozen encoder
    print("\n--- PHASE 1: Frozen DINOv2 ---")
    policy_kwargs_frozen = dict(
        features_extractor_class=VisionStateExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            vision_encoder='dinov2',
            vision_encoder_kwargs={
                'model_name': 'small',
                'freeze': True  # FROZEN
            },
            state_hidden_dim=64,
        ),
        net_arch=[256, 256],
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs_frozen,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
    )
    
    model.learn(total_timesteps=100000)
    print("✓ Phase 1 completed")
    
    # PHASE 2: Unfreeze và fine-tune
    print("\n--- PHASE 2: Fine-tune DINOv2 ---")
    
    # Unfreeze vision encoder
    for param in model.policy.features_extractor.vision_encoder.parameters():
        param.requires_grad = True
    
    # Set lower learning rate cho vision encoder
    vision_params = list(model.policy.features_extractor.vision_encoder.parameters())
    other_params = [p for p in model.policy.parameters() if p not in vision_params]
    
    # Recreate optimizer với different learning rates
    model.policy.optimizer = torch.optim.Adam([
        {'params': vision_params, 'lr': 1e-5},      # Low LR for vision
        {'params': other_params, 'lr': 3e-4}        # Normal LR for policy
    ])
    
    print("Vision encoder unfrozen, training with differential learning rates...")
    model.learn(total_timesteps=100000)
    
    model.save("./models/finetuned_dinov2/final_model")
    print("✓ Fine-tuning completed!")
    
    return model


# ============================================================
# STRATEGY 5: CLIP Encoder (Semantic Understanding)
# ============================================================
def train_clip_multimodal():
    """
    Sử dụng CLIP thay vì DINOv2
    - Better for semantic/language grounding
    - 512 dim features
    """
    print("\n" + "="*70)
    print("TRAINING: CLIP + State")
    print("="*70 + "\n")
    
    def make_env():
        env = SO101Arm2(render_mode=None)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env for _ in range(4)])
    
    policy_kwargs = dict(
        features_extractor_class=VisionStateExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            vision_encoder='clip',
            vision_encoder_kwargs={
                'model_name': 'ViT-B/32',
                'freeze': True
            },
            state_hidden_dim=64,
        ),
        net_arch=[256, 256],
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./logs/clip_multimodal/",
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/clip_multimodal/",
        name_prefix="ppo_clip"
    )
    
    model.learn(
        total_timesteps=200000,
        callback=[checkpoint_callback]
    )
    
    model.save("./models/clip_multimodal/final_model")
    return model


# ============================================================
# TESTING TRAINED MODEL
# ============================================================
def test_model(model_path, n_episodes=10, render=True):
    """
    Test trained model
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of test episodes
        render: Render environment
    """
    print(f"\nTesting model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = SO101Arm2(render_mode='human' if render else None)
    
    # Test episodes
    episode_rewards = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if info.get('reached', 0):
                success_count += 1
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    # Statistics
    print(f"\n{'='*50}")
    print(f"Test Results ({n_episodes} episodes):")
    print(f"  Mean reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"  Success rate: {success_count/n_episodes*100:.1f}%")
    print(f"{'='*50}")
    
    env.close()


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='multimodal',
                       choices=['multimodal', 'vision_only', 'state_only', 'finetune', 'clip'],
                       help='Training strategy')
    parser.add_argument('--test', type=str, default=None,
                       help='Path to model for testing')
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        test_model(args.test, n_episodes=10, render=True)
    else:
        # Training mode
        strategies = {
            'multimodal': train_multimodal_dinov2,
            'vision_only': train_vision_only,
            'state_only': train_state_only,
            'finetune': train_finetune_dinov2,
            'clip': train_clip_multimodal
        }
        
        print(f"\nSelected strategy: {args.strategy}")
        model = strategies[args.strategy]()
        
        print("\n✓ Training finished!")
        print(f"Model saved to ./models/{args.strategy}/")
        print(f"\nTo test: python train_with_vision.py --test ./models/{args.strategy}/final_model")


# ============================================================
# QUICK START EXAMPLES
# ============================================================
"""
# 1. Train với DINOv2 + State (RECOMMENDED)
python train_with_vision.py --strategy multimodal

# 2. Train vision-only
python train_with_vision.py --strategy vision_only

# 3. Train baseline (state only)
python train_with_vision.py --strategy state_only

# 4. Fine-tune DINOv2
python train_with_vision.py --strategy finetune

# 5. Train với CLIP
python train_with_vision.py --strategy clip

# 6. Test trained model
python train_with_vision.py --test ./models/multimodal/final_model

# 7. Monitor training với TensorBoard
tensorboard --logdir ./logs/
"""