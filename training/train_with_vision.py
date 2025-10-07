"""
Ví dụ training với Vision Encoders - FIXED VERSION
Tạo file: training/train_with_vision.py
"""
import sys
import os
import re
import shutil
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import numpy as np

# Import your environment
from rl_mm.envs import SO101Arm2, SO101Arm3

# Import feature extractors
from .feature_extractors import (
    VisionStateExtractor,
    VisionOnlyExtractor,
    StateOnlyExtractor
)
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ============================================================
# ENVIRONMENT WRAPPERS
# ============================================================
class SeedWrapper(gym.Wrapper):
    """Wrapper to handle seeding for Gymnasium environments"""
    def __init__(self, env, seed=None):
        super().__init__(env)
        self._seed = seed
        
    def reset(self, **kwargs):
        if self._seed is not None and 'seed' not in kwargs:
            kwargs['seed'] = self._seed
        return self.env.reset(**kwargs)


# ============================================================
# STRATEGY 1: Vision + State (DINOv2) - RECOMMENDED
# ============================================================

def make_env(env_id="rl_mm/SO101-v2", seed=0, rank=0):
    def _init():
        # Apply patches in this subprocess
        apply_mujoco_patches()
        
        # Create environment from env_id
        env = gym.make(env_id, render_mode=None)
        
        # Wrap with SeedWrapper for proper seeding
        env = SeedWrapper(env, seed=seed + rank)
        
        # Monitor wrapper for logging
        env = Monitor(env, filename=f"./logs/dinov2_multimodal/monitor/env_{rank}")
        
        # Set random seeds for reproducibility
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        
        return env
    return _init


def train_multimodal_dinov2(num_envs: int = 4, total_timesteps: int = 200_000, use_subproc: bool = False):
    """
    ✅ DINOv2 (frozen) + State encoder
    - Multi-env training
    - With Monitor logging
    - With checkpoint & eval callbacks
    
    Args:
        num_envs: Number of parallel environments
        total_timesteps: Total training steps
        use_subproc: Use SubprocVecEnv (faster but may flatten Dict spaces)
                     Set to False for Dict observation spaces
    """
    print("\n" + "="*70)
    print("TRAINING: DINOv2 (frozen) + State")
    print("="*70 + "\n")

    # ===== CREATE TRAIN ENV =====
    env_fns = [make_env(rank=i, seed=42) for i in range(num_envs)]
    
    # DummyVecEnv preserves Dict observation spaces better
    # SubprocVecEnv is faster but may have issues with Dict spaces
    if use_subproc:
        print(f"Using SubprocVecEnv (parallel execution)")
        env = SubprocVecEnv(env_fns, start_method='spawn')
    else:
        print(f"Using DummyVecEnv (sequential but stable for Dict spaces)")
        env = DummyVecEnv(env_fns)
    
    print(f"✓ Created {num_envs} training environments")
    print(f"  Observation space: {env.observation_space}")

    # ===== CREATE EVAL ENV =====
    eval_env = DummyVecEnv([make_env(rank=999, seed=123)])
    print("✓ Created evaluation environment")

    # ===== PPO POLICY CONFIG =====
    policy_kwargs = dict(
        features_extractor_class=VisionStateExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            vision_encoder='dinov2',
            vision_encoder_kwargs={
                'model_name': 'small',  # 384-dim DINOv2-small
                'freeze': True          # Frozen pre-trained weights
            },
            state_hidden_dim=64,
            normalize_state=True
        ),
        net_arch=[256, 256],
    )

    # ===== PPO MODEL =====
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

    # ===== CALLBACKS =====
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

    # ===== TRAIN =====
    print("🚀 Starting training with DINOv2 + State ...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )

    # ===== SAVE =====
    model.save("./models/dinov2_multimodal/final_model")
    env.close()
    eval_env.close()
    print("✅ Training completed and model saved!")

    return model


# ============================================================
# STRATEGY 2: Vision Only (DINOv2) - Pure Vision-based
# ============================================================
def train_vision_only():
    """Vision-based RL: Chỉ dùng image"""
    print("\n" + "="*70)
    print("TRAINING: Vision Only (DINOv2)")
    print("="*70 + "\n")
    
    def make_env():
        from gymnasium import ObservationWrapper
        
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
        save_freq=10000,
        save_path="./models/vision_only/",
        name_prefix="ppo_vision"
    )
    
    model.learn(
        total_timesteps=500000,
        callback=[checkpoint_callback]
    )
    
    model.save("./models/vision_only/final_model")
    return model


# ============================================================
# STRATEGY 3: State Only (Baseline) - No vision
# ============================================================
def train_state_only():
    """Baseline: Chỉ dùng proprioceptive state"""
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
        "MlpPolicy",
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
    """Advanced: Fine-tune DINOv2"""
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
                'freeze': True
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
    
    for param in model.policy.features_extractor.vision_encoder.parameters():
        param.requires_grad = True
    
    vision_params = list(model.policy.features_extractor.vision_encoder.parameters())
    other_params = [p for p in model.policy.parameters() if p not in vision_params]
    
    model.policy.optimizer = torch.optim.Adam([
        {'params': vision_params, 'lr': 1e-5},
        {'params': other_params, 'lr': 3e-4}
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
    """Sử dụng CLIP thay vì DINOv2"""
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
    """Test trained model"""
    print(f"\nTesting model: {model_path}")
    
    model = PPO.load(model_path)
    env = SO101Arm2(render_mode='human' if render else None)
    
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
    
    print(f"\n{'='*50}")
    print(f"Test Results ({n_episodes} episodes):")
    print(f"  Mean reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"  Success rate: {success_count/n_episodes*100:.1f}%")
    print(f"{'='*50}")
    
    env.close()


# ============================================================
# XML PREPROCESSING - FIX MESH PATHS
# ============================================================
def check_stl_files():
    """Kiểm tra tất cả STL files có tồn tại không"""
    import glob
    
    print("\n" + "="*70)
    print("CHECKING STL FILES")
    print("="*70)
    
    asset_dir = "rl_mm/asset/SO101"
    stl_pattern = f"{asset_dir}/**/*.stl"
    stl_files = glob.glob(stl_pattern, recursive=True)
    
    if not stl_files:
        print(f"❌ No STL files found in {asset_dir}")
        return False
    
    print(f"✓ Found {len(stl_files)} STL files:")
    all_exist = True
    for stl in stl_files:
        abs_path = os.path.abspath(stl)
        exists = os.path.exists(abs_path)
        status = "✓" if exists else "❌"
        print(f"  {status} {stl}")
        print(f"      → {abs_path}")
        if not exists:
            all_exist = False
    
    print("="*70 + "\n")
    return all_exist


def preprocess_xml_file():
    """
    Sửa XML file in-place: Convert relative mesh paths thành absolute paths
    Backup file gốc trước khi sửa
    """
    xml_path = "rl_mm/asset/SO101/so101_new_calib.xml"
    
    # Check if XML exists
    if not os.path.exists(xml_path):
        print(f"❌ XML file not found: {xml_path}")
        return False
    
    backup_path = xml_path + ".backup"
    
    # Backup original XML nếu chưa có
    if not os.path.exists(backup_path):
        try:
            shutil.copy(xml_path, backup_path)
            print(f"✓ Backed up XML to {backup_path}")
        except Exception as e:
            print(f"⚠️ Cannot backup XML: {e}")
    
    # Load XML content
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Cannot read XML file: {e}")
        return False
    
    # Get absolute asset directory
    asset_dir = os.path.dirname(os.path.abspath(xml_path))
    print(f"📁 Asset directory: {asset_dir}")
    
    # Check what paths are in XML before fixing
    print("\n🔍 Current mesh paths in XML:")
    original_paths = re.findall(r'file="([^"]+\.stl)"', content)
    for path in original_paths[:3]:  # Show first 3
        print(f"  - {path}")
    
    # Fix mesh file paths
    def fix_mesh_path(match):
        rel_path = match.group(1)
        
        # If already absolute, skip
        if os.path.isabs(rel_path):
            if os.path.exists(rel_path):
                return match.group(0)
            else:
                print(f"  ⚠️ Absolute path not found: {rel_path}")
        
        # Convert to absolute
        abs_path = os.path.abspath(os.path.join(asset_dir, rel_path))
        
        # Check if file exists
        if os.path.exists(abs_path):
            print(f"  ✓ {os.path.basename(abs_path)}")
            return f'file="{abs_path}"'
        else:
            print(f"  ❌ NOT FOUND: {abs_path}")
            print(f"     Original: {rel_path}")
            return match.group(0)  # Keep original if not found
    
    print("\n🔧 Fixing mesh paths...")
    content = re.sub(r'file="([^"]+\.stl)"', fix_mesh_path, content)
    
    # Fix texture paths if any
    content = re.sub(r'file="([^"]+\.png)"', fix_mesh_path, content)
    
    # Save fixed XML
    try:
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ XML file preprocessed successfully!")
        
        # Verify
        print("\n✓ Verification - paths after fixing:")
        fixed_paths = re.findall(r'file="([^"]+\.stl)"', content)
        for path in fixed_paths[:3]:
            print(f"  - {path}")
        
        return True
    except Exception as e:
        print(f"❌ Cannot write XML file: {e}")
        return False


def restore_xml_backup():
    """Khôi phục XML từ backup"""
    xml_path = "rl_mm/asset/SO101/so101_new_calib.xml"
    backup_path = xml_path + ".backup"
    
    if os.path.exists(backup_path):
        shutil.copy(backup_path, xml_path)
        print("✓ Restored XML from backup")
    else:
        print("⚠️ No backup found")


# ============================================================
# MUJOCO/DM_CONTROL PATCHES
# ============================================================
def apply_mujoco_patches():
    """
    Apply comprehensive patches for MuJoCo and dm_control
    """
    import builtins
    import mujoco
    from dm_control import mjcf
    from dm_control.utils import io as dm_io
    
    XML_ABS_PATH = os.path.abspath("rl_mm/asset/SO101/so101_new_calib.xml")
    ASSET_DIR = os.path.dirname(XML_ABS_PATH)
    
    print("=" * 70)
    print("APPLYING MUJOCO PATCHES")
    print("=" * 70)
    print(f"XML path: {XML_ABS_PATH}")
    print(f"Asset dir: {ASSET_DIR}")
    
    # -------------------------
    # 1. Patch builtins.open
    # -------------------------
    _original_open = builtins.open
    
    def open_patched(file, *args, **kwargs):
        if isinstance(file, str):
            # Fix malformed paths like "/rl_mm/..."
            if file.startswith("/rl_mm/"):
                file = file[1:]  # Remove leading "/"
            
            # Convert relative paths to absolute
            if not os.path.isabs(file) and "rl_mm" in file:
                file = os.path.abspath(file)
        
        return _original_open(file, *args, **kwargs)
    
    builtins.open = open_patched
    
    # -------------------------
    # 2. Patch mujoco.MjModel.from_xml_path
    # -------------------------
    _old_from_xml = mujoco.MjModel.from_xml_path
    
    def from_xml_path_patched(path, *args, **kwargs):
        if "so101_new_calib.xml" in str(path):
            path = XML_ABS_PATH
        return _old_from_xml(path, *args, **kwargs)
    
    mujoco.MjModel.from_xml_path = from_xml_path_patched
    
    # -------------------------
    # 3. Patch dm_control's mjcf.from_path
    # -------------------------
    _original_mjcf_from_path = mjcf.from_path
    
    def mjcf_from_path_patched(path, *args, **kwargs):
        if "so101_new_calib.xml" in str(path):
            path = XML_ABS_PATH
        return _original_mjcf_from_path(path, *args, **kwargs)
    
    mjcf.from_path = mjcf_from_path_patched
    
    # -------------------------
    # 4. Patch dm_control.GetResource
    # -------------------------
    _original_getresource = dm_io.GetResource
    
    def getresource_patched(path, *args, **kwargs):
        if isinstance(path, str):
            if path.startswith("/rl_mm/"):
                path = path[1:]
            if not os.path.isabs(path) and "rl_mm" in path:
                path = os.path.abspath(path)
        return _original_getresource(path, *args, **kwargs)
    
    dm_io.GetResource = getresource_patched
    
    print("✅ Patches applied successfully!")
    print("=" * 70 + "\n")


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":
    import argparse
    
    # =========================
    # STEP 1: Check STL files
    # =========================
    print("\n" + "="*70)
    print("STEP 1: CHECKING STL FILES")
    print("="*70 + "\n")
    
    if not check_stl_files():
        print("\n❌ STL files check failed!")
        print("Please ensure all STL files exist in rl_mm/asset/SO101/assets/")
        sys.exit(1)
    
    # =========================
    # STEP 2: Preprocess XML
    # =========================
    print("\n" + "="*70)
    print("STEP 2: XML PREPROCESSING")
    print("="*70 + "\n")
    
    success = preprocess_xml_file()
    if not success:
        print("\n❌ XML preprocessing failed! Check file paths.")
        sys.exit(1)
    
    # =========================
    # STEP 3: Apply Patches
    # =========================
    print("\n" + "="*70)
    print("STEP 3: APPLYING PATCHES")
    print("="*70 + "\n")
    
    apply_mujoco_patches()
    
    # =========================
    # STEP 4: Parse Arguments
    # =========================
    parser = argparse.ArgumentParser(
        description="Train multimodal RL models with various strategies."
    )
    parser.add_argument(
        '--strategy', type=str, default='multimodal',
        choices=['multimodal', 'vision_only', 'state_only', 'finetune', 'clip'],
        help='Training strategy to use'
    )
    parser.add_argument(
        '--num_envs', type=int, default=4,
        help='Number of parallel training environments (default: 4)'
    )
    parser.add_argument(
        '--total_timesteps', type=int, default=200_000,
        help='Total number of training timesteps (default: 200000)'
    )
    parser.add_argument(
        '--test', type=str, default=None,
        help='Path to model for testing (if provided, skips training)'
    )
    parser.add_argument(
        '--restore', action='store_true',
        help='Restore XML from backup and exit'
    )
    
    args = parser.parse_args()
    
    # =========================
    # RESTORE MODE
    # =========================
    if args.restore:
        restore_xml_backup()
        sys.exit(0)
    
    # =========================
    # TEST MODE
    # =========================
    if args.test:
        test_model(args.test, n_episodes=10, render=True)
        sys.exit(0)
    
    # =========================
    # TRAINING MODE
    # =========================
    print("\n" + "="*70)
    print("STEP 4: TRAINING")
    print("="*70)
    print(f"\nSelected strategy: {args.strategy}")
    print(f"Number of envs: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps:,}\n")
    
    try:
        # Strategy mapping
        if args.strategy == 'multimodal':
            model = train_multimodal_dinov2(
                num_envs=args.num_envs,
                total_timesteps=args.total_timesteps
            )
        elif args.strategy == 'clip':
            model = train_clip_multimodal()
        elif args.strategy == 'vision_only':
            model = train_vision_only()
        elif args.strategy == 'state_only':
            model = train_state_only()
        elif args.strategy == 'finetune':
            model = train_finetune_dinov2()
        else:
            raise ValueError(f"❌ Unknown strategy: {args.strategy}")
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETED!")
        print("="*70)
        print(f"Model saved to ./models/{args.strategy}/")
        print(f"\nTo test your model:")
        print(f"python train_with_vision.py --test ./models/{args.strategy}/final_model")
        print(f"\nTo restore original XML:")
        print(f"python train_with_vision.py --restore")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Optionally restore XML after training
        # restore_xml_backup()
        pass


# ============================================================
# QUICK START EXAMPLES
# ============================================================
"""
# 1. Train với DINOv2 + State (RECOMMENDED)
python train_with_vision.py --strategy multimodal

# 2. Train với 8 parallel envs và 500k steps
python train_with_vision.py --strategy multimodal --num_envs 8 --total_timesteps 500000

# 3. Train vision-only
python train_with_vision.py --strategy vision_only

# 4. Train baseline (state only)
python train_with_vision.py --strategy state_only

# 5. Fine-tune DINOv2
python train_with_vision.py --strategy finetune

# 6. Train với CLIP
python train_with_vision.py --strategy clip

# 7. Test trained model
python train_with_vision.py --test ./models/multimodal/final_model

# 8. Restore original XML from backup
python train_with_vision.py --restore

# 9. Monitor training với TensorBoard
tensorboard --logdir ./logs/

# 10. Debug: Check if XML was fixed correctly
cat rl_mm/asset/SO101/so101_new_calib.xml | grep "file="
"""