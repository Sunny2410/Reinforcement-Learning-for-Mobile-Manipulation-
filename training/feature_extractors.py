"""
Custom Feature Extractors for Stable-Baselines3
Tích hợp vision encoders vào SB3 training pipeline
Tạo file: training/feature_extractors.py
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observations.vision_encoders import create_vision_encoder


class VisionStateExtractor(BaseFeaturesExtractor):
    """
    Multi-modal Feature Extractor:
    - Image → Vision Encoder (DINOv2/CLIP/ResNet/CNN)
    - State → Small MLP
    - Concat → Fusion network → Final features
    Dùng với MultiInputPolicy của SB3
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        vision_encoder: str = 'dinov2',
        vision_encoder_kwargs: dict = None,
        state_hidden_dim: int = 64,
        normalize_state: bool = True,
        device: torch.device = None
    ):
        """
        Args:
            observation_space: Dict space với 'image' và 'state'
            features_dim: Kích thước feature đầu ra
            vision_encoder: 'dinov2', 'clip', 'resnet', 'cnn'
            vision_encoder_kwargs: dict các kwargs cho vision encoder
            state_hidden_dim: hidden dim cho state MLP
            normalize_state: True/False có normalize state
            device: torch.device, mặc định là 'cuda' nếu có
        """
        super().__init__(observation_space, features_dim)
        
        # Chọn device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate observation space
        assert isinstance(observation_space, spaces.Dict), "Phải dùng Dict observation space"
        assert 'image' in observation_space.spaces, "Phải có key 'image'"
        assert 'state' in observation_space.spaces, "Phải có key 'state'"
        
        # Lấy shape
        self.image_shape = observation_space['image'].shape
        self.state_dim = observation_space['state'].shape[0]

        # Vision encoder
        if vision_encoder_kwargs is None:
            vision_encoder_kwargs = {}
        vision_encoder_kwargs.setdefault('model_name', 'small')  # default DINOv2-small
        vision_encoder_kwargs.setdefault('freeze', True)         # freeze pre-trained weights
        self.vision_encoder = create_vision_encoder(vision_encoder, **vision_encoder_kwargs).to(self.device)
        vision_feature_dim = self.vision_encoder.get_feature_dim()

        # State encoder (MLP nhỏ)
        self.normalize_state = normalize_state
        if normalize_state:
            self.state_normalizer = nn.LayerNorm(self.state_dim).to(self.device)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.ReLU()
        ).to(self.device)

        # Fusion network: concat vision + state → final features
        combined_dim = vision_feature_dim + state_hidden_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        ).to(self.device)

        # Thông tin
        print(f"\n[VisionStateExtractor] Image shape: {self.image_shape}, State dim: {self.state_dim}")
        print(f"[VisionStateExtractor] Vision features: {vision_feature_dim}, State features: {state_hidden_dim}, Output features: {features_dim}")

    def forward(self, observations):
        """
        Args:
            observations: dict {'image': [B,H,W,C], 'state': [B,state_dim]}
        Returns:
            features: [B, features_dim]
        """
        image = observations['image'].to(self.device).float()
        state = observations['state'].to(self.device).float()
        if self.normalize_state:
            state = self.state_normalizer(state)
        state_features = self.state_encoder(state)
        vision_features = self.vision_encoder(image)
        combined = torch.cat([vision_features, state_features], dim=1)
        features = self.fusion_net(combined)
        return features


class VisionOnlyExtractor(BaseFeaturesExtractor):
    """
    Vision-only Feature Extractor
    Chỉ dùng image, bỏ qua state
    
    Dùng với CnnPolicy hoặc MultiInputPolicy
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        vision_encoder: str = 'dinov2',
        vision_encoder_kwargs: dict = None,
    ):
        """
        Args:
            observation_space: Box space (image) hoặc Dict space
            features_dim: Output dimension
            vision_encoder: Encoder type
            vision_encoder_kwargs: Encoder arguments
        """
        super().__init__(observation_space, features_dim)
        
        # Handle both Box and Dict spaces
        if isinstance(observation_space, spaces.Dict):
            assert 'image' in observation_space.spaces
            self.image_shape = observation_space['image'].shape
            self.is_dict = True
        else:
            self.image_shape = observation_space.shape
            self.is_dict = False
        
        print(f"\n{'='*60}")
        print(f"Initializing VisionOnlyExtractor")
        print(f"{'='*60}")
        print(f"Image shape: {self.image_shape}")
        print(f"Vision encoder: {vision_encoder}")
        
        # Create encoder
        if vision_encoder_kwargs is None:
            vision_encoder_kwargs = {}
        if vision_encoder == 'dinov2' and 'model_name' not in vision_encoder_kwargs:
            vision_encoder_kwargs['model_name'] = 'small'
        if 'freeze' not in vision_encoder_kwargs:
            vision_encoder_kwargs['freeze'] = True
        
        self.vision_encoder = create_vision_encoder(
            vision_encoder,
            **vision_encoder_kwargs
        )
        vision_feature_dim = self.vision_encoder.get_feature_dim()
        
        # Optional projection layer
        if vision_feature_dim != features_dim:
            self.projection = nn.Sequential(
                nn.Linear(vision_feature_dim, features_dim),
                nn.ReLU()
            )
        else:
            self.projection = nn.Identity()
        
        print(f"Vision features: {vision_feature_dim}")
        print(f"Output features: {features_dim}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"{'='*60}\n")
    
    def forward(self, observations):
        """
        Args:
            observations: Image [batch, H, W, C] hoặc Dict
        
        Returns:
            features: [batch, features_dim]
        """
        # Extract image
        if self.is_dict:
            image = observations['image']
        else:
            image = observations
        
        # Encode
        features = self.vision_encoder(image)
        features = self.projection(features)
        
        return features


class StateOnlyExtractor(BaseFeaturesExtractor):
    """
    State-only Feature Extractor
    Chỉ dùng proprioceptive state, bỏ image
    
    Dùng khi không cần vision hoặc để test baseline
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        hidden_dims: list = [256, 256],
        normalize: bool = True
    ):
        """
        Args:
            observation_space: Dict space với 'state' hoặc Box space
            features_dim: Output dimension
            hidden_dims: List hidden dimensions
            normalize: Use LayerNorm
        """
        super().__init__(observation_space, features_dim)
        
        # Handle both Box and Dict
        if isinstance(observation_space, spaces.Dict):
            assert 'state' in observation_space.spaces
            self.state_dim = observation_space['state'].shape[0]
            self.is_dict = True
        else:
            self.state_dim = observation_space.shape[0]
            self.is_dict = False
        
        print(f"\n{'='*60}")
        print(f"Initializing StateOnlyExtractor")
        print(f"{'='*60}")
        print(f"State dim: {self.state_dim}")
        
        # Normalizer
        self.normalize = normalize
        if normalize:
            self.normalizer = nn.LayerNorm(self.state_dim)
        
        # MLP
        layers = []
        input_dim = self.state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, features_dim))
        self.mlp = nn.Sequential(*layers)
        
        print(f"MLP architecture: {self.state_dim} -> {' -> '.join(map(str, hidden_dims))} -> {features_dim}")
        print(f"{'='*60}\n")
    
    def forward(self, observations):
        """
        Args:
            observations: State [batch, state_dim] hoặc Dict
        
        Returns:
            features: [batch, features_dim]
        """
        # Extract state
        if self.is_dict:
            state = observations['state']
        else:
            state = observations
        
        # Normalize
        if self.normalize:
            state = self.normalizer(state)
        
        # Encode
        features = self.mlp(state)
        
        return features


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_feature_extractor(
    extractor_type: str,
    observation_space: spaces.Space,
    features_dim: int = 256,
    **kwargs
):
    """
    Factory function to create feature extractor
    
    Args:
        extractor_type: 'vision_state', 'vision_only', 'state_only'
        observation_space: Gym observation space
        features_dim: Output dimension
        **kwargs: Additional arguments
    
    Returns:
        Feature extractor instance
    
    Example:
        >>> # Vision + State (Multi-modal)
        >>> extractor = create_feature_extractor(
        ...     'vision_state',
        ...     obs_space,
        ...     vision_encoder='dinov2',
        ...     vision_encoder_kwargs={'model_name': 'small', 'freeze': True}
        ... )
        
        >>> # Vision only
        >>> extractor = create_feature_extractor(
        ...     'vision_only',
        ...     obs_space,
        ...     vision_encoder='clip'
        ... )
        
        >>> # State only (baseline)
        >>> extractor = create_feature_extractor(
        ...     'state_only',
        ...     obs_space
        ... )
    """
    extractors = {
        'vision_state': VisionStateExtractor,
        'vision_only': VisionOnlyExtractor,
        'state_only': StateOnlyExtractor
    }
    
    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor: {extractor_type}. Choose from {list(extractors.keys())}")
    
    return extractors[extractor_type](
        observation_space=observation_space,
        features_dim=features_dim,
        **kwargs
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    """Test feature extractors"""
    from gymnasium import spaces
    
    # Create dummy observation space
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0, high=1, shape=(224, 224, 3), dtype='float32'),
        'state': spaces.Box(low=-1, high=1, shape=(13,), dtype='float32')
    })
    
    print("\n" + "="*70)
    print("TESTING FEATURE EXTRACTORS")
    print("="*70)
    
    # Test Vision+State
    print("\n1. Vision + State Extractor (DINOv2)")
    extractor1 = VisionStateExtractor(
        obs_space,
        features_dim=256,
        vision_encoder='dinov2',
        vision_encoder_kwargs={'model_name': 'small', 'freeze': True}
    )
    
    # Dummy forward
    dummy_obs = {
        'image': torch.rand(4, 224, 224, 3),
        'state': torch.rand(4, 13)
    }
    
    with torch.no_grad():
        features1 = extractor1(dummy_obs)
    print(f"✓ Output shape: {features1.shape}")
    
    print("\n" + "="*70)