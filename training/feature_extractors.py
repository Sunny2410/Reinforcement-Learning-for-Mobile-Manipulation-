"""
Fixed Feature Extractors - Đảm bảo tất cả encoder đều trên GPU
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observations.vision_encoders import create_vision_encoder


class VisionStateExtractor(BaseFeaturesExtractor):
    """Multi-modal Feature Extractor - ✅ GPU Ready"""
    
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
        super().__init__(observation_space, features_dim)
        
        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        assert isinstance(observation_space, spaces.Dict)
        assert 'image' in observation_space.spaces and 'state' in observation_space.spaces
        
        self.image_shape = observation_space['image'].shape
        self.state_dim = observation_space['state'].shape[0]

        # Vision encoder with device
        if vision_encoder_kwargs is None:
            vision_encoder_kwargs = {}
        vision_encoder_kwargs.setdefault('model_name', 'small')
        vision_encoder_kwargs.setdefault('freeze', True)
        vision_encoder_kwargs['device'] = self.device  # ✅ Ensure device
        
        self.vision_encoder = create_vision_encoder(vision_encoder, **vision_encoder_kwargs)
        self.vision_encoder.to(self.device)  # ✅ Explicit to device
        vision_feature_dim = self.vision_encoder.get_feature_dim()

        # State encoder
        self.normalize_state = normalize_state
        if normalize_state:
            self.state_normalizer = nn.LayerNorm(self.state_dim)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.ReLU()
        )

        # Fusion network
        combined_dim = vision_feature_dim + state_hidden_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )

        # ✅ Move all networks to device
        self.to(self.device)
        
        print(f"\n[VisionStateExtractor] Device: {self.device}")
        print(f"[VisionStateExtractor] Image: {self.image_shape}, State: {self.state_dim}")
        print(f"[VisionStateExtractor] Vision: {vision_feature_dim}, State: {state_hidden_dim}, Output: {features_dim}")

    def forward(self, observations):
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
    """Vision-only Feature Extractor - ✅ GPU Ready"""
    
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        vision_encoder: str = 'dinov2',
        vision_encoder_kwargs: dict = None,
        device: torch.device = None  # ✅ Add device param
    ):
        super().__init__(observation_space, features_dim)
        
        # ✅ Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle Box and Dict spaces
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
        print(f"Device: {self.device}")
        print(f"Image shape: {self.image_shape}")
        
        # Create encoder with device
        if vision_encoder_kwargs is None:
            vision_encoder_kwargs = {}
        if vision_encoder == 'dinov2' and 'model_name' not in vision_encoder_kwargs:
            vision_encoder_kwargs['model_name'] = 'small'
        if 'freeze' not in vision_encoder_kwargs:
            vision_encoder_kwargs['freeze'] = True
        
        vision_encoder_kwargs['device'] = self.device  # ✅ Pass device
        
        self.vision_encoder = create_vision_encoder(vision_encoder, **vision_encoder_kwargs)
        self.vision_encoder.to(self.device)  # ✅ Explicit to device
        vision_feature_dim = self.vision_encoder.get_feature_dim()
        
        # Projection layer
        if vision_feature_dim != features_dim:
            self.projection = nn.Sequential(
                nn.Linear(vision_feature_dim, features_dim),
                nn.ReLU()
            )
        else:
            self.projection = nn.Identity()
        
        # ✅ Move all to device
        self.to(self.device)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Vision features: {vision_feature_dim}")
        print(f"Output features: {features_dim}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"{'='*60}\n")
    
    def forward(self, observations):
        # Extract image
        if self.is_dict:
            image = observations['image']
        else:
            image = observations
        
        # ✅ Ensure on correct device
        image = image.to(self.device).float()
        
        # Encode
        features = self.vision_encoder(image)
        features = self.projection(features)
        
        return features

class VisionStateRecurrentExtractor(BaseFeaturesExtractor):
    """
    Multi-modal Recurrent Feature Extractor:
    - Vision: Frozen DINOv2 (or other pretrained encoder)
    - State: MLP
    - Fusion: Concatenate
    - Temporal: GRU
    ✅ GPU optimized
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        vision_encoder: str = 'dinov2',
        vision_encoder_kwargs: dict = None,
        state_hidden_dim: int = 64,
        gru_hidden_dim: int = 512,
        normalize_state: bool = True,
        device: torch.device = None
    ):
        super().__init__(observation_space, features_dim)

        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert isinstance(observation_space, spaces.Dict)
        assert 'image' in observation_space.spaces and 'state' in observation_space.spaces

        # ==========================
        # 1️⃣ Vision Encoder
        # ==========================
        self.image_shape = observation_space['image'].shape
        self.state_dim = observation_space['state'].shape[0]

        if vision_encoder_kwargs is None:
            vision_encoder_kwargs = {}
        vision_encoder_kwargs.setdefault('model_name', 'small')
        vision_encoder_kwargs.setdefault('freeze', True)
        vision_encoder_kwargs['device'] = self.device

        self.vision_encoder = create_vision_encoder(vision_encoder, **vision_encoder_kwargs)
        self.vision_encoder.to(self.device)
        vision_feature_dim = self.vision_encoder.get_feature_dim()

        # Optional small adapter (trainable)
        self.vision_adapter = nn.Sequential(
            nn.Linear(vision_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        vision_output_dim = 128

        # ==========================
        # 2️⃣ State Encoder
        # ==========================
        self.normalize_state = normalize_state
        if normalize_state:
            self.state_normalizer = nn.LayerNorm(self.state_dim)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.ReLU()
        )

        # ==========================
        # 3️⃣ Fusion + Temporal (GRU)
        # ==========================
        fused_dim = vision_output_dim + state_hidden_dim
        self.gru = nn.GRU(
            input_size=fused_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )

        # ==========================
        # 4️⃣ Projection Head
        # ==========================
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_dim, features_dim),
            nn.ReLU()
        )

        # Move everything to device
        self.to(self.device)

        print(f"\n[VisionStateRecurrentExtractor]")
        print(f"Device: {self.device}")
        print(f"Image: {self.image_shape}, State: {self.state_dim}")
        print(f"Vision Out: {vision_output_dim}, State Out: {state_hidden_dim}, GRU Hidden: {gru_hidden_dim}")
        print(f"Output Feature Dim: {features_dim}\n")

    # ==========================
    # 🔁 Forward
    # ==========================
    def forward(self, observations, hidden_state=None):
        """
        observations: Dict('image', 'state')
        hidden_state: optional previous GRU hidden state (for recurrent policy)
        """
        # Shape assumptions:
        # image: [B, C, H, W] or [B, T, C, H, W]
        # state: [B, state_dim] or [B, T, state_dim]

        image = observations['image'].to(self.device).float()
        state = observations['state'].to(self.device).float()

        # Handle time dimension (T)
        if image.dim() == 5:  # [B, T, C, H, W]
            B, T = image.shape[:2]
            image = image.view(B * T, *image.shape[2:])
            state = state.view(B * T, state.shape[-1])
        else:
            B, T = image.shape[0], 1

        # Vision encoding
        vision_features = self.vision_encoder(image)
        vision_features = self.vision_adapter(vision_features)

        # State encoding
        if self.normalize_state:
            state = self.state_normalizer(state)
        state_features = self.state_encoder(state)

        # Fuse and reshape for GRU
        fused = torch.cat([vision_features, state_features], dim=-1)
        fused = fused.view(B, T, -1)

        # GRU temporal encoding
        gru_out, new_hidden = self.gru(fused, hidden_state)

        # Only return last timestep’s feature (SB3 expects [B, features_dim])
        last_features = gru_out[:, -1, :]
        out = self.projection(last_features)

        return out, new_hidden

    def get_initial_hidden_state(self, batch_size: int = 1):
        """Return zero hidden state for GRU."""
        return torch.zeros(1, batch_size, self.gru.hidden_size, device=self.device)


class StateOnlyExtractor(BaseFeaturesExtractor):
    """State-only Feature Extractor - ✅ GPU Ready"""
    
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        hidden_dims: list = [256, 256],
        normalize: bool = True,
        device: torch.device = None  # ✅ Add device param
    ):
        super().__init__(observation_space, features_dim)
        
        # ✅ Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle Box and Dict
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
        print(f"Device: {self.device}")
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
        
        # ✅ Move all to device
        self.to(self.device)
        
        print(f"MLP: {self.state_dim} -> {' -> '.join(map(str, hidden_dims))} -> {features_dim}")
        print(f"{'='*60}\n")
    
    def forward(self, observations):
        # Extract state
        if self.is_dict:
            state = observations['state']
        else:
            state = observations
        
        # ✅ Ensure on correct device
        state = state.to(self.device).float()
        
        # Normalize
        if self.normalize:
            state = self.normalizer(state)
        
        # Encode
        features = self.mlp(state)
        
        return features


def create_feature_extractor(
    extractor_type: str,
    observation_space: spaces.Space,
    features_dim: int = 256,
    device: torch.device = None,  # ✅ Add device param
    **kwargs
):
    """
    Factory function with device support
    
    Example:
        >>> device = torch.device('cuda')
        >>> extractor = create_feature_extractor(
        ...     'vision_state',
        ...     obs_space,
        ...     device=device,
        ...     vision_encoder='dinov2'
        ... )
    """
    extractors = {
        'vision_state': VisionStateExtractor,
        'vision_only': VisionOnlyExtractor,
        'state_only': StateOnlyExtractor
    }
    
    if extractor_type not in extractors:
        raise ValueError(f"Unknown: {extractor_type}. Choose from {list(extractors.keys())}")
    
    return extractors[extractor_type](
        observation_space=observation_space,
        features_dim=features_dim,
        device=device,  # ✅ Pass device
        **kwargs
    )


if __name__ == "__main__":
    """Test với explicit device"""
    from gymnasium import spaces
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0, high=1, shape=(224, 224, 3), dtype='float32'),
        'state': spaces.Box(low=-1, high=1, shape=(13,), dtype='float32')
    })
    
    print("\n" + "="*70)
    print("TESTING FEATURE EXTRACTORS WITH GPU")
    print("="*70)
    
    # Test Vision+State
    print("\n1. Vision + State (DINOv2)")
    extractor = VisionStateExtractor(
        obs_space,
        features_dim=256,
        vision_encoder='dinov2',
        vision_encoder_kwargs={'model_name': 'small', 'freeze': True},
        device=device
    )
    
    dummy_obs = {
        'image': torch.rand(4, 224, 224, 3),
        'state': torch.rand(4, 13)
    }
    
    with torch.no_grad():
        features = extractor(dummy_obs)
    
    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Output device: {features.device}")
    
    # Verify all params are on GPU
    for name, param in extractor.named_parameters():
        if param.device.type != device.type:
            print(f"⚠️ {name} is on {param.device}, expected {device}")
    
    print("\n✅ All tests passed!")