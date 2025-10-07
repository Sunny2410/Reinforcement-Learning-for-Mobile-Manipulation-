"""
Vision Encoders - Pre-trained models for visual feature extraction
Tạo file: observations/vision_encoders.py
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal


class DINOv2Encoder(nn.Module):
    """
    DINOv2 Vision Encoder - Facebook's self-supervised vision model
    
    Variants:
        - small: dinov2_vits14 (21M params, 384 dim)
        - base: dinov2_vitb14 (86M params, 768 dim)
        - large: dinov2_vitl14 (300M params, 1024 dim)
        - giant: dinov2_vitg14 (1.1B params, 1536 dim)
    """

    def __init__(
        self,
        model_name: Literal['small', 'base', 'large', 'giant'] = 'small',
        freeze: bool = True,
        use_cls_token: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_name: Kích thước model ('small', 'base', 'large', 'giant')
            freeze: Có freeze weights không (recommend: True)
            use_cls_token: Dùng [CLS] token hay average pooling
            device: 'cuda' hoặc 'cpu'
        """
        super().__init__()

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.freeze = freeze
        self.use_cls_token = use_cls_token

        # Load pre-trained DINOv2
        model_mapping = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }
        try:
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2',
                model_mapping[model_name]
            )
            print(f"✓ Loaded DINOv2-{model_name} successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2: {e}")

        # Feature dimension
        self.feature_dim = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }[model_name]

        # Freeze
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print(f"✓ DINOv2 weights frozen")

        # Input normalization (ImageNet stats)
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        )

        # Move encoder to device
        self.encoder.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor [batch, H, W, C] hoặc [batch, C, H, W], values trong [0,1]
        Returns:
            features: [batch, feature_dim]
        """
        x = x.to(self.device)

        # Convert NHWC -> NCHW
        if x.dim() == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)

        # Normalize
        x = (x - self.mean) / self.std

        # Forward
        if self.freeze:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        return features

    def get_feature_dim(self) -> int:
        return self.feature_dim


class CLIPEncoder(nn.Module):
    """
    CLIP Vision Encoder - OpenAI's vision-language model
    Good for semantic understanding
    """
    
    def __init__(
        self,
        model_name: Literal['RN50', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'] = 'ViT-B/32',
        freeze: bool = True
    ):
        """
        Args:
            model_name: CLIP variant
            freeze: Freeze weights
        """
        super().__init__()
        
        try:
            import clip
            self.clip_module = clip
        except ImportError:
            raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Load CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder, self.preprocess = clip.load(model_name, device=device)
        print(f"✓ Loaded CLIP-{model_name} successfully")
        
        # Feature dimensions
        self.feature_dim = {
            'RN50': 1024,
            'ViT-B/32': 512,
            'ViT-B/16': 512,
            'ViT-L/14': 768
        }[model_name]
        
        # Freeze
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print(f"✓ CLIP weights frozen")
    
    def forward(self, x):
        """
        Args:
            x: Image [batch, H, W, C] or [batch, C, H, W], values in [0, 1]
        
        Returns:
            features: [batch, feature_dim]
        """
        # Convert to NCHW if needed
        if x.dim() == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        
        # CLIP expects [-1, 1] normalized
        # Preprocess handles this internally
        
        if self.freeze:
            with torch.no_grad():
                features = self.encoder.encode_image(x)
        else:
            features = self.encoder.encode_image(x)
        
        return features.float()
    
    def get_feature_dim(self):
        return self.feature_dim


class ResNetEncoder(nn.Module):
    """
    ResNet Encoder pre-trained on ImageNet
    Good baseline, faster than ViT-based models
    """
    
    def __init__(
        self,
        model_name: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101'] = 'resnet50',
        freeze: bool = True,
        use_pretrained: bool = True
    ):
        """
        Args:
            model_name: ResNet variant
            freeze: Freeze weights
            use_pretrained: Load ImageNet weights
        """
        super().__init__()
        
        import torchvision.models as models
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Load ResNet
        resnet = getattr(models, model_name)(pretrained=use_pretrained)
        
        # Remove classification head
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimensions
        self.feature_dim = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048
        }[model_name]
        
        print(f"✓ Loaded {model_name} (pretrained={use_pretrained})")
        
        # Freeze
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print(f"✓ ResNet weights frozen")
        
        # ImageNet normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Image [batch, H, W, C] or [batch, C, H, W], values in [0, 1]
        
        Returns:
            features: [batch, feature_dim]
        """
        # Convert to NCHW
        if x.dim() == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        
        # Normalize
        x = (x - self.mean) / self.std
        
        # Extract
        if self.freeze:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        return features
    
    def get_feature_dim(self):
        return self.feature_dim


class SimpleCNN(nn.Module):
    """
    Simple CNN baseline - Train from scratch
    Use when you have enough data and compute
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 256
    ):
        """
        Args:
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
            feature_dim: Output feature dimension
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        self.conv_net = nn.Sequential(
            # Layer 1: [C, 84, 84] → [32, 20, 20]
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Layer 2: [32, 20, 20] → [64, 9, 9]
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Layer 3: [64, 9, 9] → [64, 7, 7]
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            nn.Flatten(),
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out_size = self.conv_net(dummy).shape[1]
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
            nn.ReLU()
        )
        
        print(f"✓ SimpleCNN initialized (trainable)")
    
    def forward(self, x):
        """
        Args:
            x: Image [batch, H, W, C] or [batch, C, H, W]
        
        Returns:
            features: [batch, feature_dim]
        """
        # Convert to NCHW
        if x.dim() == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        
        features = self.conv_net(x)
        features = self.fc(features)
        
        return features
    
    def get_feature_dim(self):
        return self.feature_dim


def create_vision_encoder(
    encoder_type: str = 'dinov2',
    **kwargs
) -> nn.Module:
    """
    Factory function to create vision encoder
    
    Args:
        encoder_type: 'dinov2', 'clip', 'resnet', 'cnn'
        **kwargs: Arguments for specific encoder
    
    Returns:
        Vision encoder instance
    
    Example:
        >>> encoder = create_vision_encoder('dinov2', model_name='small', freeze=True)
        >>> encoder = create_vision_encoder('clip', model_name='ViT-B/32')
        >>> encoder = create_vision_encoder('cnn', feature_dim=256)
    """
    encoders = {
        'dinov2': DINOv2Encoder,
        'clip': CLIPEncoder,
        'resnet': ResNetEncoder,
        'cnn': SimpleCNN
    }
    
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder: {encoder_type}. Choose from {list(encoders.keys())}")
    
    return encoders[encoder_type](**kwargs)


# ============================================================
# TESTING UTILITIES
# ============================================================
def test_encoder(encoder_type='dinov2'):
    """Test encoder với dummy input"""
    print(f"\n{'='*50}")
    print(f"Testing {encoder_type.upper()} Encoder")
    print(f"{'='*50}")
    
    # Create encoder
    if encoder_type == 'dinov2':
        encoder = create_vision_encoder('dinov2', model_name='small', freeze=True)
        input_size = (224, 224)
    elif encoder_type == 'clip':
        encoder = create_vision_encoder('clip', model_name='ViT-B/32', freeze=True)
        input_size = (224, 224)
    elif encoder_type == 'resnet':
        encoder = create_vision_encoder('resnet', model_name='resnet50', freeze=True)
        input_size = (224, 224)
    else:  # cnn
        encoder = create_vision_encoder('cnn', input_channels=3, feature_dim=256)
        input_size = (84, 84)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.rand(batch_size, *input_size, 3)  # NHWC format
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        features = encoder(dummy_input)
    
    print(f"Output shape: {features.shape}")
    print(f"Feature dim: {encoder.get_feature_dim()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen: {total_params - trainable_params:,}")
    
    print(f"\n✓ {encoder_type.upper()} test passed!")


if __name__ == "__main__":
    # Test all encoders
    test_encoder('dinov2')
    # test_encoder('clip')      # Uncomment if CLIP installed
    # test_encoder('resnet')
    # test_encoder('cnn')