"""Discriminator architecture for GANsformer-based image inpainting.

This module implements a PatchGAN-style discriminator with bipartite attention
for global context awareness and spectral normalization for training stability.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import spectral_norm

from .bipartite import BipartiteAttention


class ResidualDiscriminatorBlock(nn.Module):
    """Residual block for discriminator with spectral normalization.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        downsample: Whether to downsample spatially
    """
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        
        self.downsample = downsample
        
        # Main path
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Skip connection
        if in_ch != out_ch or downsample:
            self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1))
        else:
            self.skip = nn.Identity()
        
        # Downsampling
        if downsample:
            self.pool = nn.AvgPool2d(2)
        else:
            self.pool = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        # Main path
        h = self.activation(self.conv1(x))
        h = self.conv2(h)
        h = self.pool(h)
        
        # Skip connection
        skip = self.pool(self.skip(x))
        
        return self.activation(h + skip)


class DiscriminatorBlock(nn.Module):
    """Discriminator block with convolution, normalization and residual connections.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        num_residual: Number of residual blocks after downsampling
    """
    
    def __init__(self, in_ch: int, out_ch: int, num_residual: int = 1):
        super().__init__()
        
        # Downsampling with residual
        self.downsample = ResidualDiscriminatorBlock(in_ch, out_ch, downsample=True)
        
        # Additional residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualDiscriminatorBlock(out_ch, out_ch, downsample=False)
            for _ in range(num_residual)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = self.residual_blocks(x)
        return x


class Discriminator(nn.Module):
    """Discriminator with bipartite attention for global reasoning.
    
    Architecture:
    - Feature extractor: Strided convolutions with residual blocks for downsampling
    - Bipartite attention: Global context aggregation
    - Multi-layer classifier: MLP head for real/fake classification
    
    Args:
        config: Configuration object with model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        feature_ch = 512  # Increased feature channels for better capacity
        
        # Initial convolution
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(config.img_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Feature extractor with progressive downsampling
        # 128×128 → 64×64 → 32×32 → 16×16
        self.block1 = DiscriminatorBlock(64, 128, num_residual=1)
        self.block2 = DiscriminatorBlock(128, 256, num_residual=1)
        self.block3 = DiscriminatorBlock(256, feature_ch, num_residual=1)
        
        # Additional residual blocks for deeper feature extraction
        self.deep_features = nn.Sequential(
            ResidualDiscriminatorBlock(feature_ch, feature_ch),
            ResidualDiscriminatorBlock(feature_ch, feature_ch),
        )
        
        # Bipartite attention for global context
        self.attention = BipartiteAttention(feature_ch, num_heads=config.num_heads)
        self.latent = nn.Parameter(torch.randn(1, config.latent_num, feature_ch) * 0.02)
        
        # Post-attention processing
        self.post_attention = nn.Sequential(
            ResidualDiscriminatorBlock(feature_ch, feature_ch),
        )
        
        # Multi-layer classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(feature_ch, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(128, 1)),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute discriminator logits.
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            Logits (B, 1) indicating real/fake probability
        """
        B = x.shape[0]
        
        # Initial convolution: (B, 3, 256, 256) → (B, 64, 128, 128)
        x = self.initial(x)
        
        # Progressive feature extraction
        x = self.block1(x)  # (B, 128, 64, 64)
        x = self.block2(x)  # (B, 256, 32, 32)
        x = self.block3(x)  # (B, 512, 16, 16)
        
        # Deeper feature extraction
        features = self.deep_features(x)
        _, C, H, W = features.shape
        
        # Flatten for attention: (B, 512, 16, 16) → (B, 256, 512)
        features_flat = features.flatten(2).transpose(1, 2)
        
        # Expand latents: (1, M, C) → (B, M, C)
        z = self.latent.expand(B, -1, -1)
        
        # Apply bipartite attention
        features_attn, _ = self.attention(features_flat, z)
        
        # Reshape back: (B, 256, 512) → (B, 512, 16, 16)
        features_attn = features_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # Post-attention processing
        features_attn = self.post_attention(features_attn)
        
        # Classification
        logits = self.classifier(features_attn)
        
        return logits
