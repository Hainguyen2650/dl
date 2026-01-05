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


class Discriminator(nn.Module):
    """Discriminator with bipartite attention for global reasoning.
    
    Architecture:
    - Feature extractor: Strided convolutions with residual blocks for downsampling
    - Bipartite attention: Global context aggregation
    - Classifier head for real/fake classification
    
    Args:
        config: Configuration object with model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        feature_ch = 256  # Balanced feature channels
        
        # Initial convolution
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(config.img_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Feature extractor with progressive downsampling
        # 128×128 → 64×64 → 32×32
        self.block1 = ResidualDiscriminatorBlock(64, 128, downsample=True)
        self.block2 = ResidualDiscriminatorBlock(128, 192, downsample=True)
        self.block3 = ResidualDiscriminatorBlock(192, feature_ch, downsample=True)
        
        # Bipartite attention for global context
        self.attention = BipartiteAttention(feature_ch, num_heads=config.num_heads)
        self.latent = nn.Parameter(torch.randn(1, config.latent_num, feature_ch) * 0.02)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(feature_ch, 1)),
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
        x = self.block2(x)  # (B, 192, 32, 32)
        features = self.block3(x)  # (B, 256, 16, 16)
        _, C, H, W = features.shape
        
        # Flatten for attention: (B, 256, 16, 16) → (B, 256, 256)
        features_flat = features.flatten(2).transpose(1, 2)
        
        # Expand latents: (1, M, C) → (B, M, C)
        z = self.latent.expand(B, -1, -1)
        
        # Apply bipartite attention
        features_attn, _ = self.attention(features_flat, z)
        
        # Reshape back: (B, 256, 256) → (B, 256, 16, 16)
        features_attn = features_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # Classification
        logits = self.classifier(features_attn)
        
        return logits
