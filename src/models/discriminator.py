"""Discriminator architecture for GANsformer-based image inpainting.

This module implements a PatchGAN-style discriminator with bipartite attention
for global context awareness.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .bipartite import BipartiteAttention


class Discriminator(nn.Module):
    """Discriminator with bipartite attention for global reasoning.
    
    Architecture:
    - Feature extractor: Strided convolutions for downsampling
    - Bipartite attention: Global context aggregation
    - Classifier: Outputs real/fake logits
    
    Args:
        config: Configuration object with model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        feature_ch = 256  # Feature channels after extraction
        
        # Feature extractor: 256×256 → 32×32
        self.features = nn.Sequential(
            self._conv_block(config.img_channels, 64),
            self._conv_block(64, 128),
            self._conv_block(128, feature_ch),
        )
        
        # Bipartite attention for global context
        self.attention = BipartiteAttention(feature_ch, num_heads=config.num_heads)
        self.latent = nn.Parameter(torch.randn(1, config.latent_dim, feature_ch) * 0.02)
        
        # Global pooling and classification
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(feature_ch, 1, kernel_size=1)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a strided convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute discriminator logits.
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            Logits (B, 1) indicating real/fake probability
        """
        B = x.shape[0]
        
        # Extract features: (B, 3, 256, 256) → (B, 256, 32, 32)
        features = self.features(x)
        _, C, H, W = features.shape
        
        # Flatten for attention: (B, 256, 32, 32) → (B, 1024, 256)
        features_flat = features.flatten(2).transpose(1, 2)
        
        # Expand latents: (1, M, C) → (B, M, C)
        z = self.latent.expand(B, -1, -1)
        
        # Apply bipartite attention
        features_attn, _ = self.attention(features_flat, z)
        
        # Reshape back: (B, 1024, 256) → (B, 256, 32, 32)
        features_attn = features_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # Global pooling and classification
        pooled = self.pool(features_attn)  # (B, 256, 1, 1)
        logits = self.classifier(pooled)    # (B, 1, 1, 1)
        
        return logits.view(B, -1)
