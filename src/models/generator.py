"""Generator architecture for GANsformer-based image inpainting.

This module implements the encoder-bottleneck-decoder generator with
GANsformer blocks in the bottleneck for global context modeling.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .bipartite import BipartiteAttention


class GansformerBlock(nn.Module):
    """GANsformer block combining convolutions with bipartite attention.
    
    Each block applies:
    1. Convolutional feature extraction
    2. Bipartite attention for global context
    3. Residual connection
    
    Args:
        channels: Number of input/output channels
        num_latents: Number of latent variables for attention
        num_heads: Number of attention heads (default: 8)
    """
    
    def __init__(self, channels: int, num_latents: int, num_heads: int = 8):
        super().__init__()
        
        self.channels = channels
        self.num_latents = num_latents
        
        # Convolutional layers
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Bipartite attention
        self.attention = BipartiteAttention(channels, num_heads=num_heads)
        
        # Learnable latent variables (shared across batch)
        self.latent = nn.Parameter(torch.randn(1, num_latents, channels) * 0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply GANsformer block.
        
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Output features (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        
        # Initial convolution
        x = self.activation(self.conv_in(x))
        
        # Flatten spatial dimensions: (B, C, H, W) → (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Expand latents for batch: (1, M, C) → (B, M, C)
        z = self.latent.expand(B, -1, -1)
        
        # Apply bipartite attention
        x_attn, _ = self.attention(x_flat, z)
        
        # Reshape back: (B, H*W, C) → (B, C, H, W)
        x_attn = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # Output convolution with residual
        out = self.conv_out(x_attn) + residual
        
        return out


class Generator(nn.Module):
    """U-Net style generator with GANsformer bottleneck.
    
    Architecture:
    - Encoder: Strided convolutions for downsampling
    - Bottleneck: Stacked GANsformer blocks for global attention
    - Decoder: Transposed convolutions for upsampling
    
    Args:
        config: Configuration object with model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        channels = config.encoder_channels
        bottleneck_ch = config.bottleneck_channels
        
        # Encoder: Progressive downsampling
        # 256×256 → 128×128 → 64×64 → 32×32
        self.encoder = nn.Sequential(
            self._conv_block(config.img_channels, channels[0], downsample=True),
            self._conv_block(channels[0], channels[1], downsample=True),
            self._conv_block(channels[1], channels[2], downsample=True, activation=False),
        )
        
        # Bottleneck: GANsformer blocks for global context
        self.bottleneck = nn.Sequential(*[
            GansformerBlock(
                channels=bottleneck_ch,
                num_latents=config.latent_dim,
                num_heads=config.num_heads
            )
            for _ in range(config.num_gansformer_blocks)
        ])
        
        # Decoder: Progressive upsampling
        # 32×32 → 64×64 → 128×128 → 256×256
        self.decoder = nn.Sequential(
            self._deconv_block(channels[2], channels[1]),
            self._deconv_block(channels[1], channels[0]),
            self._deconv_block(channels[0], config.img_channels, final=True),
        )
    
    def _conv_block(
        self, 
        in_ch: int, 
        out_ch: int, 
        downsample: bool = False,
        activation: bool = True
    ) -> nn.Module:
        """Create a convolutional block with optional downsampling."""
        layers = [
            nn.Conv2d(
                in_ch, out_ch, 
                kernel_size=4 if downsample else 3,
                stride=2 if downsample else 1,
                padding=1
            )
        ]
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _deconv_block(
        self, 
        in_ch: int, 
        out_ch: int, 
        final: bool = False
    ) -> nn.Module:
        """Create a transposed convolutional block for upsampling."""
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        ]
        if final:
            layers.append(nn.Tanh())  # Output in [-1, 1]
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """Generate inpainted image from masked input.
        
        Args:
            x: Masked input image (B, 3, H, W)
        
        Returns:
            Inpainted output image (B, 3, H, W)
        """
        features = self.encoder(x)
        features = self.bottleneck(features)
        output = self.decoder(features)
        return output


# Backward compatibility alias
Gansformer = GansformerBlock
