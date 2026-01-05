"""Generator architecture for GANsformer-based image inpainting.

This module implements the encoder-bottleneck-decoder generator with
GANsformer blocks in the bottleneck for global context modeling.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .bipartite import BipartiteAttention


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection.
    
    Args:
        channels: Number of input/output channels
        use_norm: Whether to use instance normalization
    """
    
    def __init__(self, channels: int, use_norm: bool = True):
        super().__init__()
        
        layers = []
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        if use_norm:
            layers.append(nn.InstanceNorm2d(channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        if use_norm:
            layers.append(nn.InstanceNorm2d(channels))
        
        self.block = nn.Sequential(*layers)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


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
        
        # Convolutional layers with normalization
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm_in = nn.InstanceNorm2d(channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm_out = nn.InstanceNorm2d(channels)
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
        
        # Initial convolution with normalization
        x = self.activation(self.norm_in(self.conv_in(x)))
        
        # Flatten spatial dimensions: (B, C, H, W) → (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Expand latents for batch: (1, M, C) → (B, M, C)
        z = self.latent.expand(B, -1, -1)
        
        # Apply bipartite attention
        x_attn, _ = self.attention(x_flat, z)
        
        # Reshape back: (B, H*W, C) → (B, C, H, W)
        x_attn = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # Output convolution with residual
        out = self.norm_out(self.conv_out(x_attn)) + residual
        
        return out


class EncoderBlock(nn.Module):
    """Encoder block with downsampling and residual connections.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        num_residual: Number of residual blocks
    """
    
    def __init__(self, in_ch: int, out_ch: int, num_residual: int = 2):
        super().__init__()
        
        # Downsampling convolution
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Residual blocks for feature refinement
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(out_ch) for _ in range(num_residual)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = self.residual_blocks(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and residual blocks.
    
    Args:
        in_ch: Input channels (includes skip connection)
        out_ch: Output channels
        num_residual: Number of residual blocks
        final: Whether this is the final block (uses Tanh activation)
    """
    
    def __init__(self, in_ch: int, out_ch: int, num_residual: int = 2, final: bool = False):
        super().__init__()
        
        self.final = final
        
        # Upsampling with transposed convolution
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        
        if final:
            self.norm = nn.Identity()
            self.activation = nn.Tanh()
            self.residual_blocks = nn.Identity()
        else:
            self.norm = nn.InstanceNorm2d(out_ch)
            self.activation = nn.ReLU(inplace=True)
            # Residual blocks for feature refinement
            self.residual_blocks = nn.Sequential(*[
                ResidualBlock(out_ch) for _ in range(num_residual)
            ])
    
    def forward(self, x: Tensor, skip: Tensor = None) -> Tensor:
        x = self.upsample(x)
        x = self.activation(self.norm(x))
        
        # Add skip connection if provided
        if skip is not None:
            x = x + skip
        
        x = self.residual_blocks(x)
        return x


class Generator(nn.Module):
    """U-Net style generator with GANsformer bottleneck and skip connections.
    
    Architecture:
    - Encoder: Strided convolutions with residual blocks for downsampling
    - Bottleneck: Stacked GANsformer blocks for global attention
    - Decoder: Transposed convolutions with skip connections for upsampling
    
    Args:
        config: Configuration object with model hyperparameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        channels = config.encoder_channels
        bottleneck_ch = config.bottleneck_channels
        
        # Initial convolution (no downsampling)
        self.initial = nn.Sequential(
            nn.Conv2d(config.img_channels, channels[0], kernel_size=7, padding=3),
            nn.InstanceNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Encoder blocks with progressive downsampling
        # 256×256 → 128×128 → 64×64 → 32×32
        self.encoder1 = EncoderBlock(channels[0], channels[0], num_residual=2)
        self.encoder2 = EncoderBlock(channels[0], channels[1], num_residual=2)
        self.encoder3 = EncoderBlock(channels[1], channels[2], num_residual=2)
        
        # Bottleneck: GANsformer blocks for global context
        self.bottleneck = nn.Sequential(*[
            GansformerBlock(
                channels=bottleneck_ch,
                num_latents=config.latent_num,
                num_heads=config.num_heads
            )
            for _ in range(config.num_gansformer_blocks)
        ])
        
        # Decoder blocks with progressive upsampling and skip connections
        # 32×32 → 64×64 → 128×128 → 256×256
        self.decoder3 = DecoderBlock(channels[2], channels[1], num_residual=2)
        self.decoder2 = DecoderBlock(channels[1], channels[0], num_residual=2)
        self.decoder1 = DecoderBlock(channels[0], channels[0], num_residual=2)
        
        # Final output convolution
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], config.img_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Generate inpainted image from masked input.
        
        Args:
            x: Masked input image (B, 3, H, W)
        
        Returns:
            Inpainted output image (B, 3, H, W)
        """
        # Initial features
        x0 = self.initial(x)
        
        # Encoder with skip connections
        e1 = self.encoder1(x0)   # 128x128
        e2 = self.encoder2(e1)   # 64x64
        e3 = self.encoder3(e2)   # 32x32
        
        # Bottleneck with GANsformer attention
        b = self.bottleneck(e3)
        
        # Decoder with skip connections
        d3 = self.decoder3(b, e2)     # 64x64 + skip from e2
        d2 = self.decoder2(d3, e1)    # 128x128 + skip from e1
        d1 = self.decoder1(d2, x0)    # 256x256 + skip from x0
        
        # Final output
        output = self.final(d1)
        
        return output


# Backward compatibility alias
Gansformer = GansformerBlock
