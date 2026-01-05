"""Bipartite Attention mechanism for GANsformer.

This module implements the core attention mechanism that enables bidirectional
communication between pixel features and latent variables.
"""

import torch
import torch.nn as nn
from torch import Tensor


class BipartiteAttention(nn.Module):
    """Bidirectional attention between pixel features and latent variables.
    
    Implements the two-phase attention mechanism from GANsformer:
    1. Aggregation (Pixels → Latents): Latents gather information from pixels
    2. Broadcast (Latents → Pixels): Pixels receive context from updated latents
    
    Args:
        dim: Feature dimension for both pixels and latents
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.0)
    
    Shape:
        - Input x (pixels): (B, N, C) where N = H*W spatial positions
        - Input z (latents): (B, M, C) where M = number of latent variables
        - Output x_new: (B, N, C)
        - Output z_new: (B, M, C)
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Aggregation projections (Pixels → Latents)
        self.q_latent = nn.Linear(dim, dim, bias=False)
        self.k_pixel = nn.Linear(dim, dim, bias=False)
        self.v_pixel = nn.Linear(dim, dim, bias=False)
        
        # Broadcast projections (Latents → Pixels)
        self.q_pixel = nn.Linear(dim, dim, bias=False)
        self.k_latent = nn.Linear(dim, dim, bias=False)
        self.v_latent = nn.Linear(dim, dim, bias=False)
        
        # Layer normalization for residual connections
        self.norm_latent = nn.LayerNorm(dim)
        self.norm_pixel = nn.LayerNorm(dim)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """Reshape tensor for multi-head attention: (B, N, C) → (B, H, N, D)."""
        B, N, _ = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _reshape_from_attention(self, x: Tensor) -> Tensor:
        """Reshape tensor from multi-head attention: (B, H, N, D) → (B, N, C)."""
        B, H, N, D = x.shape
        return x.transpose(1, 2).contiguous().reshape(B, N, H * D)
    
    def _attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute scaled dot-product attention.
        
        Args:
            q: Query tensor (B, H, N, D)
            k: Key tensor (B, H, M, D)
            v: Value tensor (B, H, M, D)
        
        Returns:
            Attention output (B, H, N, D)
        """
        # Compute attention in fp32 for numerical stability
        dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # (B, H, N, D) @ (B, H, D, M) → (B, H, N, M)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # (B, H, N, M) @ (B, H, M, D) → (B, H, N, D)
        out = attn_weights @ v
        return out.to(dtype)
    
    def forward(self, x: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Apply bidirectional attention between pixels and latents.
        
        Args:
            x: Pixel features (B, N, C)
            z: Latent variables (B, M, C)
        
        Returns:
            Tuple of (updated_pixels, updated_latents)
        """
        # Phase 1: Aggregation (Pixels → Latents)
        # Latents query pixel features to gather spatial information
        q_z = self._reshape_for_attention(self.q_latent(z))
        k_x = self._reshape_for_attention(self.k_pixel(x))
        v_x = self._reshape_for_attention(self.v_pixel(x))
        
        z_aggregated = self._attention(q_z, k_x, v_x)
        z_new = self.norm_latent(z + self._reshape_from_attention(z_aggregated))
        
        # Phase 2: Broadcast (Latents → Pixels)
        # Pixels query updated latents to receive global context
        q_x = self._reshape_for_attention(self.q_pixel(x))
        k_z = self._reshape_for_attention(self.k_latent(z_new))
        v_z = self._reshape_for_attention(self.v_latent(z_new))
        
        x_broadcasted = self._attention(q_x, k_z, v_z)
        x_new = self.norm_pixel(x + self._reshape_from_attention(x_broadcasted))
        
        return x_new, z_new
