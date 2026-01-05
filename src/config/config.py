"""Configuration settings for the GANsformer model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class Config:
    """Configuration for GANsformer training and model architecture."""
    
    # System
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    seed: int = 42
    
    # Data paths
    data_root: Path = Path("dataset/celeba_hq_256")
    masked_subdir: str = "extracted"
    gt_subdir: str = "ground_truth"
    
    # Image settings
    img_size: int = 256
    img_channels: int = 3
    
    # Dataloader
    batch_size: int = 16
    shuffle: bool = True
    
    # Model architecture (GANsformer specific)
    latent_num: int = 32       # M: Number of latent variables
    latent_dim: int = 512      # C: Feature dimension
    num_heads: int = 8         # Multi-head attention heads
    
    # Generator architecture
    encoder_channels: Tuple[int, ...] = (64, 128, 256)
    bottleneck_channels: int = 256
    num_gansformer_blocks: int = 4  # Increased for better global context modeling
    
    # Training hyperparameters
    lr_G: float = 0.0002         # Generator learning rate
    lr_D: float = 0.0002         # Discriminator learning rate (matched with G)
    num_epochs: int = 3
    betas: Tuple[float, float] = (0.0, 0.99)
    
    # Loss weights
    l1_weight: float = 100.0
    
    # Regularization
    r1_gamma: float = 1.0      # R1 gradient penalty weight (reduced to prevent D collapse)
    d_reg_interval: int = 16   # Lazy regularization interval
    r1_warmup_steps: int = 500 # Disable R1 for first N steps to let D establish
    grad_clip: float = 1.0     # Gradient clipping max norm (prevents explosion)
    
    # Mixed precision
    use_amp: bool = True       # Set False to disable mixed precision if NaN issues
    compile_models: bool = True  # Use torch.compile() for faster training (PyTorch 2.0+)
    
    # Logging
    log_interval: int = 50
    save_interval: int = 1000
    checkpoint_dir: Path = Path("checkpoints")
    
    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "gansformer-inpainting"
    wandb_entity: str | None = None  # Your wandb username/team
    wandb_run_name: str | None = None  # Auto-generated if None
    log_images_interval: int = 500  # Log sample images every N steps
    
    @property
    def masked_path(self) -> Path:
        """Full path to masked images directory."""
        return self.data_root / self.masked_subdir
    
    @property
    def gt_path(self) -> Path:
        """Full path to ground truth images directory."""
        return self.data_root / self.gt_subdir
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string paths to Path objects if needed
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Config initialized - Device: {self.device}")
