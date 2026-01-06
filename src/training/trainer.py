"""Training loop for GANsformer-based image inpainting.

This module implements the training procedure including:
- GAN adversarial training with mixed precision (fp16)
- R1 gradient penalty regularization
- Checkpoint saving and loading
- Training metrics logging with Weights & Biases
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Trainer for GANsformer image inpainting model.
    
    Implements:
    - Non-saturating GAN loss with R1 regularization
    - Lazy regularization for discriminator
    - L1 reconstruction loss for generator
    - Mixed precision training (fp16) for faster training and lower memory
    - Checkpoint management
    - Weights & Biases logging
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        dataloader: Training data loader
        config: Training configuration
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        dataloader: DataLoader,
        config,
    ):
        self.device = config.device
        self.config = config
        
        # Models
        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)
        
        # Compile models for faster training (PyTorch 2.0+)
        if config.compile_models and hasattr(torch, 'compile'):
            print("Compiling models with torch.compile()...")
            self.G = torch.compile(self.G)
            self.D = torch.compile(self.D)
        
        self.dataloader = dataloader
        
        # Lazy regularization parameters
        self.d_reg_interval = config.d_reg_interval
        self.r1_gamma = config.r1_gamma
        
        # Optimizers with separate learning rates for G and D
        # D learning rate adjusted for lazy regularization
        reg_ratio = self.d_reg_interval / (self.d_reg_interval + 1)
        
        self.opt_G = Adam(
            self.G.parameters(),
            lr=config.lr_G,
            betas=config.betas,
        )
        self.opt_D = Adam(
            self.D.parameters(),
            lr=config.lr_D * reg_ratio,
            betas=(config.betas[0], config.betas[1] ** reg_ratio),
        )
        
        # Mixed precision training scalers
        self.use_amp = config.device == "cuda" and config.use_amp
        self.scaler_G = GradScaler("cuda", enabled=self.use_amp)
        self.scaler_D = GradScaler("cuda", enabled=self.use_amp)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Wandb state
        self.wandb_initialized = False
        
        # Store sample batch for consistent visualization
        self._sample_batch = None
        
        if self.use_amp:
            print("Mixed precision training (fp16) enabled")
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.config.use_wandb:
            return
        
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Install with: pip install wandb")
            self.config.use_wandb = False
            return
        
        # Prepare config dict for wandb
        config_dict = {
            "batch_size": self.config.batch_size,
            "lr_G": self.config.lr_G,
            "lr_D": self.config.lr_D,
            "num_epochs": self.config.num_epochs,
            "img_size": self.config.img_size,
            "latent_dim": self.config.latent_dim,
            "latent_num": self.config.latent_num,
            "num_heads": self.config.num_heads,
            "l1_weight": self.config.l1_weight,
            "r1_gamma": self.config.r1_gamma,
            "d_reg_interval": self.config.d_reg_interval,
            "encoder_channels": self.config.encoder_channels,
            "num_gansformer_blocks": self.config.num_gansformer_blocks,
            "generator_params": sum(p.numel() for p in self.G.parameters()),
            "discriminator_params": sum(p.numel() for p in self.D.parameters()),
            "mixed_precision": self.use_amp,
        }
        
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            config=config_dict,
            resume="allow",
        )
        
        # Watch models for gradient logging
        wandb.watch(self.G, log="gradients", log_freq=100)
        wandb.watch(self.D, log="gradients", log_freq=100)
        
        self.wandb_initialized = True
        print(f"Wandb initialized: {wandb.run.url}")
    
    def _log_to_wandb(
        self, 
        g_metrics: dict, 
        d_metrics: dict,
        masked_imgs: Optional[Tensor] = None,
        fake_imgs: Optional[Tensor] = None,
        gt_imgs: Optional[Tensor] = None,
    ):
        """Log metrics and images to Weights & Biases.
        
        Args:
            g_metrics: Generator metrics dictionary
            d_metrics: Discriminator metrics dictionary
            masked_imgs: Optional masked input images for visualization
            fake_imgs: Optional generated images for visualization
            gt_imgs: Optional ground truth images for visualization
        """
        if not self.wandb_initialized:
            return
        
        # Log scalar metrics
        log_dict = {
            "train/d_loss": d_metrics["d_loss"],
            "train/d_real": d_metrics["d_real"],
            "train/d_fake": d_metrics["d_fake"],
            "train/r1_loss": d_metrics["r1_loss"],
            "train/g_loss": g_metrics["g_loss"],
            "train/g_gan": g_metrics["g_gan"],
            "train/g_l1": g_metrics["g_l1"],
            "train/epoch": self.current_epoch,
            "train/global_step": self.global_step,
        }
        
        # Log sample images at specified interval
        if (
            self.global_step % self.config.log_images_interval == 0
            and masked_imgs is not None
            and fake_imgs is not None
            and gt_imgs is not None
        ):
            log_dict.update(self._create_image_logs(masked_imgs, fake_imgs, gt_imgs))
        
        wandb.log(log_dict, step=self.global_step)
    
    def _create_image_logs(
        self,
        masked_imgs: Tensor,
        fake_imgs: Tensor,
        gt_imgs: Tensor,
        num_samples: int = 4,
    ) -> dict:
        """Create wandb image logs for visualization.
        
        Args:
            masked_imgs: Masked input images (B, C, H, W)
            fake_imgs: Generated images (B, C, H, W)
            gt_imgs: Ground truth images (B, C, H, W)
            num_samples: Number of samples to log
        
        Returns:
            Dictionary with wandb Image objects
        """
        num_samples = min(num_samples, masked_imgs.shape[0])
        
        # Denormalize images from [-1, 1] to [0, 1]
        def denorm(x):
            return (x * 0.5 + 0.5).clamp(0, 1)
        
        images = []
        for i in range(num_samples):
            # Create side-by-side comparison: masked | generated | ground truth
            masked = denorm(masked_imgs[i]).cpu().float()
            fake = denorm(fake_imgs[i]).cpu().float()
            gt = denorm(gt_imgs[i]).cpu().float()
            
            # Concatenate horizontally
            comparison = torch.cat([masked, fake, gt], dim=2)
            images.append(wandb.Image(
                comparison,
                caption=f"Sample {i}: Masked | Generated | Ground Truth"
            ))
        
        return {"samples/comparison": images}
    
    def _finish_wandb(self):
        """Finish wandb run and upload final artifacts."""
        if not self.wandb_initialized:
            return
        
        # Log final checkpoint as artifact
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description="Final trained model checkpoint",
        )
        
        final_checkpoint = self.config.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        if final_checkpoint.exists():
            artifact.add_file(str(final_checkpoint))
            wandb.log_artifact(artifact)
        
        wandb.finish()
        print("Wandb run finished")
    
    def compute_r1_penalty(self, real_images: Tensor) -> Tensor:
        """Compute R1 gradient penalty on real images.
        
        R1 regularization penalizes the gradient of the discriminator
        with respect to real images, encouraging smoother decision boundaries.
        
        Note: R1 penalty is computed in fp32 for numerical stability.
        
        Args:
            real_images: Real images tensor (B, C, H, W)
        
        Returns:
            R1 penalty scalar
        """
        real_images = real_images.detach().requires_grad_(True)
        
        # Forward pass (in fp32 for gradient stability)
        with autocast("cuda", enabled=False):
            real_logits = self.D(real_images.float())
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=real_logits.sum(),
                inputs=real_images,
                create_graph=True,
                retain_graph=True,
            )[0]
            
            # R1 penalty: ||grad||^2
            penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
        
        return penalty
    
    def train_discriminator(
        self, 
        real_images: Tensor, 
        fake_images: Tensor,
        apply_r1: bool = False,
    ) -> dict:
        """Train discriminator for one step with mixed precision.
        
        Args:
            real_images: Ground truth images
            fake_images: Generated images (detached)
            apply_r1: Whether to apply R1 regularization
        
        Returns:
            Dictionary of loss values
        """
        self.opt_D.zero_grad()
        
        # Forward pass with mixed precision
        with autocast("cuda", enabled=self.use_amp):
            # Real loss: maximize D(real) → minimize -log(sigmoid(D(real)))
            d_real = self.D(real_images)
            loss_real = F.softplus(-d_real).mean()
            
            # Fake loss: minimize D(fake) → minimize log(sigmoid(D(fake)))
            d_fake = self.D(fake_images.detach())
            loss_fake = F.softplus(d_fake).mean()
            
            # Total adversarial loss
            loss_d = loss_real + loss_fake
        
        # Backward with gradient scaling
        self.scaler_D.scale(loss_d).backward()
        
        # R1 regularization (applied every d_reg_interval steps)
        # Computed in fp32 for numerical stability
        r1_loss = torch.tensor(0.0, device=self.device)
        if apply_r1:
            r1_penalty = self.compute_r1_penalty(real_images)
            r1_loss = self.r1_gamma / 2 * self.d_reg_interval * r1_penalty
            self.scaler_D.scale(r1_loss).backward()
        
        # Unscale gradients and clip to prevent explosion
        self.scaler_D.unscale_(self.opt_D)
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.config.grad_clip)
        
        # Optimizer step
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        
        return {
            "d_loss": loss_d.item(),
            "d_real": d_real.mean().item(),
            "d_fake": d_fake.mean().item(),
            "r1_loss": r1_loss.item(),
        }
    
    def train_generator(
        self, 
        masked_images: Tensor,
        real_images: Tensor,
    ) -> tuple[Tensor, dict]:
        """Train generator for one step with mixed precision.
        
        Args:
            masked_images: Input masked images
            real_images: Ground truth images
        
        Returns:
            Tuple of (generated images, loss dictionary)
        """
        self.opt_G.zero_grad()
        
        # Forward pass with mixed precision
        with autocast("cuda", enabled=self.use_amp):
            # Generate fake images
            fake_images = self.G(masked_images)
            
            # Adversarial loss: fool discriminator
            d_fake = self.D(fake_images)
            loss_gan = F.softplus(-d_fake).mean()
            
            # L1 reconstruction loss
            loss_l1 = F.l1_loss(fake_images, real_images) * self.config.l1_weight
            
            # Total generator loss
            loss_g = loss_gan + loss_l1
        
        # Backward with gradient scaling
        self.scaler_G.scale(loss_g).backward()
        
        # Unscale gradients and clip to prevent explosion
        self.scaler_G.unscale_(self.opt_G)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.grad_clip)
        
        # Optimizer step
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        
        return fake_images, {
            "g_loss": loss_g.item(),
            "g_gan": loss_gan.item(),
            "g_l1": loss_l1.item(),
        }
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save training checkpoint including scaler states.
        
        Args:
            path: Optional custom path. Uses config.checkpoint_dir by default.
        """
        if path is None:
            path = self.config.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        
        checkpoint = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "scaler_G": self.scaler_G.state_dict(),
            "scaler_D": self.scaler_D.state_dict(),
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        # Log checkpoint to wandb
        if self.wandb_initialized:
            wandb.save(str(path))
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint including scaler states.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.G.load_state_dict(checkpoint["generator"])
        self.D.load_state_dict(checkpoint["discriminator"])
        self.opt_G.load_state_dict(checkpoint["opt_G"])
        self.opt_D.load_state_dict(checkpoint["opt_D"])
        
        # Load scaler states if available (backward compatibility)
        if "scaler_G" in checkpoint:
            self.scaler_G.load_state_dict(checkpoint["scaler_G"])
        if "scaler_D" in checkpoint:
            self.scaler_D.load_state_dict(checkpoint["scaler_D"])
        
        print(f"Checkpoint loaded: {path} (step {self.global_step})")
    
    def train(self):
        """Run the full training loop with mixed precision."""
        # Initialize wandb
        self._init_wandb()
        
        print(f"Starting training on {self.device}")
        print(f"Mixed precision: {'enabled' if self.use_amp else 'disabled'}")
        print(f"Generator params: {sum(p.numel() for p in self.G.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.D.parameters()):,}")
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                for batch_idx, (masked_imgs, gt_imgs) in enumerate(self.dataloader):
                    masked_imgs = masked_imgs.to(self.device)
                    gt_imgs = gt_imgs.to(self.device)
                    
                    # Train generator first (to get fake images)
                    fake_imgs, g_metrics = self.train_generator(masked_imgs, gt_imgs)
                    
                    # Train discriminator (only every d_train_freq steps)
                    if self.global_step % self.config.d_train_freq == 0:
                        # R1 warmup: disable R1 for first N steps to let D establish
                        past_warmup = self.global_step >= self.config.r1_warmup_steps
                        apply_r1 = past_warmup and (self.global_step % self.d_reg_interval == 0)
                        d_metrics = self.train_discriminator(gt_imgs, fake_imgs, apply_r1)
                    else:
                        # Skip D training, just get metrics for logging
                        with torch.no_grad():
                            d_real = self.D(gt_imgs)
                            d_fake = self.D(fake_imgs.detach())
                        d_metrics = {
                            "d_loss": 0.0,
                            "d_real": d_real.mean().item(),
                            "d_fake": d_fake.mean().item(),
                            "r1_loss": 0.0,
                        }
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        self._log_metrics(epoch, batch_idx, g_metrics, d_metrics)
                        self._log_to_wandb(
                            g_metrics, 
                            d_metrics,
                            masked_imgs,
                            fake_imgs,
                            gt_imgs,
                        )
                    
                    # Checkpointing
                    if self.global_step % self.config.save_interval == 0 and self.global_step > 0:
                        self.save_checkpoint()
                    
                    self.global_step += 1
            
            # Final checkpoint
            self.save_checkpoint()
            print("Training complete!")
            
        finally:
            # Always finish wandb run
            self._finish_wandb()
    
    def _log_metrics(
        self, 
        epoch: int, 
        batch_idx: int, 
        g_metrics: dict, 
        d_metrics: dict,
    ):
        """Log training metrics to console."""
        print(
            f"Epoch [{epoch}/{self.config.num_epochs}] "
            f"Step [{batch_idx}/{len(self.dataloader)}] "
            f"D: {d_metrics['d_loss']:.4f} "
            f"G_GAN: {g_metrics['g_gan']:.4f} "
            f"G_L1: {g_metrics['g_l1']:.4f}"
        )
