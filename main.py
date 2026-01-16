"""Main entry point for GANsformer face inpainting training.

Usage:
    python main.py                    # Train with default config
    python main.py --epochs 10        # Train for 10 epochs
    python main.py --batch-size 8     # Use batch size of 8
"""

import argparse
import sys

import torch

from src.config import Config
from src.dataloader import get_loader
from src.models import Generator, Discriminator
from src.training import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GANsformer for face inpainting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr-g", type=float, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, help="Discriminator learning rate")
    
    # Data paths
    parser.add_argument("--data-root", type=str, help="Root directory for dataset")
    
    # System
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--workers", type=int, help="Number of data loading workers")
    
    # Checkpointing
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", default=None, help="Enable wandb logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity (username/team)")
    parser.add_argument("--wandb-name", type=str, help="Wandb run name")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Override config with command-line arguments
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr_g is not None:
        config.lr_G = args.lr_g
    if args.lr_d is not None:
        config.lr_D = args.lr_d
    if args.data_root is not None:
        config.data_root = args.data_root
    if args.device is not None:
        config.device = args.device
    if args.workers is not None:
        config.num_workers = args.workers
    
    # Wandb configuration
    if args.no_wandb:
        config.use_wandb = False
    elif args.wandb:
        config.use_wandb = True
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        config.wandb_entity = args.wandb_entity
    if args.wandb_name is not None:
        config.wandb_run_name = args.wandb_name
    
    print("=" * 60)
    print("GANsformer Face Inpainting")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning Rate G: {config.lr_G}")
    print(f"Learning Rate D: {config.lr_D}")
    print(f"Wandb: {'Enabled' if config.use_wandb else 'Disabled'}")
    print("=" * 60)
    
    # Prepare data
    print("\nInitializing dataloader...")
    try:
        dataloader = get_loader(config)
        print(f"Dataset size: {len(dataloader.dataset)} images")
        print(f"Batches per epoch: {len(dataloader)}")
        
        # Quick sanity check
        sample_masked, sample_gt = next(iter(dataloader))
        print(f"Input shape: {sample_masked.shape}")
        print(f"Target shape: {sample_gt.shape}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure the dataset directories exist:")
        print(f"  - {config.masked_path}")
        print(f"  - {config.gt_path}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Initialize models
    print("\nBuilding models...")
    generator = Generator(config)
    discriminator = Discriminator(config)
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    print(f"Total parameters: {g_params + d_params:,}")
    
    # Initialize trainer
    trainer = Trainer(generator, discriminator, dataloader, config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train()


if __name__ == "__main__":
    main()
