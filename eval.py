"""Evaluation script for GANsformer face inpainting model.

Usage:
    python eval.py --checkpoint checkpoints/checkpoint_50000.pt
    python eval.py --checkpoint checkpoints/checkpoint_50000.pt --output results/
    python eval.py --checkpoint checkpoints/checkpoint_50000.pt --num-samples 100
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.config import Config
from src.models import Generator


class Testset(Dataset):
    """Test dataset for FFHQ face inpainting evaluation.
    
    Args:
        root: Path to the .npy file containing test images
    """
    
    def __init__(self, root: str):
        self.root = root
        self.images = np.load(root)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict:
        image = Image.fromarray(self.images[idx], 'RGB')
        image = self.transform(image)
        
        # Create mask for mouth region
        size = 256
        mask = np.ones((size, size), dtype=np.float32)
        mask_top = int(size * 0.55)      # Start below nose bridge
        mask_bottom = int(size * 0.95)   # End at chin
        mask_left = int(size * 0.20)
        mask_right = int(size * 0.80)
        mask[mask_top:mask_bottom, mask_left:mask_right] = 0.0
        
        mask = torch.from_numpy(mask[None, ...])
        masked_image = image * mask
        mask_inverted = 1.0 - mask
        
        return {
            "image": image,
            "mask": mask_inverted,
            "masked_image": masked_image
        }


def denormalize(tensor: Tensor) -> Tensor:
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def compute_metrics(generated: Tensor, ground_truth: Tensor, mask: Tensor) -> dict:
    """Compute evaluation metrics.
    
    Args:
        generated: Generated images (B, C, H, W)
        ground_truth: Ground truth images (B, C, H, W)
        mask: Mask of inpainted region (B, 1, H, W)
    
    Returns:
        Dictionary of metrics
    """
    # L1 loss (full image)
    l1_full = torch.nn.functional.l1_loss(generated, ground_truth).item()
    
    # L1 loss (masked region only)
    mask_expanded = mask.expand_as(generated)
    masked_gen = generated * mask_expanded
    masked_gt = ground_truth * mask_expanded
    mask_sum = mask_expanded.sum()
    if mask_sum > 0:
        l1_masked = (torch.abs(masked_gen - masked_gt).sum() / mask_sum).item()
    else:
        l1_masked = 0.0
    
    # PSNR (Peak Signal-to-Noise Ratio)
    mse = torch.nn.functional.mse_loss(generated, ground_truth).item()
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')
    
    return {
        "l1_full": l1_full,
        "l1_masked": l1_masked,
        "psnr": psnr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GANsformer face inpainting model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset/ffhq/test_dataset.npy",
        help="Path to test dataset .npy file"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save generated images"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("GANsformer Face Inpainting - Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and model
    config = Config()
    generator = Generator(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    
    print(f"Model loaded from step {checkpoint.get('global_step', 'unknown')}")
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Load dataset
    testset = Testset(args.dataset)
    if args.num_samples:
        # Limit number of samples
        testset.images = testset.images[:args.num_samples]
    
    dataloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Test samples: {len(testset)}")
    print("=" * 60)
    
    # Evaluation loop
    all_metrics = {
        "l1_full": [],
        "l1_masked": [],
        "psnr": [],
    }
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            masked_images = batch["masked_image"].to(device)
            
            # Generate inpainted images
            generated = generator(masked_images)
            
            # Compute metrics
            metrics = compute_metrics(generated, images, masks)
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # Save images if requested
            if args.save_images:
                for i in range(generated.size(0)):
                    # Create comparison image: masked | generated | ground truth
                    comparison = torch.cat([
                        denormalize(masked_images[i]),
                        denormalize(generated[i]),
                        denormalize(images[i]),
                    ], dim=2)
                    
                    save_path = output_dir / f"sample_{sample_idx:04d}.png"
                    save_image(comparison, save_path)
                    sample_idx += 1
    
    # Compute average metrics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    results = {}
    for key, values in all_metrics.items():
        avg = np.mean(values)
        std = np.std(values)
        results[key] = {"mean": avg, "std": std}
        print(f"{key}: {avg:.4f} ± {std:.4f}")
    
    # Save results to file
    results_file = output_dir / "metrics.txt"
    with open(results_file, "w") as f:
        f.write("GANsformer Face Inpainting - Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Num samples: {len(testset)}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value['mean']:.4f} ± {value['std']:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    if args.save_images:
        print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
