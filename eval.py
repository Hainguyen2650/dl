"""Evaluation script for GANsformer face inpainting model.

Usage:
    python eval.py --checkpoint checkpoints/checkpoint_50000.pt
    python eval.py --checkpoint checkpoints/checkpoint_50000.pt --output results/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from tqdm import tqdm
from scipy import linalg

from src.config import Config
from src.models import Generator


class Testset(Dataset):
    """Test dataset for FFHQ face inpainting evaluation."""
    
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
        mask_top = int(size * 0.55)
        mask_bottom = int(size * 0.95)
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


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""
    
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Use layers up to relu4_4
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
            vgg[18:27] # relu4_4
        ])
        
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        self.to(device)
        self.eval()
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: Tensor) -> Tensor:
        """Normalize from [-1, 1] to ImageNet range."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std
    
    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        """Compute perceptual loss."""
        gen = self.normalize(generated)
        tgt = self.normalize(target)
        
        loss = 0.0
        for block in self.blocks:
            gen = block(gen)
            tgt = block(tgt)
            loss += F.l1_loss(gen, tgt)
        
        return loss


class InceptionV3(nn.Module):
    """Inception V3 for FID and IS calculation."""
    
    def __init__(self, device):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # For FID: use pool3 features (2048-dim)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # For IS: use final classifier
        self.fc = inception.fc
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.to(device)
        self.eval()
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: Tensor) -> Tensor:
        """Normalize from [-1, 1] to ImageNet range and resize to 299."""
        x = (x + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return (x - self.mean) / self.std
    
    def get_features(self, x: Tensor) -> Tensor:
        """Get 2048-dim features for FID."""
        x = self.normalize(x)
        x = self.blocks(x)
        return x.view(x.size(0), -1)
    
    def get_logits(self, x: Tensor) -> Tensor:
        """Get class logits for IS."""
        features = self.get_features(x)
        return self.fc(features)


def compute_ssim(img1: Tensor, img2: Tensor, window_size: int = 11) -> float:
    """Compute SSIM between two images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Convert to [0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute Fréchet Inception Distance."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_inception_score(logits: np.ndarray, splits: int = 10) -> tuple:
    """Compute Inception Score."""
    preds = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    
    scores = []
    split_size = len(preds) // splits
    
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        py = part.mean(axis=0)
        scores.append(np.exp((part * (np.log(part + 1e-10) - np.log(py + 1e-10))).sum(axis=1).mean()))
    
    return float(np.mean(scores)), float(np.std(scores))


def denormalize(tensor: Tensor) -> Tensor:
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GANsformer face inpainting model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="dataset/ffhq/test_dataset.npy", help="Path to test dataset")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("GANsformer Face Inpainting - Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    config = Config()
    generator = Generator(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    print(f"Model loaded from step {checkpoint.get('global_step', 'unknown')}")
    
    # Load evaluation models
    print("Loading VGG for perceptual loss...")
    vgg_loss = VGGPerceptualLoss(device)
    
    print("Loading Inception V3 for FID/IS...")
    inception = InceptionV3(device)
    
    # Load dataset
    testset = Testset(args.dataset)
    dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test samples: {len(testset)}")
    print("=" * 60)
    
    # Storage for metrics
    metrics = {
        "psnr_masked": [],
        "psnr_full": [],
        "ssim_full": [],
        "vgg_perceptual": [],
        "mse_masked": [],
        "mse_full": [],
    }
    
    real_features = []
    fake_features = []
    fake_logits = []
    
    # Store samples for visualization
    sample_images = {"masked": [], "generated": [], "gt": []}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            masked_images = batch["masked_image"].to(device)
            
            # Generate
            generated = generator(masked_images)
            
            # Convert to [0, 1] for metrics
            gen_01 = (generated + 1) / 2
            gt_01 = (images + 1) / 2
            
            # Expand mask for RGB
            mask_rgb = masks.expand_as(images)
            
            for i in range(images.size(0)):
                # MSE (full)
                mse_full = F.mse_loss(gen_01[i], gt_01[i]).item()
                metrics["mse_full"].append(mse_full)
                
                # MSE (masked region)
                mask_sum = mask_rgb[i].sum()
                if mask_sum > 0:
                    mse_masked = ((gen_01[i] - gt_01[i]).pow(2) * mask_rgb[i]).sum() / mask_sum
                    metrics["mse_masked"].append(mse_masked.item())
                
                # PSNR (full)
                if mse_full > 0:
                    psnr_full = 10 * np.log10(1.0 / mse_full)
                else:
                    psnr_full = 100.0
                metrics["psnr_full"].append(psnr_full)
                
                # PSNR (masked)
                if len(metrics["mse_masked"]) > 0 and metrics["mse_masked"][-1] > 0:
                    psnr_masked = 10 * np.log10(1.0 / metrics["mse_masked"][-1])
                else:
                    psnr_masked = 100.0
                metrics["psnr_masked"].append(psnr_masked)
                
                # SSIM (full)
                ssim = compute_ssim(generated[i:i+1], images[i:i+1])
                metrics["ssim_full"].append(ssim)
            
            # VGG perceptual loss (batch)
            vgg = vgg_loss(generated, images).item()
            metrics["vgg_perceptual"].extend([vgg / images.size(0)] * images.size(0))
            
            # Inception features for FID
            real_feat = inception.get_features(images).cpu().numpy()
            fake_feat = inception.get_features(generated).cpu().numpy()
            real_features.append(real_feat)
            fake_features.append(fake_feat)
            
            # Inception logits for IS
            logits = inception.get_logits(generated).cpu().numpy()
            fake_logits.append(logits)
            
            # Save first 4 samples for visualization
            if batch_idx == 0:
                for i in range(min(4, images.size(0))):
                    sample_images["masked"].append(masked_images[i].cpu())
                    sample_images["generated"].append(generated[i].cpu())
                    sample_images["gt"].append(images[i].cpu())
    
    # Compute FID
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    fid = compute_fid(real_features, fake_features)
    
    # Compute IS
    fake_logits = np.concatenate(fake_logits, axis=0)
    is_mean, is_std = compute_inception_score(fake_logits)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    results = {}
    for key, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        results[key] = {"mean": mean, "std": std}
        print(f"{key}: {mean:.4f} ± {std:.4f}")
    
    print(f"fid: {fid:.4f}")
    print(f"inception_score: {is_mean:.4f} ± {is_std:.4f}")
    
    # Save results
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
        f.write(f"fid: {fid:.4f}\n")
        f.write(f"inception_score: {is_mean:.4f} ± {is_std:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Save 4 sample comparison images
    print("\nSaving sample images...")
    for i in range(min(4, len(sample_images["masked"]))):
        # Create comparison: masked | generated | ground truth
        comparison = torch.cat([
            denormalize(sample_images["masked"][i]),
            denormalize(sample_images["generated"][i]),
            denormalize(sample_images["gt"][i]),
        ], dim=2)
        
        save_path = output_dir / f"sample_{i+1}.png"
        save_image(comparison, save_path)
        print(f"  Saved: {save_path}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
