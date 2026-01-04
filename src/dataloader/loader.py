"""Data loading utilities for face inpainting dataset.

This module provides dataset and dataloader implementations for
paired masked/ground-truth image datasets.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FaceInpaintingDataset(Dataset):
    """Dataset for paired masked and ground-truth face images.
    
    Expects two directories with matching filenames:
    - masked_dir: Contains images with masked regions
    - gt_dir: Contains corresponding ground-truth images
    
    Args:
        masked_dir: Path to directory containing masked images
        gt_dir: Path to directory containing ground-truth images
        img_size: Target image size (default: 256)
        transform: Optional custom transform (overrides default)
    
    Raises:
        FileNotFoundError: If either directory doesn't exist
        ValueError: If directories contain no matching images
    """
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    def __init__(
        self,
        masked_dir: Path | str,
        gt_dir: Path | str,
        img_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.masked_dir = Path(masked_dir)
        self.gt_dir = Path(gt_dir)
        self.img_size = img_size
        
        # Validate directories
        self._validate_directories()
        
        # Get sorted list of image files
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(
                f"No matching images found in:\n"
                f"  Masked: {self.masked_dir}\n"
                f"  GT: {self.gt_dir}"
            )
        
        # Default transform: resize, normalize to [-1, 1]
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _validate_directories(self):
        """Check that both directories exist."""
        if not self.masked_dir.exists():
            raise FileNotFoundError(f"Masked images directory not found: {self.masked_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Ground-truth directory not found: {self.gt_dir}")
    
    def _get_image_files(self) -> list[str]:
        """Get sorted list of image filenames present in both directories."""
        masked_files = {
            f.name for f in self.masked_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        }
        gt_files = {
            f.name for f in self.gt_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        }
        
        # Only include files present in both directories
        common_files = sorted(masked_files & gt_files)
        
        if len(masked_files) != len(common_files):
            missing = len(masked_files) - len(common_files)
            print(f"Warning: {missing} masked images have no matching ground-truth")
        
        return common_files
    
    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Load and return a masked/ground-truth image pair.
        
        Args:
            idx: Index of the image pair
        
        Returns:
            Tuple of (masked_image, ground_truth_image) tensors
        """
        filename = self.image_files[idx]
        
        masked_path = self.masked_dir / filename
        gt_path = self.gt_dir / filename
        
        masked_img = Image.open(masked_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        
        return self.transform(masked_img), self.transform(gt_img)


def get_loader(config) -> DataLoader:
    """Create a DataLoader from configuration.
    
    Args:
        config: Configuration object with data loading parameters
    
    Returns:
        Configured DataLoader instance
    """
    dataset = FaceInpaintingDataset(
        masked_dir=config.masked_path,
        gt_dir=config.gt_path,
        img_size=config.img_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
