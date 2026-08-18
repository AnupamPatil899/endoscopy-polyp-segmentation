"""
Polyp Dataset and Albumentations Preprocessing Pipeline.

Key features:
1. Standardized image resizing with bilinear interpolation for RGB and nearest-neighbor for binary masks.
2. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
3. Endoscopic domain-specific augmentations:
   - Motion blur (simulating colonoscope movement)
   - Color / Brightness / Contrast jitter (simulating scope illumination changes)
   - Affine transforms (flips, 90-degree rotations, shift-scale-rotate)
   - Gaussian noise / texture perturbation
4. Explicit mask binarization (cleans JPEG ringing artifacts).
5. PyTorch DataLoader construction with pinned memory and configurable worker threads.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Standard ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: Tuple[int, int] = (352, 352)) -> A.Compose:
    """Builds training data augmentation pipeline."""
    return A.Compose(
        [
            A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.0625, 0.0625),
                rotate=(-25, 25),
                interpolation=cv2.INTER_LINEAR,
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: Tuple[int, int] = (352, 352)) -> A.Compose:
    """Builds deterministic validation / testing preprocessing pipeline."""
    return A.Compose(
        [
            A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class PolypDataset(Dataset):
    """
    PyTorch Dataset for Polyp Segmentation.
    Loads RGB endoscopy frames and corresponding binary ground-truth masks.
    """

    def __init__(
        self,
        samples: Union[List[Dict], str, Path],
        transforms: Optional[A.Compose] = None,
        base_dir: Optional[str] = None,
    ):
        """
        Args:
            samples: List of sample dictionaries, or path to split JSON file.
            transforms: Albumentations Compose pipeline.
            base_dir: Optional base directory if paths in JSON are relative.
        """
        if isinstance(samples, (str, Path)):
            with open(samples, "r") as f:
                data = json.load(f)
                self.samples = data["samples"] if "samples" in data else data
        else:
            self.samples = samples

        self.transforms = transforms
        self.base_dir = Path(base_dir) if base_dir else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        record = self.samples[idx]
        img_p = record["image_path"]
        mask_p = record["mask_path"]

        if self.base_dir:
            img_p = str(self.base_dir / img_p)
            mask_p = str(self.base_dir / mask_p)

        # Load RGB image
        image = cv2.imread(img_p)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {img_p}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load Grayscale mask and binarize
        raw_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"Failed to load mask at {mask_p}")

        # Clean JPEG ringing / compression artifacts: strictly {0.0, 1.0}
        mask = (raw_mask > 127).astype(np.float32)

        # Apply Albumentations
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Fallback if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        # Ensure mask is (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[0] != 1:
            mask = mask.permute(2, 0, 1)

        return {
            "image": image,  # (3, H, W) float32 normalized
            "mask": mask,    # (1, H, W) float32 in {0.0, 1.0}
            "filename": record.get("filename", Path(img_p).name),
            "size_bucket": record.get("size_bucket", "Unknown"),
            "area_pct": float(record.get("area_pct", 0.0)),
        }


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """
    Reverses ImageNet normalization on a (3, H, W) tensor and converts to uint8 RGB numpy array.
    """
    img = tensor.detach().cpu().numpy().copy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def get_dataloaders(
    train_json: str = "data/splits/kvasir_train.json",
    val_json: str = "data/splits/kvasir_val.json",
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (352, 352),
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to instantiate training and validation PyTorch DataLoaders.
    """
    train_dataset = PolypDataset(
        samples=train_json,
        transforms=get_train_transforms(img_size=img_size),
    )
    val_dataset = PolypDataset(
        samples=val_json,
        transforms=get_val_transforms(img_size=img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
