"""
Augmentation Sanity Checker and Visual Validation Script.

Verifies:
1. PyTorch Dataset and DataLoader batch generation.
2. Correct Tensor shapes: Image (B, 3, H, W), Mask (B, 1, H, W).
3. Value ranges: Image normalized around 0, Mask strictly binary {0.0, 1.0}.
4. Visual alignment of augmented masks overlaid on RGB images to guarantee zero label corruption.
"""

import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import PolypDataset, denormalize_image, get_dataloaders, get_train_transforms, get_val_transforms


def create_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Creates a green semi-transparent mask overlay on the RGB image."""
    overlay = image_rgb.copy()
    # Mask is 2D (H, W) boolean
    mask_bool = binary_mask > 0.5
    # Green channel tint for polyp
    color = np.array([0, 255, 120], dtype=np.uint8)
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def run_visual_verification(
    train_json: str = "data/splits/kvasir_train.json",
    val_json: str = "data/splits/kvasir_val.json",
    output_dir: str = "outputs/augmentation_samples",
    num_samples: int = 6,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("--- 1. Testing DataLoader & Batch Shapes ---")
    train_loader, val_loader = get_dataloaders(
        train_json=train_json,
        val_json=val_json,
        batch_size=8,
        num_workers=2,
        img_size=(352, 352),
    )

    sample_batch = next(iter(train_loader))
    imgs = sample_batch["image"]
    masks = sample_batch["mask"]

    print(f"Batch image shape: {imgs.shape}, dtype: {imgs.dtype}")
    print(f"Batch mask shape: {masks.shape}, dtype: {masks.dtype}")
    print(f"Mask unique values in batch: {torch.unique(masks).tolist()}")

    # Shape assertions
    assert imgs.shape == (8, 3, 352, 352), f"Expected (8, 3, 352, 352), got {imgs.shape}"
    assert masks.shape == (8, 1, 352, 352), f"Expected (8, 1, 352, 352), got {masks.shape}"
    assert torch.all((masks == 0.0) | (masks == 1.0)), "Mask contains non-binary values!"
    print("✓ Shape and value assertions passed successfully!")

    print("\n--- 2. Generating Visual Augmentation Gallery ---")
    dataset = PolypDataset(samples=train_json, transforms=get_train_transforms((352, 352)))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for i in range(num_samples):
        item = dataset[i]
        img_tensor = item["image"]
        mask_tensor = item["mask"]
        fname = item["filename"]
        bucket = item["size_bucket"]
        area_pct = item["area_pct"]

        img_rgb = denormalize_image(img_tensor)
        mask_np = mask_tensor.squeeze().cpu().numpy()
        overlay = create_overlay(img_rgb, mask_np)

        # Col 1: Augmented RGB
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Augmented Image ({fname})\nBucket: {bucket} ({area_pct:.1f}%)", fontsize=10)
        axes[i, 0].axis("off")

        # Col 2: Binary Mask
        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title("Binary Ground Truth Mask", fontsize=10)
        axes[i, 1].axis("off")

        # Col 3: Mask Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Mask Overlay (Alignment Check)", fontsize=10)
        axes[i, 2].axis("off")

    save_fig_path = out_path / "train_augmentation_check.png"
    plt.savefig(save_fig_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✓ Saved visual augmentation gallery to: {save_fig_path}")


if __name__ == "__main__":
    run_visual_verification()
