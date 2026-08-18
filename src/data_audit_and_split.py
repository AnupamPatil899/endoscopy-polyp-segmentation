"""
Data Audit, Quality Validation, and Stratified Train/Val Split Generator for Kvasir-SEG.

Performs:
1. Image and mask integrity validation (matching filenames, non-corrupt files).
2. Mask binarization & unique value validation (detects any grayscale interpolation artifacts).
3. Polyp foreground area calculation and size bucketing:
   - Small (< 5% of total frame area)
   - Medium (5% - 20% of total frame area)
   - Large (> 20% of total frame area)
4. Deterministic stratified 80/20 train/val split generation.
5. Exports split JSON files and comprehensive dataset metadata CSV.
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def run_audit_and_split(
    data_dir: str = "Data/Kvasir-SEG",
    output_splits_dir: str = "data/splits",
    seed: int = 42,
    train_ratio: float = 0.8,
):
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    out_dir = Path(output_splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Missing images or masks directory in {data_dir}")

    image_files = sorted([f.name for f in images_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    mask_files = sorted([f.name for f in masks_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    print(f"Found {len(image_files)} images and {len(mask_files)} masks.")

    # 1. Verify 1-to-1 match
    img_set = set(image_files)
    mask_set = set(mask_files)
    if img_set != mask_set:
        diff_img = img_set - mask_set
        diff_mask = mask_set - img_set
        raise ValueError(f"Mismatch between images and masks: {len(diff_img)} extra images, {len(diff_mask)} extra masks.")

    records = []
    print("Auditing image integrity, dimensions, and mask properties...")

    for fname in image_files:
        img_p = str(images_dir / fname)
        mask_p = str(masks_dir / fname)

        img = cv2.imread(img_p)
        if img is None:
            raise ValueError(f"Corrupted image file: {img_p}")
        h, w, c = img.shape

        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Corrupted mask file: {mask_p}")
        mh, mw = mask.shape

        if (h, w) != (mh, mw):
            raise ValueError(f"Dimension mismatch for {fname}: Image is {(h, w)}, Mask is {(mh, mw)}")

        # Check mask values
        unique_vals = np.unique(mask)
        # Binarize strictly at threshold 127
        binary_mask = (mask > 127).astype(np.uint8)
        polyp_pixels = int(np.sum(binary_mask))
        total_pixels = h * w
        area_pct = (polyp_pixels / total_pixels) * 100.0

        if area_pct == 0:
            bucket = "Empty"
        elif area_pct < 5.0:
            bucket = "Small"
        elif area_pct <= 20.0:
            bucket = "Medium"
        else:
            bucket = "Large"

        records.append(
            {
                "filename": fname,
                "image_path": img_p,
                "mask_path": mask_p,
                "width": w,
                "height": h,
                "channels": c,
                "polyp_pixels": polyp_pixels,
                "total_pixels": total_pixels,
                "area_pct": round(area_pct, 4),
                "size_bucket": bucket,
                "is_clean_binary": set(unique_vals.tolist()).issubset({0, 255}),
            }
        )

    df = pd.DataFrame(records)
    print("\n=== Dataset Audit Summary ===")
    print(f"Total verified samples: {len(df)}")
    print(f"Resolution range: Width [{df['width'].min()} - {df['width'].max()}], Height [{df['height'].min()} - {df['height'].max()}]")
    print(f"Clean binary masks (strictly {{0, 255}}): {df['is_clean_binary'].sum()} / {len(df)}")
    print(f"Polyp Area % Summary: Min={df['area_pct'].min():.2f}%, Median={df['area_pct'].median():.2f}%, Mean={df['area_pct'].mean():.2f}%, Max={df['area_pct'].max():.2f}%")
    print("\nSize Bucket Distribution:")
    print(df["size_bucket"].value_counts())

    # Save metadata CSV
    metadata_csv_path = out_dir / "kvasir_metadata.csv"
    df.to_csv(metadata_csv_path, index=False)
    print(f"\nSaved metadata CSV to: {metadata_csv_path}")

    # 2. Stratified 80/20 split
    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=df["size_bucket"],
        shuffle=True,
    )

    print("\n=== Split Distribution ===")
    print("Train Split Buckets (N =", len(train_df), "):")
    print(train_df["size_bucket"].value_counts(normalize=True).round(4) * 100)
    print("Val Split Buckets (N =", len(val_df), "):")
    print(val_df["size_bucket"].value_counts(normalize=True).round(4) * 100)

    # Save train / val JSONs
    train_data = {
        "dataset": "Kvasir-SEG",
        "split": "train",
        "num_samples": len(train_df),
        "seed": seed,
        "samples": train_df.to_dict(orient="records"),
    }
    val_data = {
        "dataset": "Kvasir-SEG",
        "split": "val",
        "num_samples": len(val_df),
        "seed": seed,
        "samples": val_df.to_dict(orient="records"),
    }

    train_json_path = out_dir / "kvasir_train.json"
    val_json_path = out_dir / "kvasir_val.json"

    with open(train_json_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_json_path, "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"Saved train split ({len(train_df)} samples) to: {train_json_path}")
    print(f"Saved val split ({len(val_df)} samples) to: {val_json_path}")


if __name__ == "__main__":
    run_audit_and_split()
