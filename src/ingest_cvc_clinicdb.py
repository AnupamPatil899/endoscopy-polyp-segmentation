"""
Ingestion, Audit, and Test Split Generator for CVC-ClinicDB (Held-out OOD Test Set).

Performs:
1. Ingests images and masks from source2 into Data/CVC-ClinicDB/ (images and masks).
2. Verifies 100% 1-to-1 matching and image/mask dimension alignment across all 612 frames.
3. Computes polyp foreground area percentage and categorizes into Small (<5%), Medium (5-20%), and Large (>20%).
4. Generates deterministic evaluation split JSON: data/splits/cvc_clinicdb_test.json.
5. Saves comprehensive dataset metadata: data/splits/cvc_clinicdb_metadata.csv.
"""

import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd


def ingest_cvc_clinicdb(
    source_dir: str = "/home/anupa/CVC_DB/source2",
    target_dir: str = "Data/CVC-ClinicDB",
    output_splits_dir: str = "data/splits",
):
    src_path = Path(source_dir)
    target_path = Path(target_dir)
    splits_path = Path(output_splits_dir)

    src_images = src_path / "PNG" / "Original"
    src_masks = src_path / "PNG" / "Ground Truth"

    if not src_images.exists() or not src_masks.exists():
        raise FileNotFoundError(f"Source PNG folders not found in {source_dir}")

    target_images = target_path / "images"
    target_masks = target_path / "masks"
    target_images.mkdir(parents=True, exist_ok=True)
    target_masks.mkdir(parents=True, exist_ok=True)
    splits_path.mkdir(parents=True, exist_ok=True)

    # List all PNG images
    img_files = sorted([f for f in src_images.glob("*.png")])
    mask_files = sorted([f for f in src_masks.glob("*.png")])

    print(f"Found {len(img_files)} images and {len(mask_files)} masks in {source_dir}/PNG")
    assert len(img_files) == 612, f"Expected 612 images in CVC-ClinicDB, found {len(img_files)}"
    assert len(mask_files) == 612, f"Expected 612 masks in CVC-ClinicDB, found {len(mask_files)}"

    records = []
    print("Ingesting files, auditing dimensions, and calculating polyp areas...")

    for img_p in img_files:
        fname = img_p.name
        mask_p = src_masks / fname
        if not mask_p.exists():
            raise FileNotFoundError(f"Missing matching mask for {fname}")

        # Destination paths
        dest_img_p = target_images / fname
        dest_mask_p = target_masks / fname

        # Copy files into workspace if not already present
        if not dest_img_p.exists() or dest_img_p.stat().st_size != img_p.stat().st_size:
            shutil.copy2(img_p, dest_img_p)
        if not dest_mask_p.exists() or dest_mask_p.stat().st_size != mask_p.stat().st_size:
            shutil.copy2(mask_p, dest_mask_p)

        # Audit
        img = cv2.imread(str(dest_img_p))
        h, w, c = img.shape
        mask = cv2.imread(str(dest_mask_p), cv2.IMREAD_GRAYSCALE)
        mh, mw = mask.shape
        assert (h, w) == (mh, mw), f"Dimension mismatch for {fname}: Image {(h, w)} != Mask {(mh, mw)}"

        # Strict binarization & area computation
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
                "image_path": str(dest_img_p),
                "mask_path": str(dest_mask_p),
                "width": w,
                "height": h,
                "channels": c,
                "polyp_pixels": polyp_pixels,
                "total_pixels": total_pixels,
                "area_pct": round(area_pct, 4),
                "size_bucket": bucket,
            }
        )

    df = pd.DataFrame(records)
    print("\n=== CVC-ClinicDB Dataset Audit Summary ===")
    print(f"Total verified test samples: {len(df)}")
    print(f"Resolution: All images are {df['width'].iloc[0]}x{df['height'].iloc[0]} (Standardized 384x288 format)")
    print(f"Polyp Area % Summary: Min={df['area_pct'].min():.2f}%, Median={df['area_pct'].median():.2f}%, Mean={df['area_pct'].mean():.2f}%, Max={df['area_pct'].max():.2f}%")
    print("\nSize Bucket Distribution:")
    print(df["size_bucket"].value_counts())
    print("\nBucket Proportions (%):")
    print((df["size_bucket"].value_counts(normalize=True) * 100).round(2))

    # Save metadata CSV
    metadata_csv_path = splits_path / "cvc_clinicdb_metadata.csv"
    df.to_csv(metadata_csv_path, index=False)
    print(f"\nSaved metadata CSV to: {metadata_csv_path}")

    # Save Test Split JSON
    test_data = {
        "dataset": "CVC-ClinicDB",
        "split": "test_ood",
        "num_samples": len(df),
        "note": "Strictly held-out out-of-distribution evaluation set (Hospital Clinic Barcelona)",
        "samples": df.to_dict(orient="records"),
    }
    test_json_path = splits_path / "cvc_clinicdb_test.json"
    with open(test_json_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Saved OOD test split JSON to: {test_json_path}")


if __name__ == "__main__":
    ingest_cvc_clinicdb()
