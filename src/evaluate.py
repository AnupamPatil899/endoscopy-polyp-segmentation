"""
Multi-Tier Evaluation Engine for Polyp Segmentation.

Implements the complete 3-Tier clinical evaluation framework:
1. Tier 1 (Pixel-level): Mean, Median, Worst-Decile (10th percentile) Dice & IoU, Precision, Recall.
2. Tier 2 (Polyp-level): Connected-component detection sensitivity (did the model find each individual lesion?).
3. Tier 3 (Calibration & Uncertainty): Continuous probability calibration, Expected Calibration Error (ECE), Brier Score, and Reliability diagram binning.
4. Stratification: Granular breakdown across Small (<5%), Medium (5-20%), and Large (>20%) polyps.

Designed to run identically on both in-distribution validation (Kvasir-SEG) and held-out OOD test (CVC-ClinicDB).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import torch
from tqdm import tqdm

from src.dataset import PolypDataset, get_val_transforms
from src.losses import compute_batch_dice, compute_batch_iou
from src.model import build_model


def compute_polyp_level_detection(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
    min_overlap_ratio: float = 0.1,
) -> Tuple[int, int]:
    """
    Computes object-level polyp detection sensitivity via connected components.
    Args:
        pred_binary: 2D numpy array {0, 1}
        target_binary: 2D numpy array {0, 1}
        min_overlap_ratio: minimum overlap fraction of ground truth lesion to count as detected.
    Returns:
        (detected_polyps_count, total_ground_truth_polyps_count)
    """
    # Label connected components in target mask
    labeled_target, num_polyps = ndimage.label(target_binary)

    if num_polyps == 0:
        return 0, 0

    detected_count = 0
    for polyp_id in range(1, num_polyps + 1):
        polyp_mask = (labeled_target == polyp_id)
        polyp_area = np.sum(polyp_mask)

        # Overlap with predicted foreground
        overlap = np.sum(polyp_mask & (pred_binary > 0))
        if polyp_area > 0 and (overlap / polyp_area) >= min_overlap_ratio:
            detected_count += 1

    return detected_count, num_polyps


def compute_calibration_metrics(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, float, List[Dict]]:
    """
    Computes Brier Score, Expected Calibration Error (ECE), and Reliability diagram bins.
    """
    # Flatten arrays
    probs = all_probs.flatten()
    targets = all_targets.flatten()

    # Brier Score = MSE(probs, targets)
    brier_score = float(np.mean((probs - targets) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_details = []
    ece = 0.0
    total_pixels = len(probs)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples in this bin
        if i == num_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            avg_confidence = float(np.mean(probs[in_bin]))
            avg_accuracy = float(np.mean(targets[in_bin]))
            bin_weight = bin_count / total_pixels
            ece += bin_weight * abs(avg_accuracy - avg_confidence)

            bin_details.append(
                {
                    "bin_index": i,
                    "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                    "confidence": round(avg_confidence, 4),
                    "accuracy": round(avg_accuracy, 4),
                    "pixel_count": bin_count,
                }
            )

    return brier_score, float(ece), bin_details


def run_evaluation(
    model: torch.nn.Module,
    dataset: PolypDataset,
    device: torch.device,
    threshold: float = 0.5,
    img_size: int = 352,
) -> Tuple[Dict, pd.DataFrame]:
    """Runs complete 3-tier evaluation on dataset and returns metrics dictionary and sample-level dataframe."""
    model.eval()

    sample_records = []
    all_probs_list = []
    all_targets_list = []

    total_gt_polyps = 0
    total_detected_polyps = 0

    print(f"Evaluating {len(dataset)} samples on {device}...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            item = dataset[i]
            img = item["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
            target = item["mask"].unsqueeze(0).to(device) # (1, 1, H, W)
            fname = item["filename"]
            bucket = item["size_bucket"]
            area_pct = item["area_pct"]

            # Continuous probabilities and thresholded mask
            probs_tensor = model.predict_proba(img)
            pred_mask_tensor = (probs_tensor >= threshold).float()

            dice = compute_batch_dice(probs_tensor, target, threshold=threshold, is_logits=False).item()
            iou = compute_batch_iou(probs_tensor, target, threshold=threshold, is_logits=False).item()

            pred_np = pred_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
            target_np = target.squeeze().cpu().numpy().astype(np.uint8)
            prob_np = probs_tensor.squeeze().cpu().numpy().astype(np.float32)

            # Precision & Recall
            intersection = np.sum((pred_np == 1) & (target_np == 1))
            pred_sum = np.sum(pred_np == 1)
            target_sum = np.sum(target_np == 1)

            precision = float(intersection / (pred_sum + 1e-7)) if pred_sum > 0 else 1.0
            recall = float(intersection / (target_sum + 1e-7)) if target_sum > 0 else 1.0

            # Tier 2: Polyp-level detection
            detected, count = compute_polyp_level_detection(pred_np, target_np)
            total_detected_polyps += detected
            total_gt_polyps += count

            # Subsample probabilities for calibration to keep memory lightweight
            step = 4
            all_probs_list.append(prob_np[::step, ::step].flatten())
            all_targets_list.append(target_np[::step, ::step].flatten())

            sample_records.append(
                {
                    "filename": fname,
                    "size_bucket": bucket,
                    "area_pct": area_pct,
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "polyps_in_image": count,
                    "polyps_detected": detected,
                }
            )

    df_samples = pd.DataFrame(sample_records)

    # 1. Tier 1 Pixel Metrics
    tier1_metrics = {
        "mean_dice": float(df_samples["dice"].mean()),
        "median_dice": float(df_samples["dice"].median()),
        "worst_decile_dice_p10": float(df_samples["dice"].quantile(0.10)),
        "mean_iou": float(df_samples["iou"].mean()),
        "median_iou": float(df_samples["iou"].median()),
        "worst_decile_iou_p10": float(df_samples["iou"].quantile(0.10)),
        "mean_precision": float(df_samples["precision"].mean()),
        "mean_recall": float(df_samples["recall"].mean()),
    }

    # 2. Tier 2 Polyp Detection Sensitivity
    polyp_sensitivity = float(total_detected_polyps / max(total_gt_polyps, 1))
    tier2_metrics = {
        "total_ground_truth_polyps": total_gt_polyps,
        "total_detected_polyps": total_detected_polyps,
        "polyp_detection_sensitivity": polyp_sensitivity,
    }

    # 3. Tier 3 Probability Calibration
    concat_probs = np.concatenate(all_probs_list)
    concat_targets = np.concatenate(all_targets_list)
    brier, ece, bin_details = compute_calibration_metrics(concat_probs, concat_targets)

    tier3_metrics = {
        "brier_score": round(brier, 6),
        "expected_calibration_error_ece": round(ece, 6),
        "reliability_bins": bin_details,
    }

    # 4. Stratified by Polyp Size Bucket
    stratified_metrics = {}
    for bucket_name, group in df_samples.groupby("size_bucket"):
        stratified_metrics[bucket_name] = {
            "num_samples": len(group),
            "mean_dice": float(group["dice"].mean()),
            "median_dice": float(group["dice"].median()),
            "worst_decile_dice_p10": float(group["dice"].quantile(0.10)),
            "mean_iou": float(group["iou"].mean()),
            "polyp_sensitivity": float(group["polyps_detected"].sum() / max(group["polyps_in_image"].sum(), 1)),
        }

    full_results = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(df_samples),
        "threshold": threshold,
        "tier1_pixel_metrics": tier1_metrics,
        "tier2_polyp_detection": tier2_metrics,
        "tier3_calibration": tier3_metrics,
        "stratified_by_size": stratified_metrics,
    }

    return full_results, df_samples


def plot_evaluation_summary(
    df_samples: pd.DataFrame,
    eval_results: Dict,
    output_dir: Path,
    dataset_name: str = "Kvasir-SEG (Val)",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution Plot (Histogram + KDE + 10th percentile marker)
    plt.figure(figsize=(9, 6))
    dice_vals = df_samples["dice"].values
    p10 = eval_results["tier1_pixel_metrics"]["worst_decile_dice_p10"]
    mean_d = eval_results["tier1_pixel_metrics"]["mean_dice"]
    median_d = eval_results["tier1_pixel_metrics"]["median_dice"]

    plt.hist(dice_vals, bins=25, color="#3b82f6", edgecolor="black", alpha=0.7, density=True)
    plt.axvline(mean_d, color="#1e3a8a", linestyle="--", lw=2, label=f"Mean Dice: {mean_d:.4f}")
    plt.axvline(median_d, color="#10b981", linestyle="-", lw=2, label=f"Median Dice: {median_d:.4f}")
    plt.axvline(p10, color="#ef4444", linestyle=":", lw=2.5, label=f"10th Percentile (Worst-Decile): {p10:.4f}")

    plt.title(f"{dataset_name} — Dice Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Dice Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    dist_plot_path = output_dir / "eval_dice_distribution.png"
    plt.savefig(dist_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Stratification Bar Plot
    plt.figure(figsize=(8, 5))
    strat = eval_results["stratified_by_size"]
    buckets = ["Small", "Medium", "Large"]
    mean_dices = [strat.get(b, {}).get("mean_dice", 0) for b in buckets]
    sensitivities = [strat.get(b, {}).get("polyp_sensitivity", 0) for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35

    plt.bar(x - width/2, mean_dices, width, label="Mean Dice", color="#3b82f6")
    plt.bar(x + width/2, sensitivities, width, label="Polyp Sensitivity", color="#10b981")

    plt.title(f"{dataset_name} — Performance Stratified by Polyp Size", fontsize=13, fontweight="bold")
    plt.xticks(x, [f"{b}\n(N={strat.get(b, {}).get('num_samples', 0)})" for b in buckets], fontsize=11)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis="y")
    strat_plot_path = output_dir / "eval_size_stratification.png"
    plt.savefig(strat_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Reliability Diagram (Calibration Curve)
    bins = eval_results["tier3_calibration"]["reliability_bins"]
    if bins:
        confs = [b["confidence"] for b in bins]
        accs = [b["accuracy"] for b in bins]

        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.plot(confs, accs, "s-", color="#8b5cf6", lw=2, label=f"Model (ECE={eval_results['tier3_calibration']['expected_calibration_error_ece']:.4f})")
        plt.title(f"{dataset_name} — Reliability Diagram", fontsize=13, fontweight="bold")
        plt.xlabel("Confidence (Predicted Probability)", fontsize=12)
        plt.ylabel("Accuracy (Empirical Proportion)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        calib_plot_path = output_dir / "eval_calibration_curve.png"
        plt.savefig(calib_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved evaluation plots to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Polyp Segmentation Model")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/baseline_kvasir_unet_resnet34_best.pth", help="Path to checkpoint")
    parser.add_argument("--data_json", type=str, default="data/splits/kvasir_val.json", help="Path to evaluation split JSON")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_in_distribution", help="Output directory for reports & plots")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--img_size", type=int, default=352, help="Image spatial resolution")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model & load weights
    model = build_model(encoder_name="resnet34", encoder_weights=None, device=device)
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        print(f"Loading checkpoint weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Evaluating with initialized weights for testing.")

    # Dataset
    dataset = PolypDataset(
        samples=args.data_json,
        transforms=get_val_transforms(img_size=(args.img_size, args.img_size)),
    )

    # Run Evaluation
    results, df_samples = run_evaluation(
        model=model,
        dataset=dataset,
        device=device,
        threshold=args.threshold,
        img_size=args.img_size,
    )

    # Save JSON & CSV
    results_json_path = out_dir / "metrics_report.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)

    samples_csv_path = out_dir / "per_sample_metrics.csv"
    df_samples.to_csv(samples_csv_path, index=False)

    # Plots
    plot_evaluation_summary(df_samples, results, out_dir, dataset_name="Kvasir-SEG Validation")

    # Print Summary Table
    print("\n==================================================================")
    print("           MULTI-TIER EVALUATION RESULTS (IN-DISTRIBUTION)        ")
    print("==================================================================")
    print(f"Total Evaluated Samples: {results['total_samples']}")
    print("\n--- Tier 1: Pixel Overlap Metrics ---")
    t1 = results["tier1_pixel_metrics"]
    print(f"Mean Dice:           {t1['mean_dice']:.4f}")
    print(f"Median Dice:         {t1['median_dice']:.4f}")
    print(f"Worst-Decile (P10):  {t1['worst_decile_dice_p10']:.4f}  <-- Tail Risk Indicator")
    print(f"Mean IoU (Jaccard):  {t1['mean_iou']:.4f}")
    print(f"Median IoU:          {t1['median_iou']:.4f}")
    print(f"Mean Precision:      {t1['mean_precision']:.4f}")
    print(f"Mean Recall:         {t1['mean_recall']:.4f}")

    print("\n--- Tier 2: Polyp-Level Detection Sensitivity ---")
    t2 = results["tier2_polyp_detection"]
    print(f"Total Ground Truth Polyps: {t2['total_ground_truth_polyps']}")
    print(f"Detected Polyps:           {t2['total_detected_polyps']}")
    print(f"Polyp Detection Recall:    {t2['polyp_detection_sensitivity']*100:.2f}%")

    print("\n--- Tier 3: Probability Calibration & Uncertainty ---")
    t3 = results["tier3_calibration"]
    print(f"Brier Score: {t3['brier_score']:.6f}  (Lower is better)")
    print(f"Expected Calibration Error (ECE): {t3['expected_calibration_error_ece']:.4f}")

    print("\n--- Stratification by Polyp Size ---")
    for bucket, b_metrics in results["stratified_by_size"].items():
        print(f"  [{bucket.upper()} (N={b_metrics['num_samples']})]: Mean Dice = {b_metrics['mean_dice']:.4f} | Median Dice = {b_metrics['median_dice']:.4f} | Sensitivity = {b_metrics['polyp_sensitivity']*100:.1f}%")
    print("==================================================================")


if __name__ == "__main__":
    main()
