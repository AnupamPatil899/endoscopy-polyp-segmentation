"""
Generates summary report JSON and plots from per_sample_metrics.csv.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(
    csv_path: str = "outputs/eval_in_distribution/per_sample_metrics.csv",
    output_dir: str = "outputs/eval_in_distribution",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1. Tier 1 Pixel Metrics
    tier1 = {
        "mean_dice": round(float(df["dice"].mean()), 4),
        "median_dice": round(float(df["dice"].median()), 4),
        "worst_decile_dice_p10": round(float(df["dice"].quantile(0.10)), 4),
        "mean_iou": round(float(df["iou"].mean()), 4),
        "median_iou": round(float(df["iou"].median()), 4),
        "worst_decile_iou_p10": round(float(df["iou"].quantile(0.10)), 4),
        "mean_precision": round(float(df["precision"].mean()), 4),
        "mean_recall": round(float(df["recall"].mean()), 4),
    }

    # 2. Tier 2 Polyp Detection
    total_gt = int(df["polyps_in_image"].sum())
    total_det = int(df["polyps_detected"].sum())
    sensitivity = float(total_det / max(total_gt, 1))

    tier2 = {
        "total_ground_truth_polyps": total_gt,
        "total_detected_polyps": total_det,
        "polyp_detection_sensitivity": round(sensitivity, 4),
    }

    # 3. Stratified by Size
    stratified = {}
    for bucket in ["Small", "Medium", "Large"]:
        group = df[df["size_bucket"] == bucket]
        if len(group) > 0:
            b_gt = int(group["polyps_in_image"].sum())
            b_det = int(group["polyps_detected"].sum())
            stratified[bucket] = {
                "num_samples": len(group),
                "mean_dice": round(float(group["dice"].mean()), 4),
                "median_dice": round(float(group["dice"].median()), 4),
                "worst_decile_dice_p10": round(float(group["dice"].quantile(0.10)), 4),
                "mean_iou": round(float(group["iou"].mean()), 4),
                "polyp_sensitivity": round(float(b_det / max(b_gt, 1)), 4),
            }

    report = {
        "total_samples": len(df),
        "tier1_pixel_metrics": tier1,
        "tier2_polyp_detection": tier2,
        "stratified_by_size": stratified,
    }

    report_path = out_path / "metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # 1. Distribution Plot
    plt.figure(figsize=(9, 5.5))
    dice_vals = df["dice"].values
    p10 = tier1["worst_decile_dice_p10"]
    mean_d = tier1["mean_dice"]
    median_d = tier1["median_dice"]

    plt.hist(dice_vals, bins=25, color="#3b82f6", edgecolor="black", alpha=0.7, density=True)
    plt.axvline(mean_d, color="#1e3a8a", linestyle="--", lw=2, label=f"Mean Dice: {mean_d:.4f}")
    plt.axvline(median_d, color="#10b981", linestyle="-", lw=2, label=f"Median Dice: {median_d:.4f}")
    plt.axvline(p10, color="#ef4444", linestyle=":", lw=2.5, label=f"10th Percentile (P10): {p10:.4f}")

    plt.title("Kvasir-SEG Validation — Dice Score Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Dice Score", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path / "eval_dice_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Stratification Plot
    plt.figure(figsize=(8, 5))
    buckets = ["Small", "Medium", "Large"]
    mean_dices = [stratified.get(b, {}).get("mean_dice", 0) for b in buckets]
    sensitivities = [stratified.get(b, {}).get("polyp_sensitivity", 0) for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35

    plt.bar(x - width/2, mean_dices, width, label="Mean Dice", color="#3b82f6")
    plt.bar(x + width/2, sensitivities, width, label="Polyp Sensitivity", color="#10b981")

    plt.title("Kvasir-SEG Validation — Stratified by Polyp Size", fontsize=13, fontweight="bold")
    plt.xticks(x, [f"{b}\n(N={stratified.get(b, {}).get('num_samples', 0)})" for b in buckets], fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(out_path / "eval_size_stratification.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated metrics report and plots in {out_path}")
    print("\n--- Summary Report ---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    summarize()
