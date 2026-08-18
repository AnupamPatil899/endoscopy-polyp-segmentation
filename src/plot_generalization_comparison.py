"""
Comprehensive Generalization Study: In-Distribution (Kvasir-SEG) vs Out-of-Distribution (CVC-ClinicDB).

Calculates the exact generalization drop, stratified shifts, calibration metrics,
and plots publication-grade comparative diagnostic figures.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_generalization_analysis(
    in_dist_report_path: str = "outputs/eval_in_distribution/metrics_report.json",
    ood_report_path: str = "outputs/eval_ood_cvc_clinicdb/metrics_report.json",
    output_dir: str = "outputs",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(in_dist_report_path) as f:
        in_dist = json.load(f)
    with open(ood_report_path) as f:
        ood = json.load(f)

    # 1. Pixel Metrics Comparison
    in_t1 = in_dist["tier1_pixel_metrics"]
    ood_t1 = ood["tier1_pixel_metrics"]

    # 2. Detection Sensitivity Comparison
    in_t2 = in_dist["tier2_polyp_detection"]
    ood_t2 = ood["tier2_polyp_detection"]

    # 3. Stratified Comparison
    in_strat = in_dist["stratified_by_size"]
    ood_strat = ood["stratified_by_size"]

    # Compute exact deltas
    comparison = {
        "metrics_summary": {
            "mean_dice": {
                "in_dist_kvasir": round(in_t1["mean_dice"], 4),
                "ood_cvc_clinicdb": round(ood_t1["mean_dice"], 4),
                "delta": round(ood_t1["mean_dice"] - in_t1["mean_dice"], 4),
                "pct_change": round(((ood_t1["mean_dice"] - in_t1["mean_dice"]) / in_t1["mean_dice"]) * 100, 2),
            },
            "median_dice": {
                "in_dist_kvasir": round(in_t1["median_dice"], 4),
                "ood_cvc_clinicdb": round(ood_t1["median_dice"], 4),
                "delta": round(ood_t1["median_dice"] - in_t1["median_dice"], 4),
                "pct_change": round(((ood_t1["median_dice"] - in_t1["median_dice"]) / in_t1["median_dice"]) * 100, 2),
            },
            "worst_decile_dice_p10": {
                "in_dist_kvasir": round(in_t1["worst_decile_dice_p10"], 4),
                "ood_cvc_clinicdb": round(ood_t1["worst_decile_dice_p10"], 4),
                "delta": round(ood_t1["worst_decile_dice_p10"] - in_t1["worst_decile_dice_p10"], 4),
                "pct_change": round(((ood_t1["worst_decile_dice_p10"] - in_t1["worst_decile_dice_p10"]) / in_t1["worst_decile_dice_p10"]) * 100, 2),
            },
            "mean_iou": {
                "in_dist_kvasir": round(in_t1["mean_iou"], 4),
                "ood_cvc_clinicdb": round(ood_t1["mean_iou"], 4),
                "delta": round(ood_t1["mean_iou"] - in_t1["mean_iou"], 4),
                "pct_change": round(((ood_t1["mean_iou"] - in_t1["mean_iou"]) / in_t1["mean_iou"]) * 100, 2),
            },
            "polyp_sensitivity": {
                "in_dist_kvasir": round(in_t2["polyp_detection_sensitivity"], 4),
                "ood_cvc_clinicdb": round(ood_t2["polyp_detection_sensitivity"], 4),
                "delta": round(ood_t2["polyp_detection_sensitivity"] - in_t2["polyp_detection_sensitivity"], 4),
                "pct_change": round(((ood_t2["polyp_detection_sensitivity"] - in_t2["polyp_detection_sensitivity"]) / in_t2["polyp_detection_sensitivity"]) * 100, 2),
            },
        },
        "size_stratified_deltas": {
            "Small": {
                "in_dist_dice": round(in_strat["Small"]["mean_dice"], 4),
                "ood_dice": round(ood_strat["Small"]["mean_dice"], 4),
                "dice_delta": round(ood_strat["Small"]["mean_dice"] - in_strat["Small"]["mean_dice"], 4),
                "in_dist_sens": round(in_strat["Small"]["polyp_sensitivity"], 4),
                "ood_sens": round(ood_strat["Small"]["polyp_sensitivity"], 4),
                "sens_delta": round(ood_strat["Small"]["polyp_sensitivity"] - in_strat["Small"]["polyp_sensitivity"], 4),
            },
            "Medium": {
                "in_dist_dice": round(in_strat["Medium"]["mean_dice"], 4),
                "ood_dice": round(ood_strat["Medium"]["mean_dice"], 4),
                "dice_delta": round(ood_strat["Medium"]["mean_dice"] - in_strat["Medium"]["mean_dice"], 4),
                "in_dist_sens": round(in_strat["Medium"]["polyp_sensitivity"], 4),
                "ood_sens": round(ood_strat["Medium"]["polyp_sensitivity"], 4),
                "sens_delta": round(ood_strat["Medium"]["polyp_sensitivity"] - in_strat["Medium"]["polyp_sensitivity"], 4),
            },
            "Large": {
                "in_dist_dice": round(in_strat["Large"]["mean_dice"], 4),
                "ood_dice": round(ood_strat["Large"]["mean_dice"], 4),
                "dice_delta": round(ood_strat["Large"]["mean_dice"] - in_strat["Large"]["mean_dice"], 4),
                "in_dist_sens": round(in_strat["Large"]["polyp_sensitivity"], 4),
                "ood_sens": round(ood_strat["Large"]["polyp_sensitivity"], 4),
                "sens_delta": round(ood_strat["Large"]["polyp_sensitivity"] - in_strat["Large"]["polyp_sensitivity"], 4),
            },
        },
    }

    # Save summary json
    with open(out_path / "generalization_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # 4-Panel Comparative Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Primary Metrics In-Dist vs OOD
    ax1 = axes[0, 0]
    metrics_names = ["Mean Dice", "Median Dice", "P10 Dice", "Mean IoU", "Sensitivity"]
    in_vals = [
        in_t1["mean_dice"],
        in_t1["median_dice"],
        in_t1["worst_decile_dice_p10"],
        in_t1["mean_iou"],
        in_t2["polyp_detection_sensitivity"],
    ]
    ood_vals = [
        ood_t1["mean_dice"],
        ood_t1["median_dice"],
        ood_t1["worst_decile_dice_p10"],
        ood_t1["mean_iou"],
        ood_t2["polyp_detection_sensitivity"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = ax1.bar(x - width/2, in_vals, width, label="Kvasir-SEG (In-Dist)", color="#2563eb", alpha=0.9)
    rects2 = ax1.bar(x + width/2, ood_vals, width, label="CVC-ClinicDB (OOD)", color="#dc2626", alpha=0.9)

    ax1.set_title("Overall Generalization Metrics: In-Dist vs OOD", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=10)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3, axis="y")

    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 2: Mean Dice Stratified by Size
    ax2 = axes[0, 1]
    buckets = ["Small", "Medium", "Large"]
    in_dices = [in_strat[b]["mean_dice"] for b in buckets]
    ood_dices = [ood_strat[b]["mean_dice"] for b in buckets]

    x_b = np.arange(len(buckets))
    ax2.bar(x_b - width/2, in_dices, width, label="Kvasir-SEG (In-Dist)", color="#2563eb", alpha=0.9)
    ax2.bar(x_b + width/2, ood_dices, width, label="CVC-ClinicDB (OOD)", color="#dc2626", alpha=0.9)

    ax2.set_title("Mean Dice by Polyp Size Bucket", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_b)
    ax2.set_xticklabels([f"{b}\n(K={in_strat[b]['num_samples']} | CVC={ood_strat[b]['num_samples']})" for b in buckets], fontsize=10)
    ax2.set_ylabel("Mean Dice", fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Polyp Sensitivity Stratified by Size
    ax3 = axes[1, 0]
    in_sens = [in_strat[b]["polyp_sensitivity"] for b in buckets]
    ood_sens = [ood_strat[b]["polyp_sensitivity"] for b in buckets]

    ax3.bar(x_b - width/2, in_sens, width, label="Kvasir-SEG (In-Dist)", color="#10b981", alpha=0.9)
    ax3.bar(x_b + width/2, ood_sens, width, label="CVC-ClinicDB (OOD)", color="#f59e0b", alpha=0.9)

    ax3.set_title("Clinical Polyp Sensitivity (Recall) by Size Bucket", fontsize=12, fontweight="bold")
    ax3.set_xticks(x_b)
    ax3.set_xticklabels([f"{b}\n(K={in_strat[b]['num_samples']} | CVC={ood_strat[b]['num_samples']})" for b in buckets], fontsize=10)
    ax3.set_ylabel("Sensitivity (Recall)", fontsize=11)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Calibration Reliability Diagram (ECE & Confidence)
    ax4 = axes[1, 1]
    if "tier3_calibration" in ood:
        bins = ood["tier3_calibration"]["reliability_bins"]
        confidences = [b["confidence"] for b in bins if b["pixel_count"] > 0]
        accuracies = [b["accuracy"] for b in bins if b["pixel_count"] > 0]

        ax4.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfect Calibration")
        ax4.plot(confidences, accuracies, "o-", color="#8b5cf6", lw=2, ms=6, label=f"OOD Calibration (ECE = {ood['tier3_calibration']['expected_calibration_error_ece']:.4f})")
        ax4.fill_between(confidences, accuracies, confidences, color="#8b5cf6", alpha=0.2)

        ax4.set_title(f"Reliability Diagram (CVC-ClinicDB OOD)\nBrier Score: {ood['tier3_calibration']['brier_score']:.4f} | ECE: {ood['tier3_calibration']['expected_calibration_error_ece']:.4f}", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=10)
        ax4.set_ylabel("Empirical Accuracy", fontsize=10)
        ax4.set_xlim(-0.02, 1.02)
        ax4.set_ylim(-0.02, 1.02)
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_img_path = out_path / "generalization_comparison.png"
    plt.savefig(comparison_img_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Generated generalization comparison plot: {comparison_img_path}")
    print("\n--- Generalization Summary ---")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    generate_generalization_analysis()
