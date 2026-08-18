"""
Plots training and validation loss, Dice score, IoU, and Learning Rate curves from training log CSV.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(
    csv_path: str = "outputs/baseline_kvasir_unet_resnet34_training_log.csv",
    output_png: str = "outputs/baseline_training_curves.png",
):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = df["epoch"]

    # 1. Loss Curve
    axes[0, 0].plot(epochs, df["train_loss"], label="Train Loss (Dice+BCE)", color="#1f77b4", lw=2)
    axes[0, 0].plot(epochs, df["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0, 0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Dice Score Curve
    axes[0, 1].plot(epochs, df["train_dice"], label="Train Dice", color="#2ca02c", lw=2)
    axes[0, 1].plot(epochs, df["val_dice"], label="Val Dice", color="#d62728", lw=2, linestyle="--")
    best_dice = df["val_dice"].max()
    best_epoch = df.loc[df["val_dice"].idxmax(), "epoch"]
    axes[0, 1].scatter([best_epoch], [best_dice], color="#d62728", s=100, zorder=5, label=f"Best Val Dice: {best_dice:.4f} (Ep {int(best_epoch)})")
    axes[0, 1].set_title("Dice Overlap Score (Train vs Val)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice Score")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. IoU Curve
    axes[1, 0].plot(epochs, df["val_iou"], label="Val IoU (Jaccard)", color="#9467bd", lw=2)
    best_iou = df["val_iou"].max()
    axes[1, 0].scatter([best_epoch], [df.loc[df["epoch"] == best_epoch, "val_iou"].values[0]], color="#9467bd", s=100, zorder=5, label=f"Peak IoU: {best_iou:.4f}")
    axes[1, 0].set_title("Validation IoU (Jaccard Index)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Learning Rate Schedule
    axes[1, 1].plot(epochs, df["lr"], label="Cosine LR Schedule with Warmup", color="#8c564b", lw=2)
    axes[1, 1].set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves plot to: {output_png}")


if __name__ == "__main__":
    plot_curves()
