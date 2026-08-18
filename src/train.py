"""
Training Engine & Experiment Tracker for Polyp Segmentation.

Features:
1. Mixed precision support (torch.cuda.amp) for fast GPU execution.
2. AdamW optimizer with differential learning rates (encoder vs decoder).
3. Cosine Annealing learning rate schedule with linear warmup.
4. Checkpointing strictly on best validation Dice score.
5. Per-epoch progress logging to CSV and experiment registry.
6. Support for fast CPU dry-runs and distributed hardware execution (Lightning AI / Local GPU / Colab).
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

# Ensure repo root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.losses import CombinedDiceBCELoss, compute_batch_dice, compute_batch_iou
from src.model import build_model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss, _, _ = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, _, _ = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        with torch.no_grad():
            batch_dice = compute_batch_dice(logits, masks, is_logits=True).sum().item()

        total_loss += loss.item() * batch_size
        total_dice += batch_dice
        num_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{batch_dice / batch_size:.4f}"})

    epoch_loss = total_loss / max(num_samples, 1)
    epoch_dice = total_dice / max(num_samples, 1)
    return epoch_loss, epoch_dice


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Runs validation pass and computes mean Loss, Dice, and IoU."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_samples = 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        logits = model(images)
        loss, _, _ = criterion(logits, masks)

        batch_dice = compute_batch_dice(logits, masks, is_logits=True).sum().item()
        batch_iou = compute_batch_iou(logits, masks, is_logits=True).sum().item()

        total_loss += loss.item() * batch_size
        total_dice += batch_dice
        total_iou += batch_iou
        num_samples += batch_size

    val_loss = total_loss / max(num_samples, 1)
    val_dice = total_dice / max(num_samples, 1)
    val_iou = total_iou / max(num_samples, 1)
    return val_loss, val_dice, val_iou


def log_experiment_to_registry(
    registry_path: Path,
    record: Dict,
):
    """Logs the completed experiment configuration and final metrics into master registry CSV."""
    if registry_path.exists():
        df = pd.read_csv(registry_path)
        # Append or replace if run_id exists
        if record["run_id"] in df["run_id"].values:
            df = df[df["run_id"] != record["run_id"]]
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(registry_path, index=False)
    print(f"\nExperiment registered in: {registry_path}")


def run_training(
    run_id: str = "baseline_unet_resnet34",
    train_json: str = "data/splits/kvasir_train.json",
    val_json: str = "data/splits/kvasir_val.json",
    output_dir: str = "outputs",
    epochs: int = 40,
    batch_size: int = 16,
    encoder_lr: float = 1e-4,
    decoder_lr: float = 5e-4,
    weight_decay: float = 1e-4,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    warmup_epochs: int = 3,
    num_workers: int = 4,
    img_size: int = 352,
    device_name: Optional[str] = None,
    dry_run: bool = False,
    dry_run_batches: int = 3,
):
    start_time = time.time()
    out_path = Path(output_dir)
    checkpoints_dir = out_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"==================================================")
    print(f"Starting Run: {run_id}")
    print(f"Device: {device} | Dry Run: {dry_run} | Epochs: {epochs}")
    print(f"Image Size: {img_size}x{img_size} | Batch Size: {batch_size}")
    print(f"==================================================")

    # Load Data
    train_loader, val_loader = get_dataloaders(
        train_json=train_json,
        val_json=val_json,
        batch_size=batch_size,
        num_workers=num_workers if not dry_run else 0,
        img_size=(img_size, img_size),
    )

    # If dry-run, create minimal subsets
    if dry_run:
        train_loader = list(train_loader)[:dry_run_batches]
        val_loader = list(val_loader)[:dry_run_batches]
        epochs = min(epochs, 2)

    # Build Model
    model = build_model(
        encoder_name="resnet34",
        encoder_weights="imagenet" if not dry_run else None,
        device=device,
    )

    # Criterion & Optimizer
    criterion = CombinedDiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight)
    param_groups = model.get_parameter_groups(
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        weight_decay=weight_decay,
    )
    optimizer = optim.AdamW(param_groups)

    # Learning Rate Scheduler with Warmup
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Tracking
    best_val_dice = 0.0
    best_val_iou = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[1]["lr"]  # Decoder LR

        train_loss, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )

        val_loss, val_dice, val_iou = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
            }
        )

        # Checkpoint on best Val Dice
        if val_dice > best_val_dice and not dry_run:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_epoch = epoch
            checkpoint_path = checkpoints_dir / f"{run_id}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "config": {
                        "encoder": "resnet34",
                        "img_size": img_size,
                        "dice_weight": dice_weight,
                        "bce_weight": bce_weight,
                    },
                },
                checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint to {checkpoint_path} (Val Dice: {val_dice:.4f})")

    # If dry run, save a sample checkpoint to verify file writing
    if dry_run:
        sample_ckpt = checkpoints_dir / f"{run_id}_dry_run.pth"
        torch.save({"dry_run": True, "model_state_dict": model.state_dict()}, sample_ckpt)
        best_val_dice = history[-1]["val_dice"]
        best_val_iou = history[-1]["val_iou"]
        best_epoch = epochs

    # Save per-epoch training curve CSV
    history_df = pd.DataFrame(history)
    history_csv_path = out_path / f"{run_id}_training_log.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"\nSaved training curve to: {history_csv_path}")

    # Register into Master Registry
    total_elapsed = time.time() - start_time
    registry_record = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "encoder_lr": encoder_lr,
        "decoder_lr": decoder_lr,
        "dice_weight": dice_weight,
        "bce_weight": bce_weight,
        "best_epoch": best_epoch,
        "best_val_dice": round(best_val_dice, 4),
        "best_val_iou": round(best_val_iou, 4),
        "total_train_time_min": round(total_elapsed / 60.0, 2),
        "is_dry_run": dry_run,
    }
    log_experiment_to_registry(out_path / "experiments_registry.csv", registry_record)

    return registry_record


def parse_args():
    parser = argparse.ArgumentParser(description="Train Polyp Segmentation Model")
    parser.add_argument("--run_id", type=str, default="baseline_unet_resnet34", help="Unique run identifier")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--encoder_lr", type=float, default=1e-4, help="Learning rate for encoder")
    parser.add_argument("--decoder_lr", type=float, default=5e-4, help="Learning rate for decoder")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Weight for Dice loss")
    parser.add_argument("--bce_weight", type=float, default=1.0, help="Weight for BCE loss")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--img_size", type=int, default=352, help="Input spatial dimension (HxW)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps)")
    parser.add_argument("--dry_run", action="store_true", help="Run quick 2-epoch dry run for validation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args() if "parser_args" in locals() else parse_args()
    run_training(
        run_id=args.run_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        warmup_epochs=args.warmup_epochs,
        img_size=args.img_size,
        device_name=args.device,
        dry_run=args.dry_run,
    )
