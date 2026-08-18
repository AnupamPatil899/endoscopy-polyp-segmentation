"""
Loss Functions and Evaluation Metrics for Polyp Segmentation.

Includes:
1. Soft Dice Loss (differentiable overlap optimization).
2. Binary Cross-Entropy with Logits (stable pixel-wise gradient signal).
3. Combined Dice + BCE Loss (hybrid loss balancing scale invariance and smooth convergence).
4. Exact metric calculators: Pixel Dice, IoU / Jaccard Index, Precision, Recall.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss computed directly from raw logits.
    """

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw unscaled model outputs.
            targets: (B, 1, H, W) binary ground truth {0.0, 1.0}.
        Returns:
            Scalar tensor loss: 1.0 - Dice.
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        loss = 1.0 - dice
        return loss.mean()


class CombinedDiceBCELoss(nn.Module):
    """
    Hybrid Loss combining Dice Loss and Binary Cross-Entropy Loss:
        Total Loss = alpha * DiceLoss + beta * BCEWithLogitsLoss
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: (B, 1, H, W) raw model outputs.
            targets: (B, 1, H, W) ground-truth masks.
        Returns:
            Tuple of (total_loss, dice_loss_val, bce_loss_val).
        """
        l_dice = self.dice_loss(logits, targets)
        l_bce = self.bce_loss(logits, targets)
        l_total = (self.dice_weight * l_dice) + (self.bce_weight * l_bce)
        return l_total, l_dice, l_bce


@torch.no_grad()
def compute_batch_dice(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes hard per-sample Dice score on thresholded predictions.
    Returns 1D tensor of length B containing each sample's Dice score.
    """
    if is_logits:
        probs = torch.sigmoid(logits_or_probs)
    else:
        probs = logits_or_probs

    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    cardinality = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    # If both pred and target are completely empty, Dice = 1.0
    both_empty = (cardinality == 0)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    dice[both_empty] = 1.0

    return dice


@torch.no_grad()
def compute_batch_iou(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes hard per-sample IoU (Jaccard Index) on thresholded predictions.
    Returns 1D tensor of length B containing each sample's IoU score.
    """
    if is_logits:
        probs = torch.sigmoid(logits_or_probs)
    else:
        probs = logits_or_probs

    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    both_empty = (union == 0)
    iou = (intersection + eps) / (union + eps)
    iou[both_empty] = 1.0

    return iou
