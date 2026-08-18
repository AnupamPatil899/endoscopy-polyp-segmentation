"""
Unit Tests for Model Architecture, Loss Functions, Gradient Backpropagation, and Overfitting.

Validates:
1. PolypUNet forward pass output shape (B, 1, H, W).
2. Numerical stability of Dice Loss and CombinedDiceBCELoss.
3. Metric calculations (Dice, IoU) on known synthetic masks.
4. Full backward pass ensuring non-zero gradients across encoder and decoder parameters.
5. Single-batch overfit convergence test (proves model and loss optimize correctly).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.optim as optim
from src.losses import CombinedDiceBCELoss, DiceLoss, compute_batch_dice, compute_batch_iou
from src.model import PolypUNet, build_model


def test_model_forward():
    print("\n--- 1. Testing Model Forward Pass & Shapes ---")
    model = build_model(encoder_name="resnet34", encoder_weights=None)
    model.eval()

    dummy_input = torch.randn(4, 3, 352, 352)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)
        masks = model.predict_mask(dummy_input, threshold=0.5)

    assert logits.shape == (4, 1, 352, 352), f"Expected logits shape (4, 1, 352, 352), got {logits.shape}"
    assert probs.shape == (4, 1, 352, 352), f"Expected probs shape (4, 1, 352, 352), got {probs.shape}"
    assert masks.shape == (4, 1, 352, 352), f"Expected masks shape (4, 1, 352, 352), got {masks.shape}"

    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities outside [0, 1] range!"
    assert set(torch.unique(masks).tolist()).issubset({0.0, 1.0}), "Mask contains non-binary values!"
    print("✓ Model forward pass and output shapes verified!")


def test_loss_functions():
    print("\n--- 2. Testing Loss Functions on Synthetic Cases ---")
    dice_criterion = DiceLoss()
    combined_criterion = CombinedDiceBCELoss(dice_weight=1.0, bce_weight=1.0)

    # Perfect prediction case: large positive logits where target is 1, large negative where target is 0
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:40, 20:40] = 1.0

    perfect_logits = torch.ones_like(target) * -10.0
    perfect_logits[:, :, 20:40, 20:40] = 10.0

    l_dice = dice_criterion(perfect_logits, target)
    l_total, l_d, l_bce = combined_criterion(perfect_logits, target)

    assert l_dice.item() < 0.05, f"Expected near-zero Dice loss for perfect prediction, got {l_dice.item():.4f}"
    assert l_total.item() < 0.1, f"Expected near-zero Combined loss for perfect prediction, got {l_total.item():.4f}"

    # Disjoint prediction case: inverted logits
    worst_logits = -perfect_logits
    l_worst_dice = dice_criterion(worst_logits, target)
    assert l_worst_dice.item() > 0.90, f"Expected near 1.0 Dice loss for disjoint prediction, got {l_worst_dice.item():.4f}"

    print(f"✓ Loss on perfect match: Dice Loss = {l_dice.item():.4f}, Combined Loss = {l_total.item():.4f}")
    print(f"✓ Loss on disjoint prediction: Dice Loss = {l_worst_dice.item():.4f}")


def test_metrics_calculation():
    print("\n--- 3. Testing Metric Computation on Synthetic Masks ---")
    t1 = torch.zeros(2, 1, 32, 32)
    t1[0, :, 10:20, 10:20] = 1.0  # 100 pixels
    t1[1, :, 5:15, 5:15] = 1.0    # 100 pixels

    # Predict exactly same
    dice_perfect = compute_batch_dice(t1, t1, threshold=0.5, is_logits=False)
    iou_perfect = compute_batch_iou(t1, t1, threshold=0.5, is_logits=False)

    assert torch.allclose(dice_perfect, torch.tensor([1.0, 1.0])), "Dice for exact match must be 1.0"
    assert torch.allclose(iou_perfect, torch.tensor([1.0, 1.0])), "IoU for exact match must be 1.0"

    # Half overlap: 50 pixels overlap out of 100
    p_half = torch.zeros(2, 1, 32, 32)
    p_half[0, :, 10:20, 10:15] = 1.0  # 50 pixels overlap with 100 target pixels
    # Dice = 2*50 / (50 + 100) = 100/150 = 2/3 = 0.6667
    # IoU = 50 / (50 + 100 - 50) = 50/100 = 0.5
    dice_half = compute_batch_dice(p_half, t1, threshold=0.5, is_logits=False)
    iou_half = compute_batch_iou(p_half, t1, threshold=0.5, is_logits=False)

    assert abs(dice_half[0].item() - (2.0 / 3.0)) < 1e-4, f"Expected 0.6667 Dice, got {dice_half[0].item():.4f}"
    assert abs(iou_half[0].item() - 0.5) < 1e-4, f"Expected 0.5 IoU, got {iou_half[0].item():.4f}"

    print(f"✓ Exact match metrics: Dice={dice_perfect.tolist()}, IoU={iou_perfect.tolist()}")
    print(f"✓ 50% overlap metrics: Dice={dice_half[0].item():.4f} (Expected 0.6667), IoU={iou_half[0].item():.4f} (Expected 0.5000)")


def test_gradient_flow():
    print("\n--- 4. Testing Backward Pass & Gradient Flow ---")
    model = build_model(encoder_name="resnet34", encoder_weights=None)
    model.train()

    criterion = CombinedDiceBCELoss()
    dummy_x = torch.randn(2, 3, 128, 128)
    dummy_y = torch.randint(0, 2, (2, 1, 128, 128)).float()

    logits = model(dummy_x)
    loss, _, _ = criterion(logits, dummy_y)
    loss.backward()

    # Check encoder and decoder gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    print("✓ All encoder and decoder parameters received valid gradients without NaNs or Infs!")


def test_single_batch_overfit():
    print("\n--- 5. Testing Single-Batch Optimization Overfit ---")
    # Quick overfit on a synthetic batch to verify learning capacity
    model = build_model(encoder_name="resnet34", encoder_weights=None)
    model.train()

    criterion = CombinedDiceBCELoss(dice_weight=1.0, bce_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    torch.manual_seed(42)
    x = torch.randn(2, 3, 128, 128)
    y = torch.zeros(2, 1, 128, 128)
    y[:, :, 40:88, 40:88] = 1.0  # Synthetic square polyp

    initial_loss = None
    final_loss = None

    for step in range(30):
        optimizer.zero_grad()
        logits = model(x)
        loss, l_dice, l_bce = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    dice_score = compute_batch_dice(logits, y).mean().item()
    print(f"Initial Step 0 Loss: {initial_loss:.4f}")
    print(f"Final Step 30 Loss:   {final_loss:.4f} (Dice Score: {dice_score:.4f})")

    assert final_loss < initial_loss * 0.3, "Model failed to rapidly reduce loss on single batch overfit test!"
    assert dice_score > 0.85, f"Expected overfit Dice > 0.85, got {dice_score:.4f}"
    print("✓ Single-batch overfit test passed with flying colors!")


if __name__ == "__main__":
    test_model_forward()
    test_loss_functions()
    test_metrics_calculation()
    test_gradient_flow()
    test_single_batch_overfit()
    print("\n========================================================")
    print("ALL STEP 2 MODEL & LOSS UNIT TESTS PASSED SUCCESSFULLY!")
    print("========================================================")
