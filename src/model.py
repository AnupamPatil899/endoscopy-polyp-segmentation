"""
Polyp Segmentation Model Architecture.

Wraps segmentation_models_pytorch Unet with ResNet34 backbone pretrained on ImageNet.
Outputs raw unbounded logits (activation=None) to ensure numerical stability
during training with BCEWithLogitsLoss and to allow exact probability calibration analysis.
"""

from typing import Dict, List, Optional, Tuple
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class PolypUNet(nn.Module):
    """
    U-Net with ResNet34 backbone for binary polyp segmentation.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        # Raw logits output (no activation in model definition)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw unscaled logits.
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Logits tensor of shape (B, 1, H, W)
        """
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs inference and applies Sigmoid to return continuous probabilities in [0.0, 1.0].
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Probability map tensor of shape (B, 1, H, W)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Runs inference and thresholds probabilities into a binary mask {0.0, 1.0}.
        """
        proba = self.predict_proba(x)
        return (proba >= threshold).float()

    def get_parameter_groups(
        self,
        encoder_lr: float = 1e-4,
        decoder_lr: float = 5e-4,
        weight_decay: float = 1e-4,
    ) -> List[Dict]:
        """
        Returns differential parameter groups for fine-tuning:
        Lower LR on pretrained encoder to preserve ImageNet representations,
        Higher LR on newly initialized decoder head.
        """
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = (
            list(self.model.decoder.parameters())
            + list(self.model.segmentation_head.parameters())
        )

        return [
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "encoder",
            },
            {
                "params": decoder_params,
                "lr": decoder_lr,
                "weight_decay": weight_decay,
                "name": "decoder",
            },
        ]


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    device: Optional[torch.device] = None,
) -> PolypUNet:
    """Factory helper to build and place model onto target device."""
    model = PolypUNet(encoder_name=encoder_name, encoder_weights=encoder_weights)
    if device is not None:
        model = model.to(device)
    return model
