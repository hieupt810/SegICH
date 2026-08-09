"""Combined BCE + Dice loss for binary segmentation over raw logits."""

import segmentation_models_pytorch as smp
import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCEWithLogitsLoss and smp DiceLoss, both operating on raw logits."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True, smooth=1e-5, eps=1e-7)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total, bce_component, dice_component)."""
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total, bce_loss, dice_loss
