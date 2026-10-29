"""
Loss functions and evaluation metrics for 3D Brain MRI Tumor Segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss for 3D Volumetric Segmentation.
    """

    def __init__(self, smooth: float = 1.0, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, D, H, W]
        targets: [B, D, H, W] (class indices 0 .. C-1)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets to [B, C, D, H, W]
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        start_idx = 0 if self.include_background else 1

        dice_per_class = []
        for c in range(start_idx, num_classes):
            p = probs[:, c].contiguous().view(-1)
            t = targets_one_hot[:, c].contiguous().view(-1)

            intersection = (p * t).sum()
            cardinality = (p * p).sum() + (t * t).sum()
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_per_class.append(dice)

        if not dice_per_class:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        mean_dice = torch.stack(dice_per_class).mean()
        return 1.0 - mean_dice


class DiceCELoss(nn.Module):
    """
    Combined Generalized Dice Loss + Cross-Entropy Loss for robust convergence.
    """

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth, include_background=False)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return (self.dice_weight * dice) + (self.ce_weight * ce)


def compute_binary_dice(pred_binary: torch.Tensor, target_binary: torch.Tensor, smooth: float = 1.0) -> float:
    """Computes binary Dice coefficient between two binary masks."""
    p = pred_binary.contiguous().view(-1).float()
    t = target_binary.contiguous().view(-1).float()
    intersection = (p * t).sum().item()
    total = p.sum().item() + t.sum().item()
    if total == 0:
        return 1.0  # Perfect agreement on empty mask
    return (2.0 * intersection + smooth) / (total + smooth)


def compute_subregion_dice(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes BraTS clinical composite subregion Dice scores:
      - WT (Whole Tumor): labels 1, 2, 3
      - TC (Tumor Core): labels 1, 3
      - ET (Enhancing Tumor): label 3
    """
    # Whole Tumor: any tumor label >= 1
    pred_wt = (pred_mask >= 1)
    target_wt = (target_mask >= 1)
    dice_wt = compute_binary_dice(pred_wt, target_wt)

    # Tumor Core: labels 1 (NCR) and 3 (ET)
    pred_tc = (pred_mask == 1) | (pred_mask == 3)
    target_tc = (target_mask == 1) | (target_mask == 3)
    dice_tc = compute_binary_dice(pred_tc, target_tc)

    # Enhancing Tumor: label 3
    pred_et = (pred_mask == 3)
    target_et = (target_mask == 3)
    dice_et = compute_binary_dice(pred_et, target_et)

    mean_dice = (dice_wt + dice_tc + dice_et) / 3.0

    return {
        "WT": round(dice_wt, 4),
        "TC": round(dice_tc, 4),
        "ET": round(dice_et, 4),
        "mean_dice": round(mean_dice, 4),
    }
