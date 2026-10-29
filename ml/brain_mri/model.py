"""
Standard PyTorch 3D U-Net Architecture for Volumetric Brain MRI Segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv3D(nn.Module):
    """(Conv3d -> InstanceNorm3d -> LeakyReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down3D(nn.Module):
    """Downscaling with MaxPool3d then DoubleConv3D"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """Upscaling with ConvTranspose3d then DoubleConv3D"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Spatial dimension difference padding if necessary
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)

        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x1 = F.pad(
                x1,
                [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2,
                ],
            )

        # Concatenate skip connection along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric multi-channel brain lesion segmentation.
    Default: 4 input channels (FLAIR, T1, T1ce, T2), 4 output classes (0, 1, 2, 3).
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters

        f = base_filters
        self.inc = DoubleConv3D(in_channels, f)
        self.down1 = Down3D(f, f * 2)       # 16 -> 32
        self.down2 = Down3D(f * 2, f * 4)   # 32 -> 64
        self.down3 = Down3D(f * 4, f * 8)   # 64 -> 128

        self.up1 = Up3D(f * 8, f * 4)       # 128 -> 64
        self.up2 = Up3D(f * 4, f * 2)       # 64 -> 32
        self.up3 = Up3D(f * 2, f)           # 32 -> 16
        self.outc = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Runs forward pass and takes argmax over logits to yield class mask."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
