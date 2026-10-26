"""
Spatiotemporal 3D Convolutional Neural Network Architectures Trained from Scratch.
Implements Factorized R(2+1)D Convolutions with Kaiming He Normal Initialization.
"""

from typing import Tuple
import torch
import torch.nn as nn


class Conv2Plus1D(nn.Module):
    """
    Factorized (2+1)D convolution decomposing 3D filter (t x d x d) into:
    1. Spatial Conv: (1 x d x d)
    2. BatchNorm3d + ReLU
    3. Temporal Conv: (t x 1 x 1)
    4. BatchNorm3d + ReLU
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        # Calculate optimal intermediate plane count (Tran et al., 2018)
        mid_planes = (in_planes * out_planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * out_planes)
        mid_planes = max(16, mid_planes)

        # Spatial 2D convolution over (H, W): kernel (1, 3, 3)
        self.spatial_conv = nn.Conv3d(
            in_planes,
            mid_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm3d(mid_planes)
        self.relu = nn.ReLU(inplace=False)

        # Temporal 1D convolution over (T): kernel (3, 1, 1)
        self.temporal_conv = nn.Conv3d(
            mid_planes,
            out_planes,
            kernel_size=(3, 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=False,
        )
        self.bn_temporal = nn.BatchNorm3d(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.bn_spatial(x)
        x = self.relu(x)
        x = self.temporal_conv(x)
        x = self.bn_temporal(x)
        return self.relu(x)


class ResidualBlockR2Plus1D(nn.Module):
    """
    Residual bottleneck block utilizing factorized R(2+1)D convolutions.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        self.conv1 = Conv2Plus1D(in_planes, out_planes, stride=stride)
        self.conv2 = Conv2Plus1D(out_planes, out_planes, stride=(1, 1, 1))

        self.shortcut = nn.Sequential()
        if stride != (1, 1, 1) or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return torch.relu(out)


class Shoplifting3DCNN_Scratch(nn.Module):
    """
    Custom 3D CNN for Retail Surveillance Shoplifting Detection.
    Trained completely from scratch with Kaiming He Normal Initialization.
    Total parameter footprint: ~3.6M parameters.
    """

    def __init__(self, num_classes: int = 1, dropout: float = 0.4):
        super().__init__()
        # Input: (B, 3, 16, 224, 224) or (B, 16, 3, 224, 224)
        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # 4-Stage Hierarchical Spatiotemporal Residual Blocks
        self.layer1 = ResidualBlockR2Plus1D(32, 32, stride=(1, 1, 1))
        self.layer2 = ResidualBlockR2Plus1D(32, 64, stride=(2, 2, 2))
        self.layer3 = ResidualBlockR2Plus1D(64, 128, stride=(2, 2, 2))
        self.layer4 = ResidualBlockR2Plus1D(128, 256, stride=(2, 2, 2))

        # Global spatiotemporal average pooling: (B, 256, 1, 1, 1) -> (B, 256)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Binary classification projection head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout * 0.75),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming He Normal weight initialization for from-scratch training."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts both (B, T, C, H, W) and (B, C, T, H, W).
        Returns scalar logits tensor of shape (B,).
        """
        if x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
            # Permute (B, T, C, H, W) -> (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits.view(-1)
