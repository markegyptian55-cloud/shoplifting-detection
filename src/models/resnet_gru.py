"""
Hybrid Spatiotemporal Architecture for Surveillance Video Shoplifting Detection.
Combines 2D ResNet-18 spatial feature extraction with a 2-layer Bidirectional GRU.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SpatiotemporalResNetGRU(nn.Module):
    """
    Spatiotemporal Architecture:
    - Spatial Backbone: Pretrained ResNet-18 (512-dim frame representation)
    - Temporal Aggregator: 2-Layer Bidirectional GRU (hidden_dim=256, dropout=0.3)
    - Temporal Pooling: Sequence Mean-Pooling over 16 time steps
    - Classification Head: Multi-Layer Perceptron (512 -> 256 -> 1 logit)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Bidirectional GRU for temporal aggregation
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification Head: 2 * hidden_dim -> hidden_dim -> 1 logit
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def conv1(self):
        return self.backbone.conv1

    @property
    def layer1(self):
        return self.backbone.layer1

    @property
    def layer2(self):
        return self.backbone.layer2

    @property
    def layer3(self):
        return self.backbone.layer3

    @property
    def layer4(self):
        return self.backbone.layer4

    def freeze_backbone(self) -> None:
        """Stage 1: Freeze all spatial backbone parameters while keeping BatchNorm in eval mode"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()

    def unfreeze_layer4(self) -> None:
        """Stage 2: Unfreeze layer4 while keeping earlier stages frozen and BatchNorm in eval mode"""
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input Tensor Contract:  (B, T, C, H, W)
        Output Logits Contract: (B,)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # Spatial extraction: (B*T, 512) -> (B, T, 512)
        spatial_feats = self.backbone(x_flat)
        temporal_feats = spatial_feats.view(B, T, -1)

        # Temporal aggregation: (B, T, 2*hidden_dim)
        gru_out, _ = self.gru(temporal_feats)

        # Sequence mean-pooling across all temporal steps
        pooled = torch.mean(gru_out, dim=1)  # (B, 2*hidden_dim)

        # Prediction logit: (B, 1) -> (B,)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def load_trained_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> SpatiotemporalResNetGRU:
    """
    Loads saved checkpoint, strips DataParallel prefixes, and sets model to eval mode.
    """
    device = device or torch.device("cpu")
    model = SpatiotemporalResNetGRU(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Clean potential DataParallel prefix
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model
