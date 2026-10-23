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
    - Classification Head: Multi-Layer Perceptron projecting to 1 logit
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained: bool = False,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Modular spatial stages
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Bidirectional GRU for temporal aggregation
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification Head: 2 * hidden_dim -> 128 -> 1 logit
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def extract_spatial_features(self, x_frames: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim spatial embedding for each individual frame"""
        x = self.conv1(x_frames)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input Tensor Contract:  (B, T, C, H, W)
        Output Logits Contract: (B,)
        """
        B, T, C, H, W = x.size()
        x_reshaped = x.view(B * T, C, H, W)

        # Spatial extraction: (B*T, 512) -> (B, T, 512)
        spatial_feats = self.extract_spatial_features(x_reshaped)
        spatial_feats = spatial_feats.view(B, T, -1)

        # Temporal aggregation: (B, T, 2*hidden_dim)
        gru_out, _ = self.gru(spatial_feats)

        # Sequence mean-pooling across all temporal steps
        pooled = torch.mean(gru_out, dim=1)  # (B, 2*hidden_dim)

        # Prediction logit: (B, 1) -> (B,)
        logits = self.classifier(pooled)
        return logits.squeeze(-1)

    def freeze_backbone(self) -> None:
        """Stage 1: Freeze all spatial backbone parameters"""
        for name, param in self.named_parameters():
            if not ("gru" in name or "classifier" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_layer4(self) -> None:
        """Stage 2: Unfreeze layer4 while keeping earlier stages frozen"""
        for name, param in self.named_parameters():
            if "layer4" in name or "gru" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


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
