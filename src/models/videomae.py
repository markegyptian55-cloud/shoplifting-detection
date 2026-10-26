"""
VideoMAE (Video Masked Autoencoder) Foundation Model Adaptation.
Spatiotemporal Vision Transformer fine-tuned for surveillance action recognition.
Reference: Feichtenhofer et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training", NeurIPS 2022.
"""

import os
from typing import Optional, Union
import torch
import torch.nn as nn

try:
    from transformers import VideoMAEForVideoClassification, VideoMAEConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ShopliftingVideoMAE(nn.Module):
    """
    Shoplifting Video Surveillance Classifier based on VideoMAE Vision Transformer.
    Processes video clip tensors and outputs binary classification logits.
    """

    def __init__(self, base_model: Optional[nn.Module] = None):
        super().__init__()
        if base_model is not None:
            self.base_model = base_model
        else:
            if not HAS_TRANSFORMERS:
                raise ImportError(
                    "The 'transformers' package is required to instantiate VideoMAE. "
                    "Install via: pip install transformers"
                )
            self.base_model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base",
                num_labels=1,
                ignore_mismatched_sizes=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VideoMAE classifier.
        Args:
            x: Video tensor of shape (B, T, C, H, W) or (B, C, T, H, W).
               Normalized with ImageNet statistics.
        Returns:
            torch.Tensor: Logits tensor of shape (B,) for binary classification.
        """
        # Shape normalization: ensure (B, T, C, H, W)
        if x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            # (B, C, T, H, W) -> (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        outputs = self.base_model(pixel_values=x)
        logits = outputs.logits
        if logits.dim() == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        elif logits.dim() == 1:
            pass
        else:
            logits = logits.view(-1)
        return logits


def build_videomae_classifier(
    pretrained_model_name: str = "MCG-NJU/videomae-base",
    num_labels: int = 1,
    pretrained: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> ShopliftingVideoMAE:
    """
    Factory function to instantiate ShopliftingVideoMAE with custom configuration.
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "The 'transformers' package is required. Install via: pip install transformers"
        )

    device = device or torch.device("cpu")

    if pretrained:
        raw_videomae = VideoMAEForVideoClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        config = VideoMAEConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
        )
        raw_videomae = VideoMAEForVideoClassification(config)

    model = ShopliftingVideoMAE(raw_videomae).to(device)
    return model


def load_trained_videomae(
    checkpoint_path: str,
    base_model_name: str = "MCG-NJU/videomae-base",
    device: Optional[Union[str, torch.device]] = None,
) -> ShopliftingVideoMAE:
    """
    Loads fine-tuned VideoMAE weights from a saved checkpoint dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint artifact not found at: {checkpoint_path}")

    device = device or torch.device("cpu")
    model = build_videomae_classifier(
        pretrained_model_name=base_model_name,
        num_labels=1,
        pretrained=False,
        device=device,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Strip 'module.' prefix if saved with DataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("module.", "")
        cleaned_state_dict[clean_key] = v

    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model
