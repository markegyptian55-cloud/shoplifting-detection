"""
Class-Weighted Loss Formulation for Imbalanced Surveillance Action Recognition.
"""

from typing import Optional
import torch
import torch.nn as nn


def get_weighted_loss(
    pos_weight_value: float = 1.64,
    device: Optional[torch.device] = None,
) -> nn.BCEWithLogitsLoss:
    """
    Returns BCEWithLogitsLoss with positive class weight to penalize missed shoplifting
    without distorting decision bounds via crude oversampling.
    """
    weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
