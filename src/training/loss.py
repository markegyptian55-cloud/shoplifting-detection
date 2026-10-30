"""
Class-Weighted Loss Formulation for Imbalanced Surveillance Action Recognition.
"""

from typing import Optional
import torch
import torch.nn as nn


def compute_pos_weight(num_normal: int, num_theft: int) -> float:
    """
    Computes exact balanced pos_weight ratio: num_negative / num_positive.
    For clean deduplicated training set (219 normal, 226 theft), pos_weight = 0.9690.
    """
    return float(num_normal) / float(max(num_theft, 1))


def get_weighted_loss(
    pos_weight_value: float = 0.9690,
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
