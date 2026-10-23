"""
Clip-Consistent Video Spatial Transformations using torchvision.transforms.v2.
Enforces identical geometric transforms across all temporal frames of a single clip.
"""

from typing import Tuple
import torch
from torchvision.transforms import v2


def get_video_transforms(
    is_train: bool = True,
    target_size: Tuple[int, int] = (224, 224),
) -> v2.Compose:
    """
    Returns torchvision.transforms.v2 pipeline supporting 4D/5D video tensors (T, C, H, W).
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=target_size, scale=(0.85, 1.0), antialias=True),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
