"""
PyTorch Dataset for Shoplifting Video Surveillance Action Recognition.
Strictly outputs 4D tensor of shape (T, C, H, W) per video clip.
"""

import os
from typing import Callable, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .sampling import UniformVideoSampler
from .transforms import get_video_transforms


class ShopliftingVideoDataset(Dataset):
    """
    Video Dataset loading cached .npy arrays or decoding directly from video files.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        is_train: bool = False,
        num_frames: int = 16,
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.transform = transform or get_video_transforms(is_train=is_train)
        self.sampler = UniformVideoSampler(num_frames=num_frames)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        label = torch.tensor(row["label"], dtype=torch.float32)

        # 1. Try loading cached numpy array first for speed
        loaded_frames = None
        if self.cache_dir and "cache_path" in row and os.path.exists(str(row["cache_path"])):
            loaded_frames = np.load(str(row["cache_path"]))
        elif self.cache_dir:
            expected_cache = os.path.join(self.cache_dir, f"clip_{idx:04d}.npy")
            if os.path.exists(expected_cache):
                loaded_frames = np.load(expected_cache)

        # 2. Fallback to on-the-fly decoding if not cached
        if loaded_frames is None:
            video_path = row["video_path"]
            loaded_frames = self.sampler.decode_and_sample(video_path)

        # loaded_frames shape: (T, H, W, C) in uint8 [0, 255]
        # Convert to tensor and permute to (T, C, H, W)
        clip_tensor = torch.from_numpy(loaded_frames).permute(0, 3, 1, 2)  # (T, C, H, W)

        # Apply clip-consistent transformations
        if self.transform is not None:
            clip_tensor = self.transform(clip_tensor)

        return clip_tensor, label
