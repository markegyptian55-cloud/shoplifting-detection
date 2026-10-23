"""
Fixed Uniform Temporal Frame Sampling & Video Decoding Engine.
"""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class UniformVideoSampler:
    """
    Uniform skip-interval frame sampling engine for video anomaly detection.
    Guarantees fixed temporal length T across variable-length surveillance recordings.
    """

    def __init__(
        self,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        device: Optional[torch.device] = None,
    ):
        self.num_frames = num_frames
        self.target_size = target_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_frame_indices(self, total_frames: int) -> np.ndarray:
        """
        Computes mathematically uniform indices spanning [0, total_frames - 1].
        """
        if total_frames <= 0:
            raise ValueError(f"Total frames must be positive, got {total_frames}")
        if total_frames < self.num_frames:
            # Handle short clips by linearly repeating/padding indices
            return np.linspace(0, total_frames - 1, self.num_frames).round().astype(int)
        return np.round(np.linspace(0, total_frames - 1, self.num_frames)).astype(int)

    def decode_and_sample(self, video_path: str) -> np.ndarray:
        """
        Decodes video sequentially in a single pass (avoiding slow cap.set seeks)
        and resizes frames to target_size. Returns array of shape (T, H, W, C) in RGB.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video container: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback frame counting
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        target_indices = set(self.compute_frame_indices(total_frames).tolist())
        collected_frames = {}

        current_frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame_idx in target_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
                collected_frames[current_frame_idx] = frame_resized
            current_frame_idx += 1
            if len(collected_frames) == len(target_indices):
                break

        cap.release()

        # Reconstruct ordered sequence
        ordered_indices = self.compute_frame_indices(total_frames)
        output_frames = []
        for idx in ordered_indices:
            if idx in collected_frames:
                output_frames.append(collected_frames[idx])
            else:
                # Fallback to nearest available frame
                available_idx = min(collected_frames.keys(), key=lambda k: abs(k - idx))
                output_frames.append(collected_frames[available_idx])

        return np.array(output_frames, dtype=np.uint8)  # (T, H, W, C)

    def cache_video_to_disk(self, video_path: str, cache_path: str) -> str:
        """
        Extracts uniform frames and caches pre-processed numpy tensor to disk.
        """
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if not os.path.exists(cache_path):
            frames = self.decode_and_sample(video_path)
            np.save(cache_path, frames)
        return cache_path
