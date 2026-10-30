"""
Data processing, sampling, augmentation, and split management modules.
"""

from .sampling import UniformVideoSampler
from .transforms import get_video_transforms
from .dataset import ShopliftingVideoDataset
from .split import GroupAwareStratifiedSplitter
from .deduplication import detect_duplicate_copies, compute_pairwise_cosine_similarity, audit_pairwise_similarity

__all__ = [
    "UniformVideoSampler",
    "get_video_transforms",
    "ShopliftingVideoDataset",
    "GroupAwareStratifiedSplitter",
    "detect_duplicate_copies",
    "compute_pairwise_cosine_similarity",
    "audit_pairwise_similarity",
]
