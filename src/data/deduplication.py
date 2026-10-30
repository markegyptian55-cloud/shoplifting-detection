"""
Dataset Deduplication & Cross-Class Embedding Similarity Audit.
Detects identical duplicate copies and audits feature similarity to prevent shortcut leakage.
"""

import re
from typing import List, Tuple, Set
import pandas as pd
import numpy as np


def detect_duplicate_copies(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]]]:
    """
    Identifies duplicate clips (such as '_1.mp4' sibling copies) that share
    an existing parent base filename in the dataset.
    Returns clean rows, duplicate rows, and (base, duplicate) filename pairs.
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    all_fnames: Set[str] = set(df["filename"].astype(str))
    clean_rows = []
    duplicate_rows = []
    candidate_pairs = []

    for _, row in df.iterrows():
        fname = str(row["filename"])
        match = re.fullmatch(r"(.+)_1\.(mp4|avi|mov|mkv)", fname, flags=re.IGNORECASE)
        parent = f"{match.group(1)}.{match.group(2)}" if match else None

        if parent in all_fnames:
            duplicate_rows.append(row)
            candidate_pairs.append((parent, fname))
        else:
            clean_rows.append(row)

    clean_df = pd.DataFrame(clean_rows).reset_index(drop=True)
    dup_df = pd.DataFrame(duplicate_rows).reset_index(drop=True)
    return clean_df, dup_df, candidate_pairs


def compute_pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes normalized pairwise cosine similarity matrix S in [-1, 1].
    S_ij = (e_i . e_j) / (||e_i|| * ||e_j||)
    """
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    norm_embeddings = embeddings / norm
    sim_matrix = np.matmul(norm_embeddings, norm_embeddings.T)
    return sim_matrix


def audit_pairwise_similarity(base_features: np.ndarray, duplicate_features: np.ndarray) -> np.ndarray:
    """Return cosine similarity for corresponding base/duplicate feature rows."""
    base_features = np.asarray(base_features, dtype=np.float64)
    duplicate_features = np.asarray(duplicate_features, dtype=np.float64)
    if base_features.size == 0 or duplicate_features.size == 0:
        raise ValueError("Input DataFrame cannot be empty")
    if base_features.shape != duplicate_features.shape or base_features.ndim != 2:
        raise ValueError("Base and duplicate features must have the same 2D shape.")

    base_norms = np.linalg.norm(base_features, axis=1)
    duplicate_norms = np.linalg.norm(duplicate_features, axis=1)
    denominators = base_norms * duplicate_norms
    denominators = np.maximum(denominators, np.finfo(np.float64).eps)
    return np.einsum("ij,ij->i", base_features, duplicate_features) / denominators
