"""Group-aware train, validation, and test splitting."""

import os
import re
from typing import Tuple

import numpy as np
import pandas as pd


class GroupAwareStratifiedSplitter:
    """Partition clips by recording root while balancing size and class counts."""

    @staticmethod
    def extract_group_id(filename: str, class_name: str = "") -> str:
        """Return a recording root shared by its numbered sub-clips.

        ``class_name`` is accepted for compatibility and intentionally ignored:
        a camera root must remain identical when it appears in both classes.
        """
        stem = os.path.splitext(os.path.basename(filename))[0]
        return re.sub(r"^(.*_\d+)_\d+$", r"\1", stem)

    extract_parent_id = extract_group_id

    @classmethod
    def split(
        cls,
        df: pd.DataFrame,
        random_state: int = 42,
        attempts: int = 64,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by group toward 70/15/15 while balancing class counts."""
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")
        if not {"video_path", "label"}.issubset(df.columns):
            raise ValueError("df must contain 'video_path' and 'label' columns")

        data = df.copy()
        data["group_id"] = data["video_path"].map(cls.extract_group_id)

        group_counts = pd.crosstab(data["group_id"], data["label"])
        groups = group_counts.index.to_numpy()
        if len(groups) < 3:
            raise ValueError("At least three recording groups are required for train/val/test splits")
        labels = group_counts.columns.to_list()
        counts = group_counts.to_numpy(dtype=float)
        sizes = counts.sum(axis=1)
        ratios = np.array([0.70, 0.15, 0.15])
        target_classes = ratios[:, None] * counts.sum(axis=0)[None, :]
        target_sizes = ratios * len(data)

        def score(partition_counts: np.ndarray, partition_sizes: np.ndarray) -> float:
            class_error = ((partition_counts - target_classes) / np.maximum(target_classes, 1)) ** 2
            size_error = ((partition_sizes - target_sizes) / target_sizes) ** 2
            return float(class_error.mean() + size_error.mean())

        rng = np.random.default_rng(random_state)
        best_assignment = None
        best_score = float("inf")

        for _ in range(max(1, attempts)):
            order = np.arange(len(groups))
            rng.shuffle(order)
            order = sorted(order, key=lambda i: sizes[i], reverse=True)
            assigned = np.zeros(len(groups), dtype=int)
            partition_counts = np.zeros((3, len(labels)), dtype=float)
            partition_sizes = np.zeros(3, dtype=float)

            for group_idx in order:
                candidates = []
                for partition in range(3):
                    partition_counts[partition] += counts[group_idx]
                    partition_sizes[partition] += sizes[group_idx]
                    candidates.append(score(partition_counts, partition_sizes))
                    partition_counts[partition] -= counts[group_idx]
                    partition_sizes[partition] -= sizes[group_idx]

                chosen = int(np.argmin(candidates))
                assigned[group_idx] = chosen
                partition_counts[chosen] += counts[group_idx]
                partition_sizes[chosen] += sizes[group_idx]

            final_score = score(partition_counts, partition_sizes)
            if final_score < best_score:
                best_score, best_assignment = final_score, assigned.copy()

        group_partition = dict(zip(groups, best_assignment))
        data["_partition"] = data["group_id"].map(group_partition)
        parts = [data.loc[data["_partition"] == i].drop(columns="_partition").reset_index(drop=True) for i in range(3)]
        train_df, val_df, test_df = parts

        train_groups = set(train_df["group_id"])
        val_groups = set(val_df["group_id"])
        test_groups = set(test_df["group_id"])
        assert train_groups.isdisjoint(val_groups), "Leakage detected: Train and Val share groups!"
        assert train_groups.isdisjoint(test_groups), "Leakage detected: Train and Test share groups!"
        assert val_groups.isdisjoint(test_groups), "Leakage detected: Val and Test share groups!"

        return train_df, val_df, test_df
