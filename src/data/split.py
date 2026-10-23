"""
Group-Aware Leakage-Free Stratified Data Splitting.
Clusters sub-clips and siblings into mutually exclusive parent scene groups.
"""

import os
import re
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class GroupAwareStratifiedSplitter:
    """
    Extracts parent recording IDs to group sibling clips from the same camera
    scene, preventing static background shortcut leakage.
    """

    @staticmethod
    def extract_parent_id(filename: str, class_name: str = "") -> str:
        """
        Extracts master recording ID from filename to cluster sibling sub-clips.
        Matches exact pattern from notebook Cell 3.0:
        - 'shop_lifter_105_1.mp4' -> 'shop_lifter_105'
        - 'shop_lifter_n_63_2.mp4' -> 'shop_lifter_n_63'
        - 'videyyyyyyyyyss_57_1.mp4' -> 'videyyyyyyyyyss_57'
        """
        fname = os.path.basename(filename)
        m = re.match(r'^(shop_lifter(?:_n)?_\d+)', fname, re.IGNORECASE)
        if m:
            prefix = f"{class_name}/" if class_name else ""
            return f"{prefix}{m.group(1)}"
        m2 = re.match(r'^(.*?_\d+)_\d+\.', fname)
        if m2:
            prefix = f"{class_name}/" if class_name else ""
            return f"{prefix}{m2.group(1)}"
        prefix = f"{class_name}/" if class_name else ""
        return f"{prefix}{fname}"

    @classmethod
    def split(
        cls,
        df: pd.DataFrame,
        n_splits: int = 7,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into 70% Train, 15% Val, and 15% Test partitions
        guaranteeing zero group leakage across splits.
        """
        df = df.copy()
        if "group_id" not in df.columns:
            df["group_id"] = df["video_path"].apply(cls.extract_parent_id)

        min_class_count = df["label"].value_counts().min()
        actual_splits = max(2, min(n_splits, min_class_count))

        sgkf = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        splits = list(sgkf.split(df, df["label"], df["group_id"]))

        if actual_splits >= 3:
            val_idx = splits[0][1]
            test_idx = splits[1][1]
            train_idx = [i for i in range(len(df)) if i not in val_idx and i not in test_idx]
        else:
            val_idx = splits[0][1]
            test_idx = splits[0][1]
            train_idx = splits[1][1]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Verification of zero group leakage
        train_g = set(train_df["group_id"])
        val_g = set(val_df["group_id"])
        test_g = set(test_df["group_id"])

        assert len(train_g.intersection(val_g)) == 0, "Leakage detected: Train and Val share groups!"
        assert len(train_g.intersection(test_g)) == 0, "Leakage detected: Train and Test share groups!"
        assert len(val_g.intersection(test_g)) == 0, "Leakage detected: Val and Test share groups!"

        return train_df, val_df, test_df
