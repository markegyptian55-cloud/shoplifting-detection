"""
Validation Threshold Sweep & Optimization for Imbalanced Action Recognition.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score


class ThresholdOptimizer:
    """
    Sweeps decision threshold on validation probabilities to optimize F1-Score.
    """

    def __init__(self, num_points: int = 81):
        self.thresholds = np.linspace(0.10, 0.90, num_points)

    def find_best_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Returns (best_threshold, best_f1) based on the validation predictions.
        """
        best_f1 = -1.0
        best_threshold = 0.5

        for th in self.thresholds:
            y_pred = (y_prob >= th).astype(int)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = round(float(th), 2)

        return best_threshold, best_f1
