"""
Rigorous Evaluation Metrics & Bootstrap Confidence Intervals.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)


def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Computes non-parametric empirical bootstrap confidence interval for Macro F1.
    """
    rng = np.random.RandomState(random_state)
    bootstrapped_f1s = []
    n = len(y_true)

    for _ in range(num_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        score = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        bootstrapped_f1s.append(score)

    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = float(np.percentile(bootstrapped_f1s, alpha * 100))
    upper_bound = float(np.percentile(bootstrapped_f1s, (1.0 - alpha) * 100))

    return lower_bound, upper_bound


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes complete benchmark metrics dictionary.
    """
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    binary_f1 = f1_score(y_true, y_pred, zero_division=0)

    lower_f1, upper_f1 = bootstrap_f1_ci(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "binary_f1": binary_f1,
        "f1_ci_lower": lower_f1,
        "f1_ci_upper": upper_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }
