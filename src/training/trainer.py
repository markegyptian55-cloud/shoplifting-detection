"""
Two-Stage Spatiotemporal Optimization & Validation Trainer.
Enforces BatchNorm evaluation mode and mixed-precision acceleration.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


class SpatiotemporalTrainer:
    """
    Manages two-stage training (frozen backbone Stage 1 -> fine-tuned Layer 4 Stage 2).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        use_amp: bool = True,
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def enforce_batchnorm_eval(self) -> None:
        """
        Unconditionally enforces eval mode on all BatchNorm layers
        to prevent corruption of pretrained statistics.
        """
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model.train()
        self.enforce_batchnorm_eval()
        total_loss = 0.0

        for clips, labels in loader:
            clips = clips.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(clips)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(labels)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        threshold: float = 0.5,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluates model returning (val_loss, val_acc, val_f1, all_probs, all_targets).
        """
        self.model.eval()
        total_loss = 0.0
        all_probs: List[float] = []
        all_targets: List[float] = []

        for clips, labels in loader:
            clips = clips.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(clips)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

        y_prob = np.array(all_probs)
        y_true = np.array(all_targets)
        y_pred = (y_prob >= threshold).astype(int)

        val_loss = total_loss / len(loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, zero_division=0)

        return val_loss, val_acc, val_f1, y_prob, y_true
