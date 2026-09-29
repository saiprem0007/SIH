"""
Warmup + CosineAnnealing LR scheduler.

Linearly warms up from 0 to base_lr over warmup_epochs,
then applies cosine annealing to min_lr.
"""
from __future__ import annotations

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing.

    Args:
        optimizer     : wrapped optimizer
        warmup_epochs : number of warmup epochs
        total_epochs  : total number of training epochs
        min_lr        : final learning rate after cosine decay
        last_epoch    : resume from epoch (-1 = start fresh)
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            # Linear warmup
            scale = (e + 1) / max(self.warmup_epochs, 1)
            return [base_lr * scale for base_lr in self.base_lrs]
        # Cosine decay
        cosine_epochs = self.total_epochs - self.warmup_epochs
        e_cos = e - self.warmup_epochs
        cos_val = 0.5 * (1 + math.cos(math.pi * e_cos / max(cosine_epochs, 1)))
        return [
            self.min_lr + (base_lr - self.min_lr) * cos_val
            for base_lr in self.base_lrs
        ]
