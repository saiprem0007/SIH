"""
Checkpoint manager: save / load best and last checkpoints.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from anc.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Saves 'best' and 'last' checkpoints with full training state.

    Args:
        ckpt_dir   : directory for checkpoint files
        metric_name: which metric to track for 'best' ('si_sdr' or 'loss')
        higher_is_better: True for si_sdr/stoi, False for loss
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        metric_name: str = "si_sdr",
        higher_is_better: bool = True,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name      = metric_name
        self.higher_is_better = higher_is_better
        self._best_value: float = float("-inf") if higher_is_better else float("inf")
        self._history: list = []

    def is_better(self, value: float) -> bool:
        if self.higher_is_better:
            return value > self._best_value
        return value < self._best_value

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        cfg_dict: Optional[dict] = None,
    ) -> bool:
        """Save checkpoint. Returns True if this is the new best."""
        state = {
            "epoch":       epoch,
            "global_step": global_step,
            "metrics":     metrics,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if scheduler else None,
            "cfg":         cfg_dict,
        }

        # Always save 'last'
        last_path = self.ckpt_dir / "last.pt"
        torch.save(state, last_path)

        metric_val = metrics.get(self.metric_name, float("-inf"))
        is_best = self.is_better(metric_val)
        if is_best:
            self._best_value = metric_val
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(
                "New best checkpoint (epoch=%d, %s=%.4f) → %s",
                epoch, self.metric_name, metric_val, best_path,
            )

        # Save history
        self._history.append({"epoch": epoch, **metrics})
        with open(self.ckpt_dir / "history.json", "w") as f:
            json.dump(self._history, f, indent=2)

        return is_best

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        which: str = "last",
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint.

        Args:
            which: 'last' | 'best'
        Returns:
            state dict (contains epoch, global_step, metrics, cfg)
        """
        path = self.ckpt_dir / f"{which}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=device or "cpu")
        model.load_state_dict(state["model_state"])
        if optimizer and state.get("optim_state"):
            optimizer.load_state_dict(state["optim_state"])
        if scheduler and state.get("sched_state"):
            scheduler.load_state_dict(state["sched_state"])
        # Restore best value tracker
        metric_val = state["metrics"].get(self.metric_name)
        if metric_val is not None:
            self._best_value = metric_val
        logger.info(
            "Loaded %s checkpoint from epoch %d (%s=%.4f)",
            which, state["epoch"], self.metric_name, metric_val or float("nan"),
        )
        return state
