"""
Training engine for ANC-SIH Phase 1.

Handles one epoch of training and one validation pass, with:
- Gradient clipping and NaN guards
- Mixed-precision AMP (when CUDA available)
- TensorBoard logging of losses, metrics, audio samples
- Curriculum SNR scheduling
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # tensorboard not installed
    SummaryWriter = None  # type: ignore[assignment,misc]

from anc.eval.metrics import compute_si_sdr, compute_stoi
from anc.losses import CombinedLoss
from anc.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers (cheap, no PESQ)
# ---------------------------------------------------------------------------

def _batch_si_sdr(enhanced: torch.Tensor, clean: torch.Tensor) -> float:
    """Mean SI-SDR over a batch (numpy, no grad)."""
    e = enhanced.detach().cpu().numpy().astype(np.float64)
    c = clean.detach().cpu().numpy().astype(np.float64)
    vals = [compute_si_sdr(c[i], e[i]) for i in range(len(e))]
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def _batch_stoi(enhanced: torch.Tensor, clean: torch.Tensor, sr: int) -> float:
    """Mean STOI over a batch (numpy, no grad). Slow — only used for val logging."""
    e = enhanced.detach().cpu().numpy().astype(np.float64)
    c = clean.detach().cpu().numpy().astype(np.float64)
    vals = [compute_stoi(c[i], e[i], sr) for i in range(len(e))]
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Training engine
# ---------------------------------------------------------------------------

class TrainingEngine:
    """Encapsulates one full training + validation epoch.

    Args:
        model      : the neural model (nn.Module)
        criterion  : CombinedLoss instance
        optimizer  : AdamW or similar
        scaler     : GradScaler for AMP (None = no AMP)
        cfg        : Config object
        writer     : TensorBoard SummaryWriter
        device     : target device
        sr         : audio sample rate
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: CombinedLoss,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any],
        cfg,
        writer: Optional[SummaryWriter],
        device: torch.device,
        sr: int = 16000,
    ) -> None:
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scaler     = scaler
        self.cfg        = cfg
        self.writer     = writer
        self.device     = device
        self.sr         = sr
        self.grad_clip  = float(cfg.train.grad_clip)
        self.log_every  = int(cfg.train.log_every_n_steps)

    # ------------------------------------------------------------------
    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        global_step: int,
    ) -> Tuple[float, int]:
        """Run one training epoch.

        Returns:
            (mean_loss, updated_global_step)
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            primary   = batch["primary"].to(self.device)    # (B, T)
            reference = batch["reference"].to(self.device)
            clean     = batch["clean"].to(self.device)

            # Skip silent or malformed batches
            if primary.shape[-1] < 1024:
                logger.warning("Skipping very short batch (len=%d)", primary.shape[-1])
                continue

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    enhanced = self._forward(primary, reference)
                    loss, comps = self.criterion(enhanced, clean)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss at step %d — skipping", global_step)
                    global_step += 1
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                enhanced = self._forward(primary, reference)
                loss, comps = self.criterion(enhanced, clean)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss at step %d — skipping", global_step)
                    global_step += 1
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            # TensorBoard step logging
            if self.writer and global_step % self.log_every == 0:
                self.writer.add_scalar("train/loss_total",  comps["total"].item(), global_step)
                self.writer.add_scalar("train/loss_si_snr", comps["si_snr"].item(), global_step)
                self.writer.add_scalar("train/loss_stft",   comps["stft"].item(),   global_step)
                self.writer.add_scalar("train/loss_l1",     comps["l1"].item(),     global_step)
                si_sdr_val = _batch_si_sdr(enhanced, clean)
                self.writer.add_scalar("train/si_sdr", si_sdr_val, global_step)

            global_step += 1

        elapsed = time.time() - t0
        mean_loss = total_loss / max(n_batches, 1)
        logger.info(
            "[Epoch %3d] train_loss=%.4f  batches=%d  time=%.1fs",
            epoch, mean_loss, n_batches, elapsed,
        )
        return mean_loss, global_step

    # ------------------------------------------------------------------
    def validate(
        self,
        loader: DataLoader,
        epoch: int,
        global_step: int,
        log_audio: bool = False,
        audio_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Run validation pass.

        Returns:
            dict with keys: loss, si_sdr, stoi
        """
        self.model.eval()
        total_loss   = 0.0
        si_sdr_vals: List[float] = []
        stoi_vals:   List[float] = []
        n_batches    = 0
        audio_logged = 0
        max_audio    = int(self.cfg.train.n_audio_samples)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                primary   = batch["primary"].to(self.device)
                reference = batch["reference"].to(self.device)
                clean     = batch["clean"].to(self.device)

                enhanced = self._forward(primary, reference)
                loss, _  = self.criterion(enhanced, clean)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    n_batches  += 1

                # Per-sample metrics
                si_sdr_vals.append(_batch_si_sdr(enhanced, clean))
                stoi_vals.append(_batch_stoi(enhanced, clean, self.sr))

                # Log audio to TensorBoard
                if log_audio and self.writer and audio_logged < max_audio:
                    idx = 0
                    self.writer.add_audio(
                        f"val/noisy_{audio_logged}",
                        primary[idx:idx+1].cpu(), global_step, sample_rate=self.sr,
                    )
                    self.writer.add_audio(
                        f"val/enhanced_{audio_logged}",
                        enhanced[idx:idx+1].cpu(), global_step, sample_rate=self.sr,
                    )
                    self.writer.add_audio(
                        f"val/clean_{audio_logged}",
                        clean[idx:idx+1].cpu(), global_step, sample_rate=self.sr,
                    )
                    audio_logged += 1

                    # Save audio files
                    if audio_dir is not None:
                        import soundfile as sf
                        audio_dir.mkdir(parents=True, exist_ok=True)
                        for name, sig in [
                            ("noisy", primary[idx]),
                            ("enhanced", enhanced[idx]),
                            ("clean", clean[idx]),
                        ]:
                            sf.write(
                                str(audio_dir / f"ep{epoch:03d}_{audio_logged-1:02d}_{name}.wav"),
                                sig.cpu().numpy(), self.sr,
                            )

        mean_loss  = total_loss / max(n_batches, 1)
        mean_sisdr = float(np.nanmean(si_sdr_vals)) if si_sdr_vals else float("nan")
        mean_stoi  = float(np.nanmean(stoi_vals))   if stoi_vals   else float("nan")

        if self.writer:
            self.writer.add_scalar("val/loss",    mean_loss,  global_step)
            self.writer.add_scalar("val/si_sdr",  mean_sisdr, global_step)
            self.writer.add_scalar("val/stoi",    mean_stoi,  global_step)

        logger.info(
            "[Epoch %3d] val_loss=%.4f  SI-SDR=%.2f dB  STOI=%.3f",
            epoch, mean_loss, mean_sisdr, mean_stoi,
        )
        return {"loss": mean_loss, "si_sdr": mean_sisdr, "stoi": mean_stoi}

    # ------------------------------------------------------------------
    def _forward(
        self, primary: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch model forward (handles LiteStreamNet which returns a tuple)."""
        out = self.model(primary, reference)
        if isinstance(out, tuple):
            return out[0]
        return out
