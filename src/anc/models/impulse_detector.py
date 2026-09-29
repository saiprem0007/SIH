"""
Impulse Detector -- two variants for the ANC-SIH hybrid pipeline.

v1  Rule-based (RuleImpulseDetector)
    Combines four causal features per frame:
      * Frame energy ratio  (frame energy / adaptive background energy)
      * Peak-to-RMS ratio   within the frame
      * Spectral flux        (L1 change of power spectrum between frames)
      * Rise-time            (how fast the envelope increases)
    A weighted sum is thresholded.
    Adaptive background tracker uses a slow exponential average updated
    only when no impulse is flagged (ETSI-style logic).
    A hangover counter prevents chattering on impulse tails.

v2  Learned (LearnedImpulseDetector, < 20 k params)
    Per-frame feature extractor (learnable filter bank) followed by
    a causal 1-D conv stack and a 1-layer GRU (hidden=32).
    Trained on frame-level impulse_labels from the MixingEngine.

Both expose the same interface:
    detector.reset()
    flag, score = detector.process_frame(frame)   # frame: np.ndarray (hop_len,)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ImpulseDetectorConfig:
    # Shared
    sample_rate: int = 16000
    frame_len: int = 320          # 20 ms
    hop_len: int = 160            # 10 ms
    n_fft: int = 512

    # --- Rule-based (v1) ---
    w_energy: float = 0.35        # energy-ratio weight
    w_peak_rms: float = 0.25      # peak-to-RMS weight
    w_flux: float = 0.20          # spectral-flux weight
    w_rise: float = 0.20          # rise-time weight

    threshold: float = 0.55
    hangover_frames: int = 4
    bg_slow_alpha: float = 0.995  # only updated when quiet
    min_energy_db: float = -80.0

    # Normalisation constants that map typical impulse scores to 1
    energy_ratio_ref_db: float = 12.0
    peak_rms_ref_db: float = 15.0
    flux_ref: float = 0.5
    rise_ref_db: float = 10.0

    # --- Learned (v2) ---
    cnn_channels: int = 16
    gru_hidden: int = 32


# ---------------------------------------------------------------------------
# Rule-based v1
# ---------------------------------------------------------------------------

class RuleImpulseDetector:
    """Causal rule-based impulse detector with adaptive background tracking."""

    def __init__(self, cfg: Optional[ImpulseDetectorConfig] = None) -> None:
        self.cfg = cfg or ImpulseDetectorConfig()
        self._reset_state()

    def reset(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._bg_energy: float = 1e-8
        self._prev_spectrum: Optional[np.ndarray] = None
        self._prev_frame_energy: float = 1e-8
        self._hangover: int = 0

    # ------------------------------------------------------------------
    def _safe_db(self, x: float) -> float:
        return 20.0 * np.log10(max(float(x), 10 ** (self.cfg.min_energy_db / 20.0)))

    def _energy_score(self, fe: float) -> float:
        ratio_db = self._safe_db(fe) - self._safe_db(self._bg_energy)
        return float(np.clip(ratio_db / self.cfg.energy_ratio_ref_db, 0.0, 1.0))

    def _peak_rms_score(self, frame: np.ndarray) -> float:
        rms  = float(np.sqrt(np.mean(frame ** 2)) + 1e-12)
        peak = float(np.abs(frame).max() + 1e-12)
        return float(np.clip(self._safe_db(peak / rms) / self.cfg.peak_rms_ref_db, 0.0, 1.0))

    def _flux_score(self, frame: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(frame, n=self.cfg.n_fft)) ** 2
        if self._prev_spectrum is None:
            self._prev_spectrum = spec
            return 0.0
        flux = float(np.mean(np.abs(spec - self._prev_spectrum)) /
                     (float(np.mean(self._prev_spectrum)) + 1e-12))
        self._prev_spectrum = spec
        return float(np.clip(flux / self.cfg.flux_ref, 0.0, 1.0))

    def _rise_score(self, fe: float) -> float:
        rise_db = self._safe_db(fe) - self._safe_db(self._prev_frame_energy)
        self._prev_frame_energy = fe
        return float(np.clip(rise_db / self.cfg.rise_ref_db, 0.0, 1.0))

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[int, float]:
        """Process one hop-length chunk.  Returns (flag:0/1, score:float)."""
        c = self.cfg
        frame = frame.astype(np.float64)
        fe = float(np.mean(frame ** 2) + 1e-12)

        score = (c.w_energy  * self._energy_score(fe)          +
                 c.w_peak_rms * self._peak_rms_score(frame.astype(np.float32)) +
                 c.w_flux     * self._flux_score(frame.astype(np.float32))     +
                 c.w_rise     * self._rise_score(fe))

        if score >= c.threshold:
            self._hangover = c.hangover_frames
            flag = 1
        elif self._hangover > 0:
            self._hangover -= 1
            flag = 1
        else:
            flag = 0

        if flag == 0:
            self._bg_energy = (c.bg_slow_alpha * self._bg_energy +
                               (1.0 - c.bg_slow_alpha) * fe)
        return flag, float(score)

    def process_signal(
        self, signal: np.ndarray, hop_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process full signal.  Returns (flags int8, scores float32)."""
        hop = hop_len or self.cfg.hop_len
        self.reset()
        n = len(signal) // hop
        flags  = np.zeros(n, dtype=np.int8)
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            flags[i], scores[i] = self.process_frame(
                signal[i * hop: (i + 1) * hop])
        return flags, scores


# ---------------------------------------------------------------------------
# Learned v2  (< 20 k params)
# ---------------------------------------------------------------------------

class _FrameFeatures(nn.Module):
    """Learnable per-frame feature extractor: (B,hop) -> (B,16)."""
    def __init__(self, hop_len: int = 160) -> None:
        super().__init__()
        self.hop_len = hop_len
        # 8 learnable filters over the raw waveform
        self.fbank = nn.Conv1d(1, 8, kernel_size=max(hop_len // 4, 8),
                               stride=max(hop_len // 8, 4), padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, hop_len)
        h = x.unsqueeze(1)                          # (B, 1, hop)
        if h.shape[-1] < self.fbank.kernel_size[0]:
            h = F.pad(h, (0, self.fbank.kernel_size[0] - h.shape[-1]))
        y = F.relu(self.fbank(h))                   # (B, 8, T')
        return torch.cat([y.mean(-1), y.max(-1).values], dim=-1)  # (B, 16)


class LearnedImpulseDetector(nn.Module):
    """Tiny causal CNN+GRU impulse detector (< 20 k params).

    Streaming interface::
        det.reset()
        flag, prob = det.process_frame(frame_np)  # frame_np: (hop_len,)

    Training::
        from anc.models.impulse_detector import train_detector
        losses = train_detector(det, signals, labels)
    """

    def __init__(self, cfg: Optional[ImpulseDetectorConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ImpulseDetectorConfig()
        c = self.cfg

        self._feat = _FrameFeatures(hop_len=c.hop_len)

        # Causal 1-D conv (padding on left only to avoid future look-ahead)
        ch = c.cnn_channels
        self._conv = nn.Sequential(
            nn.Conv1d(16, ch, kernel_size=3, padding=2, bias=False),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=2, bias=False),
            nn.ReLU(),
        )
        self._gru  = nn.GRU(ch, c.gru_hidden, batch_first=True)
        self._head = nn.Linear(c.gru_hidden, 1)
        self._h: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset(self) -> None:
        self._h = None

    # ------------------------------------------------------------------
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Offline forward.

        Args:
            frames: (B, T, hop_len) sequence of waveform frames

        Returns:
            logits: (B, T)
        """
        B, T, H = frames.shape
        feats = self._feat(frames.reshape(B * T, H)).reshape(B, T, 16)  # (B,T,16)
        conv_in  = feats.permute(0, 2, 1)           # (B, 16, T)
        conv_out = self._conv(conv_in)[:, :, :T]    # (B, ch, T) -- causal trim
        gru_in   = conv_out.permute(0, 2, 1)        # (B, T, ch)
        out, _   = self._gru(gru_in)                # (B, T, gru_h)
        return self._head(out).squeeze(-1)           # (B, T)

    def process_frame(self, frame: np.ndarray) -> Tuple[int, float]:
        """Streaming: one hop-length frame -> (flag:0/1, prob:float)."""
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)  # (1,H)
            feat = self._feat(t).unsqueeze(1)        # (1,1,16)
            ci   = feat.permute(0, 2, 1)             # (1,16,1)
            co   = self._conv(ci)[:, :, :1]          # (1,ch,1)
            gi   = co.permute(0, 2, 1)               # (1,1,ch)
            if self._h is not None:
                out, self._h = self._gru(gi, self._h)
            else:
                out, self._h = self._gru(gi)
            prob = float(torch.sigmoid(self._head(out.squeeze(1))).item())
        return int(prob >= self.cfg.threshold), prob

    def process_signal(
        self, signal: np.ndarray, hop_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        hop = hop_len or self.cfg.hop_len
        self.reset()
        n = len(signal) // hop
        flags  = np.zeros(n, dtype=np.int8)
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            flags[i], scores[i] = self.process_frame(
                signal[i * hop: (i + 1) * hop])
        return flags, scores


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_detector(
    model: LearnedImpulseDetector,
    signals: List[np.ndarray],
    labels: List[np.ndarray],
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """Train LearnedImpulseDetector on (signal, label) pairs.

    Args:
        signals : list of float32 1-D waveform arrays
        labels  : list of int8 frame-level impulse labels (from MixingEngine)
    Returns:
        epoch_losses
    """
    hop = model.cfg.hop_len
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    losses: List[float] = []

    # Build tensors
    Fs: List[torch.Tensor] = []
    Ls: List[torch.Tensor] = []
    for sig, lab in zip(signals, labels):
        n = min(len(sig) // hop, len(lab))
        buf = torch.zeros(1, n, hop)
        for i in range(n):
            c = sig[i * hop: (i + 1) * hop].astype(np.float32)
            buf[0, i, :len(c)] = torch.from_numpy(c)
        Fs.append(buf.to(device))
        Ls.append(torch.tensor(lab[:n], dtype=torch.float32).unsqueeze(0).to(device))

    for ep in range(epochs):
        total = 0.0
        for F_t, L_t in zip(Fs, Ls):
            opt.zero_grad()
            loss = crit(model(F_t), L_t)
            loss.backward()
            opt.step()
            total += loss.item()
        mean = total / max(len(Fs), 1)
        losses.append(mean)
        logger.debug("epoch %d/%d  bce=%.4f", ep + 1, epochs, mean)
    return losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_detector(
    detector,
    signals: List[np.ndarray],
    labels: List[np.ndarray],
    hop_len: int = 160,
) -> Dict[str, float]:
    """Compute precision, recall, F1, and mean detection latency (frames).

    Latency is the number of frames between the first true impulse frame
    of an event and the first positive prediction within that event.
    """
    tp = fp = fn = 0
    latencies: List[float] = []

    for sig, lab in zip(signals, labels):
        preds, _ = detector.process_signal(sig, hop_len)
        n = min(len(preds), len(lab))
        p = preds[:n].astype(int)
        l = lab[:n].astype(int)

        tp += int(np.sum((p == 1) & (l == 1)))
        fp += int(np.sum((p == 1) & (l == 0)))
        fn += int(np.sum((p == 0) & (l == 1)))

        # Per-event latency
        in_event = False
        onset = 0
        for t in range(n):
            if l[t] == 1 and not in_event:
                in_event = True
                onset = t
            if in_event and p[t] == 1:
                latencies.append(float(t - onset))
                in_event = False
            if l[t] == 0:
                in_event = False

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "precision":           round(prec, 4),
        "recall":              round(rec, 4),
        "f1":                  round(f1, 4),
        "mean_latency_frames": round(float(np.mean(latencies)) if latencies else float("nan"), 2),
        "tp": tp, "fp": fp, "fn": fn,
    }
