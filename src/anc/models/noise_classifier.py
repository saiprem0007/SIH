"""
Noise-type Classifier -- tiny causal CNN on log-mel features of the reference channel.

Classes (7):  impulsive | rotor | vehicle | siren | wind | crowd | other

Architecture
------------
Log-mel spectrogram (64 bins, 20-ms frames, 10-ms hop, causal)
-> 3 x causal Conv2d blocks (time-causal padding, freq-stride pooling)
-> Flatten -> 1-layer causal GRU (hidden=32)
-> Linear -> softmax

Total parameters: < 50 k.

Two usage modes
---------------
(a) FiLM conditioning: embed the class distribution into the AI enhancement
    model as a scale+shift vector (see ``NoiseClassifier.film_params()``).
(b) Checkpoint selection: pick the specialist model checkpoint that best
    matches the dominant noise class.

Streaming interface
-------------------
    clf = NoiseClassifier()
    clf.reset()
    probs = clf.process_frame(ref_frame)   # (7,) softmax probs
    label = clf.predict_class(ref_frame)   # int class index

FiLM integration helper
-----------------------
    film_scale, film_bias = clf.film_params(ref_frame, embed_dim=256)
    # Apply: out = film_scale * features + film_bias
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class definitions matching NOISE_CATEGORIES in mixing.py
# ---------------------------------------------------------------------------

NOISE_CLASSES: List[str] = [
    "impulsive",
    "rotor",
    "vehicle",
    "siren",
    "wind",
    "crowd",
    "other",
]
NUM_CLASSES: int = len(NOISE_CLASSES)
CLASS_TO_IDX: dict = {c: i for i, c in enumerate(NOISE_CLASSES)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NoiseClassifierConfig:
    sample_rate: int = 16000
    hop_len: int = 160          # 10 ms
    n_fft: int = 512
    n_mels: int = 64
    # CNN
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32, 32])
    kernel_t: int = 3           # time kernel (odd, causal-padded)
    kernel_f: int = 3           # freq kernel
    pool_f: int = 2             # freq-axis max-pool stride
    # GRU
    gru_hidden: int = 32
    # Output
    n_classes: int = NUM_CLASSES
    # Context window for streaming (frames)
    context_frames: int = 8


# ---------------------------------------------------------------------------
# Mel filterbank helper (no librosa dependency)
# ---------------------------------------------------------------------------

def _make_mel_filterbank(n_fft: int, n_mels: int, sr: int) -> np.ndarray:
    """Return a (n_mels, n_fft//2+1) linear-to-mel weight matrix."""
    f_min, f_max = 0.0, float(sr) / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb      = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_lo, f_c, f_hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_lo, f_c):
            if f_c != f_lo:
                fb[m - 1, k] = (k - f_lo) / (f_c - f_lo)
        for k in range(f_c, f_hi):
            if f_hi != f_c:
                fb[m - 1, k] = (f_hi - k) / (f_hi - f_c)
    return fb


# ---------------------------------------------------------------------------
# Log-mel feature extractor (no trainable params)
# ---------------------------------------------------------------------------

class LogMelExtractor(nn.Module):
    """Causal log-mel spectrogram extractor (no learnable params).

    Input : (B, T_samples) reference-mic waveform segment
    Output: (B, 1, n_mels, T_frames) log-mel spectrogram
    """

    def __init__(self, cfg: NoiseClassifierConfig) -> None:
        super().__init__()
        self.hop_len = cfg.hop_len
        self.n_fft   = cfg.n_fft
        self.n_mels  = cfg.n_mels
        self.sr      = cfg.sample_rate

        fb = _make_mel_filterbank(cfg.n_fft, cfg.n_mels, cfg.sample_rate)
        self.register_buffer("mel_fb", torch.from_numpy(fb))  # (n_mels, n_fft/2+1)

        # Hann window
        win = torch.hann_window(cfg.n_fft)
        self.register_buffer("window", win)

        # Causal padding: pad left by n_fft-hop so that frame t uses only
        # samples up to and including t*hop + hop - 1
        self._pad = cfg.n_fft - cfg.hop_len

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)
        x = F.pad(waveform, (self._pad, 0))          # causal left-pad
        # STFT via unfold (avoids torch.stft centering issues in older PT)
        x = x.unsqueeze(1)                            # (B, 1, T+pad)
        frames = x.unfold(-1, self.n_fft, self.hop_len)  # (B, 1, T_fr, n_fft)
        frames = frames.squeeze(1) * self.window.unsqueeze(0).unsqueeze(0)
        spec  = torch.fft.rfft(frames, n=self.n_fft)  # (B, T_fr, n_fft/2+1)
        power = spec.real ** 2 + spec.imag ** 2       # (B, T_fr, n_fft/2+1)
        mel   = torch.matmul(power, self.mel_fb.T)    # (B, T_fr, n_mels)
        logmel = torch.log(mel.clamp(min=1e-9))       # (B, T_fr, n_mels)
        return logmel.permute(0, 2, 1).unsqueeze(1)   # (B, 1, n_mels, T_fr)


# ---------------------------------------------------------------------------
# Classifier backbone
# ---------------------------------------------------------------------------

class NoiseClassifier(nn.Module):
    """Tiny causal CNN + GRU noise-type classifier.

    Streaming interface::
        clf.reset()
        probs = clf.process_frame(ref_hop)     # np.ndarray (n_classes,)
        idx   = clf.predict_class(ref_hop)     # int
        scale, bias = clf.film_params(ref_hop, embed_dim=256)

    Offline (training)::
        logits = clf(reference_waveform)       # (B, T_frames, n_classes)
    """

    def __init__(self, cfg: Optional[NoiseClassifierConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or NoiseClassifierConfig()
        c = self.cfg

        self._mel = LogMelExtractor(c)

        # Build causal Conv2d stack
        # Causal time padding = kernel_t - 1 on the left; no freq padding
        chs = [1] + c.cnn_channels
        convs = []
        freq_dim = c.n_mels
        for i in range(len(c.cnn_channels)):
            convs.append(nn.Conv2d(
                chs[i], chs[i + 1],
                kernel_size=(c.kernel_f, c.kernel_t),
                padding=(c.kernel_f // 2, 0),          # freq: symmetric; time: none
                bias=False,
            ))
            convs.append(nn.BatchNorm2d(chs[i + 1]))
            convs.append(nn.ReLU())
            convs.append(nn.MaxPool2d(kernel_size=(c.pool_f, 1),
                                       stride=(c.pool_f, 1)))
            freq_dim = freq_dim // c.pool_f
        self._convs = nn.ModuleList(convs)
        self._n_causal_pad = c.kernel_t - 1  # left-pad per conv along time

        # After conv stack, flatten freq dim
        flat_dim = c.cnn_channels[-1] * max(freq_dim, 1)

        self._gru  = nn.GRU(flat_dim, c.gru_hidden, batch_first=True)
        self._head = nn.Linear(c.gru_hidden, c.n_classes)

        # FiLM projection
        self._film_scale = nn.Linear(c.n_classes, 256)
        self._film_bias  = nn.Linear(c.n_classes, 256)

        # Streaming state
        self._h: Optional[torch.Tensor] = None
        # Frame context buffer for streaming (log-mel cols)
        self._ctx: Optional[torch.Tensor] = None  # (1, 1, n_mels, ctx)

    # ------------------------------------------------------------------
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset(self) -> None:
        self._h = None
        self._ctx = None

    # ------------------------------------------------------------------
    def _apply_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T_fr)  ->  (B, flat_dim, T_fr)"""
        T = x.shape[-1]
        for layer in self._convs:
            if isinstance(layer, nn.Conv2d):
                # Causal time padding on the left
                x = F.pad(x, (self._n_causal_pad, 0))
                x = layer(x)
                x = x[..., :T]  # trim to original T (causal)
            else:
                x = layer(x)
        # Flatten freq and channel dims: (B, ch*freq, T)
        B, C, F_d, T_out = x.shape
        return x.reshape(B, C * F_d, T_out)

    # ------------------------------------------------------------------
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Offline forward.

        Args:
            waveform: (B, T_samples) reference-channel waveform

        Returns:
            logits: (B, T_frames, n_classes)
        """
        logmel = self._mel(waveform)          # (B, 1, n_mels, T_fr)
        feats  = self._apply_conv_stack(logmel)  # (B, flat, T_fr)
        gru_in = feats.permute(0, 2, 1)      # (B, T_fr, flat)
        out, _ = self._gru(gru_in)           # (B, T_fr, gru_h)
        return self._head(out)               # (B, T_fr, n_classes)

    # ------------------------------------------------------------------
    def process_frame(self, ref_frame: np.ndarray) -> np.ndarray:
        """Streaming: one hop-length reference frame -> softmax probs (n_classes,)."""
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(ref_frame.astype(np.float32)).unsqueeze(0)  # (1,H)
            logmel = self._mel(t)             # (1, 1, n_mels, T_fr)

            # Build/update streaming context window
            if self._ctx is None:
                self._ctx = logmel
            else:
                self._ctx = torch.cat([self._ctx, logmel], dim=-1)
                # Keep at most cfg.context_frames
                if self._ctx.shape[-1] > self.cfg.context_frames:
                    self._ctx = self._ctx[..., -self.cfg.context_frames:]

            feats  = self._apply_conv_stack(self._ctx)  # (1, flat, T)
            gru_in = feats.permute(0, 2, 1)             # (1, T, flat)
            if self._h is not None:
                out, self._h = self._gru(gru_in, self._h)
            else:
                out, self._h = self._gru(gru_in)
            # Take last time step
            logits = self._head(out[:, -1, :])          # (1, n_classes)
            probs  = F.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()            # (n_classes,)

    def predict_class(self, ref_frame: np.ndarray) -> int:
        """Return integer class index (see NOISE_CLASSES)."""
        return int(np.argmax(self.process_frame(ref_frame)))

    def predict_class_name(self, ref_frame: np.ndarray) -> str:
        return NOISE_CLASSES[self.predict_class(ref_frame)]

    def film_params(
        self,
        ref_frame: np.ndarray,
        embed_dim: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return FiLM scale and bias tensors for conditioning the AI model.

        Returns:
            scale: (1, embed_dim)
            bias : (1, embed_dim)
        """
        probs_np = self.process_frame(ref_frame)
        probs = torch.from_numpy(probs_np).unsqueeze(0)  # (1, n_classes)
        # Dynamically resize projections if embed_dim differs from default
        if self._film_scale.out_features != embed_dim:
            self._film_scale = nn.Linear(self.cfg.n_classes, embed_dim)
            self._film_bias  = nn.Linear(self.cfg.n_classes, embed_dim)
        with torch.no_grad():
            scale = torch.sigmoid(self._film_scale(probs))    # (1, embed_dim)
            bias  = self._film_bias(probs)                    # (1, embed_dim)
        return scale, bias


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_classifier(
    model: NoiseClassifier,
    waveforms: List[np.ndarray],
    class_indices: List[int],
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """Train the NoiseClassifier on (waveform, class_label) pairs.

    Args:
        waveforms     : list of float32 1-D reference-channel signals
        class_indices : list of int class labels (index into NOISE_CLASSES)

    Returns:
        epoch_losses
    """
    model.to(device).train()
    opt  = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses: List[float] = []

    Xs: List[torch.Tensor] = []
    Ys: List[torch.Tensor] = []
    for wav, cls in zip(waveforms, class_indices):
        Xs.append(torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device))
        Ys.append(torch.tensor(cls, dtype=torch.long).to(device))

    for ep in range(epochs):
        total = 0.0
        for x, y in zip(Xs, Ys):
            opt.zero_grad()
            logits = model(x)           # (1, T, n_classes)
            # Predict class from the last frame (sequence classification)
            loss = crit(logits[:, -1, :], y.unsqueeze(0))
            loss.backward()
            opt.step()
            total += loss.item()
        mean = total / max(len(Xs), 1)
        losses.append(mean)
        logger.debug("epoch %d/%d  ce=%.4f", ep + 1, epochs, mean)
    return losses
