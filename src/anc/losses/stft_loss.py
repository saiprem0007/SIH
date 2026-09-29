"""
Multi-Resolution STFT Loss.

Combines:
  (1) Spectral convergence loss  = ||mag_clean - mag_enh||_F / ||mag_clean||_F
  (2) Log-magnitude L1 loss      = mean |log(mag_clean + eps) - log(mag_enh + eps)|

Evaluated at multiple STFT resolutions for robustness across time-frequency scales.

Reference
---------
Yamamoto et al. (2020) "Parallel WaveGAN: A fast waveform generation model based
on generative adversarial networks with multi-resolution spectrogram."
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft_magnitude(
    x: torch.Tensor,
    n_fft: int,
    hop_len: int,
    win_len: int,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute STFT magnitude spectrogram.

    Args:
        x      : (B, T) waveform
        n_fft  : FFT size
        hop_len: hop length
        win_len: window length
        window : pre-built window tensor (created if None)

    Returns:
        magnitude: (B, F, T_stft)
    """
    B, T = x.shape
    if window is None:
        window = torch.hann_window(win_len, device=x.device)
    elif window.device != x.device:
        window = window.to(x.device)

    X = torch.stft(
        x.reshape(B, T),
        n_fft=n_fft,
        hop_length=hop_len,
        win_length=win_len,
        window=window,
        return_complex=True,
        center=False,
        pad_mode="reflect",
    )
    return X.abs()  # (B, F, T_stft)


def spectral_convergence_loss(
    mag_clean: torch.Tensor, mag_enh: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Frobenius-norm spectral convergence."""
    diff = mag_clean - mag_enh
    return torch.norm(diff, p="fro") / (torch.norm(mag_clean, p="fro") + eps)


def log_magnitude_l1_loss(
    mag_clean: torch.Tensor, mag_enh: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """L1 loss on log magnitude spectra."""
    log_clean = torch.log(mag_clean + eps)
    log_enh   = torch.log(mag_enh   + eps)
    return F.l1_loss(log_enh, log_clean)


class STFTLoss(nn.Module):
    """Single-resolution STFT loss (spectral convergence + log-mag L1).

    Args:
        n_fft  : FFT size
        hop_len: hop length
        win_len: analysis window length
        eps    : floor for log-magnitude (default 1e-7)
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_len: int = 128,
        win_len: int = 512,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.n_fft   = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.eps     = eps
        self.register_buffer("window", torch.hann_window(win_len))

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        mag_e = _stft_magnitude(enhanced, self.n_fft, self.hop_len, self.win_len, self.window)
        mag_c = _stft_magnitude(clean,    self.n_fft, self.hop_len, self.win_len, self.window)
        sc    = spectral_convergence_loss(mag_c, mag_e, self.eps)
        lm    = log_magnitude_l1_loss(mag_c, mag_e, self.eps)
        return sc + lm

    def extra_repr(self) -> str:
        return f"n_fft={self.n_fft}, hop={self.hop_len}, win={self.win_len}"


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss: average over multiple (n_fft, hop, win) triples.

    Args:
        resolutions: list of (n_fft, hop_len, win_len) tuples.
                     Defaults to three scales if None.
        eps        : floor for log-magnitude
    """

    def __init__(
        self,
        resolutions: Optional[List[Tuple[int, int, int]]] = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if resolutions is None:
            resolutions = [(512, 128, 512), (1024, 256, 1024), (256, 64, 256)]
        self.stft_losses = nn.ModuleList(
            [STFTLoss(n, h, w, eps) for n, h, w in resolutions]
        )

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        total = sum(loss(enhanced, clean) for loss in self.stft_losses)
        return total / len(self.stft_losses)
