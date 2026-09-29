"""
Combined loss for speech enhancement training.

    L = alpha * L_si_snr  +  beta * L_mr_stft  +  gamma * L_l1

where:
  L_si_snr  = negative SI-SNR (anc.losses.si_snr)
  L_mr_stft = multi-resolution STFT loss (anc.losses.stft_loss)
  L_l1      = L1 waveform loss
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from anc.losses.si_snr import SISNRLoss
from anc.losses.stft_loss import MultiResolutionSTFTLoss


class CombinedLoss(nn.Module):
    """Weighted combination of SI-SNR, multi-resolution STFT, and L1 losses.

    Args:
        alpha       : weight for SI-SNR loss     (default 0.5)
        beta        : weight for MR-STFT loss    (default 0.3)
        gamma       : weight for L1 waveform loss (default 0.2)
        stft_resolutions: list of (n_fft, hop, win) for MR-STFT
        eps         : floor for log spectrogram
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        stft_resolutions: Optional[List[Tuple[int, int, int]]] = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.si_snr_loss = SISNRLoss(eps=eps)
        self.stft_loss   = MultiResolutionSTFTLoss(resolutions=stft_resolutions, eps=eps)

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            enhanced : (B, T) predicted clean speech
            clean    : (B, T) ground-truth clean speech

        Returns:
            (total_loss, component_dict)
            component_dict keys: si_snr, stft, l1, total
        """
        # Guard: NaN / Inf in inputs
        if torch.isnan(enhanced).any() or torch.isinf(enhanced).any():
            # Return zero loss; will be caught by training loop
            zero = enhanced.new_zeros(1).squeeze()
            return zero, {"si_snr": zero, "stft": zero, "l1": zero, "total": zero}

        l_si_snr = self.si_snr_loss(enhanced, clean)
        l_stft   = self.stft_loss(enhanced, clean)
        l_l1     = F.l1_loss(enhanced, clean)

        total = self.alpha * l_si_snr + self.beta * l_stft + self.gamma * l_l1

        return total, {
            "si_snr": l_si_snr.detach(),
            "stft":   l_stft.detach(),
            "l1":     l_l1.detach(),
            "total":  total.detach(),
        }

    @classmethod
    def from_config(cls, cfg) -> "CombinedLoss":
        """Build from cfg.loss.*"""
        resolutions = None
        if hasattr(cfg.loss, "stft_resolutions"):
            resolutions = [tuple(r) for r in cfg.loss.stft_resolutions]
        return cls(
            alpha=float(cfg.loss.alpha),
            beta=float(cfg.loss.beta),
            gamma=float(cfg.loss.gamma),
            stft_resolutions=resolutions,
        )

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}"
