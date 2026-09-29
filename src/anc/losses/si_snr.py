"""
SI-SNR (Scale-Invariant Signal-to-Noise Ratio) loss.

SI-SNR = 10 * log10 ( ||alpha * s||^2 / ||alpha * s - e||^2 )

where:
  s     = clean reference signal (zero-mean)
  e     = estimated (enhanced) signal (zero-mean)
  alpha = <e, s> / <s, s>   (projection coefficient)

We maximise SI-SNR, so the loss is its negative.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SISNRLoss(nn.Module):
    """Negative Scale-Invariant SNR loss (minimise to maximise SI-SNR).

    Args:
        eps: small constant for numerical stability (default 1e-8)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            enhanced : (B, T) predicted clean speech
            clean    : (B, T) ground-truth clean speech
        Returns:
            scalar loss (mean over batch)
        """
        if enhanced.shape != clean.shape:
            raise ValueError(
                f"Shape mismatch: enhanced {enhanced.shape} vs clean {clean.shape}"
            )
        # Zero-mean
        e = enhanced - enhanced.mean(dim=-1, keepdim=True)
        s = clean    - clean.mean(dim=-1, keepdim=True)

        # Projection
        dot    = (e * s).sum(dim=-1, keepdim=True)            # (B, 1)
        s_norm = (s * s).sum(dim=-1, keepdim=True) + self.eps  # (B, 1)
        s_proj = dot / s_norm * s                              # (B, T)

        # Noise component
        noise = e - s_proj

        si_snr = 10 * torch.log10(
            (s_proj * s_proj).sum(dim=-1) /
            ((noise * noise).sum(dim=-1) + self.eps)
        )  # (B,)
        return -si_snr.mean()

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
