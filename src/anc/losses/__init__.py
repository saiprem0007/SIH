"""
ANC-SIH losses package.

Exports
-------
SISNRLoss              — Negative SI-SNR (minimise to maximise SI-SNR)
STFTLoss               — Single-resolution spectral convergence + log-mag L1
MultiResolutionSTFTLoss— Multi-resolution STFT loss
CombinedLoss           — alpha*SI-SNR + beta*MR-STFT + gamma*L1

Usage
-----
    from anc.losses import CombinedLoss
    criterion = CombinedLoss.from_config(cfg)
    loss, components = criterion(enhanced, clean)
"""
from anc.losses.si_snr import SISNRLoss
from anc.losses.stft_loss import STFTLoss, MultiResolutionSTFTLoss
from anc.losses.combined import CombinedLoss

__all__ = [
    "SISNRLoss",
    "STFTLoss",
    "MultiResolutionSTFTLoss",
    "CombinedLoss",
]
