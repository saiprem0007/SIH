"""
ANC-SIH models package.

Exports
-------
CausalConvTasNet  — Model A: dual-input causal TCN (base / lite variant)
CausalDCUNet      — Model B: complex U-Net on STFT domain
LiteStreamNet     — Model C: tiny causal Conv+GRU (<100k params)

Usage
-----
    from anc.models import build_model
    model = build_model(cfg)
"""
from __future__ import annotations

from anc.models.convtasnet_causal import CausalConvTasNet, StreamState
from anc.models.dcunet_causal import CausalDCUNet
from anc.models.lite_stream import LiteStreamNet

__all__ = [
    "CausalConvTasNet",
    "CausalDCUNet",
    "LiteStreamNet",
    "StreamState",
    "build_model",
]


def build_model(cfg) -> "torch.nn.Module":  # type: ignore[name-defined]
    """Build a model from a Config object.

    Reads cfg.model.name ('convtasnet' | 'dcunet' | 'lite_stream')
    and delegates to the appropriate factory classmethod.

    Args:
        cfg: Config object with cfg.model.* keys.

    Returns:
        An nn.Module instance ready for training or inference.

    Raises:
        ValueError: If cfg.model.name is not recognised.
    """
    name = str(cfg.model.name).lower()
    if name == "convtasnet":
        return CausalConvTasNet.from_config(cfg)
    elif name == "dcunet":
        return CausalDCUNet.from_config(cfg)
    elif name in ("lite_stream", "lite"):
        return LiteStreamNet.from_config(cfg)
    else:
        raise ValueError(
            f"Unknown model name '{name}'. "
            "Choose from: convtasnet, dcunet, lite_stream"
        )
