"""
LiteStream: Tiny causal Conv + GRU model for MCU/DSP fallback.

Target: < 100 k parameters, INT8-quantisable, runs on Cortex-M/DSP.

Architecture
------------
1. 1-D conv encoder (causal, kernel 32, stride 16) applied to primary
   and reference — parameters shared, features concatenated.
2. Two 1×1 conv layers to project to a narrow feature space.
3. Single-layer causal GRU to model temporal context.
4. FC head → mask → multiply with encoder features → 1-D conv decoder.

Streaming
---------
`forward_stream(chunk, state)` accepts (B, chunk) and a state dict
{"gru_h": tensor}.  Returns (enhanced_chunk, new_state).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteStreamNet(nn.Module):
    """Tiny causal Conv+GRU speech enhancement model (< 100 k params).

    Args:
        encoder_kernel   : 1-D conv encoder kernel size (samples)
        encoder_stride   : encoder stride (= frame hop, samples)
        n_filters        : encoder output channels (applied to both channels)
        gru_hidden       : GRU hidden state size
        gru_layers       : number of stacked GRU layers (1 for MCU target)
        fc_channels      : FC head width
    """

    def __init__(
        self,
        encoder_kernel: int = 32,
        encoder_stride: int = 16,
        n_filters: int = 64,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        fc_channels: int = 64,
    ) -> None:
        super().__init__()
        self.encoder_kernel = encoder_kernel
        self.encoder_stride = encoder_stride
        self.n_filters = n_filters
        self.gru_hidden = gru_hidden
        self._enc_pad = encoder_kernel - encoder_stride

        # Shared encoder (applied to each channel separately)
        self.encoder = nn.Conv1d(1, n_filters, encoder_kernel, encoder_stride, bias=False)

        # Fusion projection: 2*n_filters → fc_channels
        self.proj_in = nn.Sequential(
            nn.LayerNorm(n_filters * 2),
            nn.Linear(n_filters * 2, fc_channels),
            nn.ReLU(),
        )

        # Causal GRU
        self.gru = nn.GRU(
            input_size=fc_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )

        # Mask head: GRU hidden → n_filters mask
        self.mask_head = nn.Sequential(
            nn.Linear(gru_hidden, fc_channels),
            nn.ReLU(),
            nn.Linear(fc_channels, n_filters),
            nn.Sigmoid(),
        )

        # Decoder
        self.decoder = nn.ConvTranspose1d(n_filters, 1, encoder_kernel, encoder_stride, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _encode_both(
        self, primary: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Encode both channels, return concatenated features (B, 2*F, L)."""
        ep = F.relu(self.encoder(F.pad(primary,   (self._enc_pad, 0))))
        er = F.relu(self.encoder(F.pad(reference, (self._enc_pad, 0))))
        return torch.cat([ep, er], dim=1)   # (B, 2F, L)

    def forward(
        self,
        primary: torch.Tensor,
        reference: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            primary  : (B, T)
            reference: (B, T)
            hx       : optional initial GRU hidden state
        Returns:
            (enhanced (B, T), final_hidden)
        """
        T = primary.shape[-1]
        prim = primary.unsqueeze(1)
        ref  = reference.unsqueeze(1)

        # Encode
        feat = self._encode_both(prim, ref)     # (B, 2F, L)
        B, _, L = feat.shape

        # Time-major projection
        x = feat.permute(0, 2, 1)              # (B, L, 2F)
        x = self.proj_in(x)                    # (B, L, fc_channels)

        # GRU
        gru_out, h_n = self.gru(x, hx)        # (B, L, gru_hidden)

        # Mask
        mask = self.mask_head(gru_out)         # (B, L, n_filters)
        mask = mask.permute(0, 2, 1)           # (B, n_filters, L)

        # Apply to primary encoder features
        primary_enc = feat[:, :self.n_filters, :]
        masked = mask * primary_enc

        # Decode
        out = self.decoder(masked).squeeze(1)  # (B, T')
        if out.shape[-1] > T:
            out = out[..., :T]
        elif out.shape[-1] < T:
            out = F.pad(out, (0, T - out.shape[-1]))
        return out, h_n

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------
    def init_stream_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        dev = device or next(self.parameters()).device
        return {
            "gru_h": torch.zeros(
                self.gru.num_layers, batch_size, self.gru_hidden, device=dev
            )
        }

    def forward_stream(
        self,
        primary_chunk: torch.Tensor,
        reference_chunk: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process one chunk in streaming mode.

        Args:
            primary_chunk   : (B, T_chunk)
            reference_chunk : (B, T_chunk)
            state           : {"gru_h": (n_layers, B, H)}
        Returns:
            (enhanced_chunk, new_state)
        """
        enhanced, h_n = self.forward(primary_chunk, reference_chunk, hx=state["gru_h"])
        return enhanced, {"gru_h": h_n}

    @classmethod
    def from_config(cls, cfg) -> "LiteStreamNet":
        mc = cfg.model.lite_stream
        return cls(
            encoder_kernel=int(mc.encoder_kernel),
            encoder_stride=int(mc.encoder_stride),
            n_filters=int(mc.n_filters),
            gru_hidden=int(mc.gru_hidden),
            gru_layers=int(mc.gru_layers),
            fc_channels=int(mc.fc_channels),
        )

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"LiteStreamNet("
            f"filters={self.n_filters}, gru={self.gru_hidden}, "
            f"params={self.count_params()/1e3:.1f}k)"
        )
