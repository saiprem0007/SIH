"""
Causal Dual-Input Conv-TasNet for real-time speech enhancement.

Architecture
------------
Encoder   : 1-D conv, kernel K=32 (2 ms), stride S=16, causal
Separator : Stacked dilated depthwise-separable 1-D conv blocks (TCN)
            with causal padding and cumulative LayerNorm (cLN)
Decoder   : Transposed conv (overlap-add)

Input channels
--------------
  primary   : (B, T) waveform from the primary microphone
  reference  : (B, T) waveform from the reference microphone (noise reference)

The two channels are fused inside the encoder:
  - "concat" (default): encode separately, concatenate → linear projection
  - "early_add"       : sum signals before encoding
  - "separate_encoder": two independent encoders, features concatenated

Streaming mode
--------------
`forward_stream(chunk, state)` processes one hop-sized chunk at a time,
carrying convolutional buffers between calls.  See StreamState for fields.

References
----------
Luo & Mesgarani (2019), "Conv-TasNet: Surpassing ideal T-F masking for
speech separation".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal Layer Normalisation (Cumulative Layer Norm)
# ---------------------------------------------------------------------------

class CumulativeLayerNorm(nn.Module):
    """Cumulative (causal) Layer Normalisation.

    Normalises over the feature dimension at each time step using only
    statistics accumulated up to and including that step.  This avoids
    any look-ahead and is safe for streaming inference.

    For training speed we compute the normalisation over the *feature*
    (channel) axis only, which is equivalent to ChannelWise normalisation
    and is strictly causal.
    """

    def __init__(self, n_features: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_features, 1))
        self.beta = nn.Parameter(torch.zeros(n_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            Normalised tensor (B, C, T)
        """
        # Channel-wise (instance) norm per time step — strictly causal
        # We normalise over C at each t independently.
        B, C, T = x.shape
        x_t = x.permute(0, 2, 1)          # (B, T, C)
        mean = x_t.mean(dim=-1, keepdim=True)
        var  = x_t.var(dim=-1, unbiased=False, keepdim=True)
        x_t = (x_t - mean) / (var + self.eps).sqrt()
        x_t = x_t.permute(0, 2, 1)        # (B, C, T)
        return self.gamma * x_t + self.beta

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}"


# ---------------------------------------------------------------------------
# TCN block (depthwise-separable with dilation)
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """Single causal dilated depthwise-separable conv block.

    Structure (per block):
      1-D conv (1×1 pointwise)  → PReLU → cLN
      → Depthwise conv (dilation, causal) → PReLU → cLN
      → 1-D conv (1×1) for residual
      → 1-D conv (1×1) for skip
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        bottleneck_channels: int,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        # Causal padding = (kernel_size - 1) * dilation
        self._pad = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            # Pointwise
            nn.Conv1d(in_channels, bottleneck_channels, 1),
            nn.PReLU(),
            CumulativeLayerNorm(bottleneck_channels),
            # Depthwise (causal: we pad manually before the conv)
            nn.Conv1d(
                bottleneck_channels, bottleneck_channels,
                kernel_size, dilation=dilation,
                groups=bottleneck_channels, padding=0,
            ),
            nn.PReLU(),
            CumulativeLayerNorm(bottleneck_channels),
        )
        self.res_conv  = nn.Conv1d(bottleneck_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(bottleneck_channels, skip_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T)
        Returns:
            residual output (B, C, T), skip output (B, skip_channels, T)
        """
        # Causal pad: prepend zeros on the left
        x_pad = F.pad(x, (self._pad, 0))
        # Run pointwise and depthwise layers
        B, C, T_orig = x.shape
        # We do forward manually to apply padding between layers 2 and 3
        h = self.net[0](x)      # pointwise
        h = self.net[1](h)      # PReLU
        h = self.net[2](h)      # cLN
        h = F.pad(h, (self._pad, 0))
        h = self.net[3](h)      # depthwise (no internal padding)
        h = self.net[4](h)      # PReLU
        h = self.net[5](h)      # cLN
        # Trim to original T if shape mismatch
        if h.shape[-1] > T_orig:
            h = h[..., :T_orig]
        return x + self.res_conv(h), self.skip_conv(h)

    def forward_streaming(
        self, x: torch.Tensor, buf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Streaming forward: uses a rolling buffer instead of padding.

        Args:
            x:   (B, bottleneck_channels, chunk_T) — one chunk
            buf: (B, bottleneck_channels, _pad)    — left context buffer
        Returns:
            (residual_chunk, skip_chunk, new_buf)
        """
        # We need to run the full block in streaming; we operate on h
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        # Depthwise with buffer
        h_cat = torch.cat([buf, h], dim=-1)
        new_buf = h_cat[:, :, -self._pad:].detach()
        h = self.net[3](h_cat)  # depthwise, no padding needed
        h = self.net[4](h)
        h = self.net[5](h)
        T_chunk = x.shape[-1]
        if h.shape[-1] > T_chunk:
            h = h[..., :T_chunk]
        return x + self.res_conv(h), self.skip_conv(h), new_buf


# ---------------------------------------------------------------------------
# Streaming state
# ---------------------------------------------------------------------------

@dataclass
class StreamState:
    """Holds per-block buffers for causal streaming inference."""
    # Each element: list[list[Tensor]] — [stack][block] = buffer
    block_bufs: List[List[torch.Tensor]] = field(default_factory=list)
    # Encoder left-context buffer (for the causal conv encoder)
    encoder_buf: Optional[torch.Tensor] = None
    # Decoder overlap-add buffer  shape: (B, encoder_kernel - encoder_stride)
    decoder_buf: Optional[torch.Tensor] = None

    @staticmethod
    def zeros(
        n_stacks: int,
        n_blocks: int,
        bottleneck_channels: int,
        kernel_size: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        encoder_kernel: int = 32,
        encoder_stride: int = 16,
        enc_in_channels: int = 2,
    ) -> "StreamState":
        bufs: List[List[torch.Tensor]] = []
        for s in range(n_stacks):
            stack_bufs: List[torch.Tensor] = []
            for b in range(n_blocks):
                dilation = 2 ** b
                pad = (kernel_size - 1) * dilation
                stack_bufs.append(
                    torch.zeros(batch_size, bottleneck_channels, pad, device=device)
                )
            bufs.append(stack_bufs)
        # Overlap-add tail: (encoder_kernel - encoder_stride) samples
        dec_tail = encoder_kernel - encoder_stride
        decoder_buf = torch.zeros(batch_size, dec_tail, device=device)
        # Encoder left-context buffer (enc_in_channels: 2 for concat/separate, 1 for early_add)
        enc_buf = torch.zeros(batch_size, enc_in_channels, dec_tail, device=device)
        return StreamState(block_bufs=bufs, decoder_buf=decoder_buf, encoder_buf=enc_buf)


# ---------------------------------------------------------------------------
# Conv-TasNet (Causal, Dual-Input)
# ---------------------------------------------------------------------------

class CausalConvTasNet(nn.Module):
    """Causal dual-input Conv-TasNet speech enhancement model.

    Args:
        encoder_kernel  : length of the 1-D conv encoder kernel (samples)
        encoder_stride  : encoder stride (= hop size, samples)
        n_filters       : number of encoder output channels
        n_stacks        : number of TCN stacks
        n_blocks        : number of dilated conv blocks per stack
        tcn_channels    : main width of TCN channels
        bottleneck_channels: bottleneck (depthwise) channels
        kernel_size     : depthwise conv kernel size
        mask_activation : 'sigmoid' | 'relu'
        input_fusion    : 'concat' | 'early_add' | 'separate_encoder'
    """

    def __init__(
        self,
        encoder_kernel: int = 32,
        encoder_stride: int = 16,
        n_filters: int = 512,
        n_stacks: int = 2,
        n_blocks: int = 6,
        tcn_channels: int = 256,
        bottleneck_channels: int = 128,
        kernel_size: int = 3,
        mask_activation: str = "sigmoid",
        input_fusion: str = "concat",
    ) -> None:
        super().__init__()
        self.encoder_kernel = encoder_kernel
        self.encoder_stride = encoder_stride
        self.n_filters = n_filters
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.tcn_channels = tcn_channels
        self.bottleneck_channels = bottleneck_channels
        self.kernel_size = kernel_size
        self.input_fusion = input_fusion
        self._mask_act_name = mask_activation

        # ---- Encoder ----
        # Causal 1-D conv encoder: pad left by (kernel - stride) so the
        # output length equals ceil(T / stride).
        self._enc_pad = encoder_kernel - encoder_stride  # causal padding

        if input_fusion == "separate_encoder":
            self.encoder_primary   = nn.Conv1d(1, n_filters, encoder_kernel, encoder_stride, bias=False)
            self.encoder_reference = nn.Conv1d(1, n_filters, encoder_kernel, encoder_stride, bias=False)
            sep_in = n_filters * 2
        else:
            # "concat" or "early_add": single encoder, applied to 1 or 2 channels
            in_ch = 1 if input_fusion == "early_add" else 2
            self.encoder = nn.Conv1d(in_ch, n_filters, encoder_kernel, encoder_stride, bias=False)
            sep_in = n_filters

        # Bottleneck projection into TCN
        self.bottleneck_in = nn.Sequential(
            CumulativeLayerNorm(sep_in),
            nn.Conv1d(sep_in, tcn_channels, 1),
        )

        # ---- TCN Blocks ----
        self.tcn_blocks = nn.ModuleList()
        for s in range(n_stacks):
            for b in range(n_blocks):
                dilation = 2 ** b
                self.tcn_blocks.append(
                    TCNBlock(tcn_channels, tcn_channels, kernel_size, dilation, bottleneck_channels)
                )

        # ---- Mask estimation ----
        self.mask_head = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(tcn_channels, n_filters, 1),
        )
        if mask_activation == "sigmoid":
            self.mask_act = nn.Sigmoid()
        else:
            self.mask_act = nn.ReLU()

        # ---- Decoder ----
        self.decoder = nn.ConvTranspose1d(n_filters, 1, encoder_kernel, encoder_stride, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode(self, primary: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Run encoder fusion. Returns encoded features (B, n_filters, L)."""
        if self.input_fusion == "early_add":
            combined = primary + reference      # (B, 1, T)
            return F.relu(self.encoder(F.pad(combined, (self._enc_pad, 0))))
        elif self.input_fusion == "separate_encoder":
            ep = F.relu(self.encoder_primary(F.pad(primary,    (self._enc_pad, 0))))
            er = F.relu(self.encoder_reference(F.pad(reference, (self._enc_pad, 0))))
            return torch.cat([ep, er], dim=1)   # (B, 2*n_filters, L)
        else:  # concat
            combined = torch.cat([primary, reference], dim=1)  # (B, 2, T)
            return F.relu(self.encoder(F.pad(combined, (self._enc_pad, 0))))

    def _separate(self, features: torch.Tensor) -> torch.Tensor:
        """Run TCN separator, return skip-sum (B, tcn_channels, L)."""
        h = self.bottleneck_in(features)
        skip_sum: Optional[torch.Tensor] = None
        for block in self.tcn_blocks:
            h, skip = block(h)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        return skip_sum

    # ------------------------------------------------------------------
    # Offline (training / batch) forward
    # ------------------------------------------------------------------
    def forward(
        self,
        primary: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            primary  : (B, T) float32
            reference: (B, T) float32

        Returns:
            enhanced : (B, T) float32
        """
        T = primary.shape[-1]
        prim = primary.unsqueeze(1)      # (B, 1, T)
        ref  = reference.unsqueeze(1)

        encoded  = self._encode(prim, ref)       # (B, F, L)
        skip_sum = self._separate(encoded)        # (B, C, L)
        mask     = self.mask_act(self.mask_head(skip_sum))  # (B, F, L)

        # Apply mask to primary encoder features only
        if self.input_fusion == "separate_encoder":
            primary_enc = F.relu(
                self.encoder_primary(F.pad(prim, (self._enc_pad, 0)))
            )
        elif self.input_fusion == "early_add":
            primary_enc = F.relu(self.encoder(F.pad(prim, (self._enc_pad, 0))))
        else:
            primary_enc = encoded  # mask applied to concatenated features; take first half
            primary_enc = encoded[:, :self.n_filters, :]

        masked = mask * primary_enc              # (B, F, L)
        out    = self.decoder(masked)            # (B, 1, T')
        out    = out.squeeze(1)

        # Trim / pad to original length
        if out.shape[-1] > T:
            out = out[..., :T]
        elif out.shape[-1] < T:
            out = F.pad(out, (0, T - out.shape[-1]))
        return out

    # ------------------------------------------------------------------
    # Streaming forward
    # ------------------------------------------------------------------
    def init_stream_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> StreamState:
        """Return a zeroed StreamState for streaming inference."""
        dev = device or next(self.parameters()).device
        # Determine how many input channels the encoder buffer tracks
        enc_in_ch = 1 if self.input_fusion == "early_add" else 2
        return StreamState.zeros(
            self.n_stacks, self.n_blocks,
            self.bottleneck_channels, self.kernel_size,
            batch_size=batch_size, device=dev,
            encoder_kernel=self.encoder_kernel,
            encoder_stride=self.encoder_stride,
            enc_in_channels=enc_in_ch,
        )

    def forward_stream(
        self,
        primary_chunk: torch.Tensor,
        reference_chunk: torch.Tensor,
        state: StreamState,
    ) -> Tuple[torch.Tensor, StreamState]:
        """Process one chunk in streaming mode.

        The chunk must have the same length as encoder_stride samples so that
        exactly one encoded frame is produced.  Buffers in `state` carry the
        causal left-context between calls.

        Args:
            primary_chunk  : (B, T_chunk)  where T_chunk == encoder_stride
            reference_chunk: (B, T_chunk)
            state          : StreamState carrying buffers

        Returns:
            (enhanced_chunk (B, T_chunk), new_state)
        """
        T_chunk = primary_chunk.shape[-1]
        prim = primary_chunk.unsqueeze(1)   # (B, 1, T_chunk)
        ref  = reference_chunk.unsqueeze(1)

        # ---- Encoder with rolling left-context (replaces zero-pad in offline) ----
        # state.encoder_buf: (B, n_in_ch, _enc_pad) — left context of the encoder
        enc_buf = state.encoder_buf         # (B, n_in_ch, _enc_pad)

        if self.input_fusion == "early_add":
            combined = prim + ref            # (B, 1, T_chunk)
            windowed = torch.cat([enc_buf, combined], dim=-1)  # (B, 1, enc_pad+T_chunk)
            encoded  = F.relu(self.encoder(windowed))          # (B, n_filters, 1)
            # Re-encode just primary for masking
            p_windowed = torch.cat([enc_buf[:, :1, :], prim], dim=-1)
            primary_enc = F.relu(self.encoder(p_windowed))
            new_enc_buf = combined[:, :, -self._enc_pad:].detach()
        elif self.input_fusion == "separate_encoder":
            # primary
            p_windowed = torch.cat([enc_buf[:, 0:1, :], prim], dim=-1)
            ep = F.relu(self.encoder_primary(p_windowed))
            # reference
            r_windowed = torch.cat([enc_buf[:, 1:2, :], ref], dim=-1)
            er = F.relu(self.encoder_reference(r_windowed))
            encoded = torch.cat([ep, er], dim=1)   # (B, 2*n_filters, 1)
            primary_enc = ep
            new_enc_buf = torch.cat([prim[:, :, -self._enc_pad:],
                                     ref[:, :, -self._enc_pad:]], dim=1).detach()
        else:  # concat (default)
            combined = torch.cat([prim, ref], dim=1)  # (B, 2, T_chunk)
            windowed = torch.cat([enc_buf, combined], dim=-1)  # (B, 2, enc_pad+T_chunk)
            encoded  = F.relu(self.encoder(windowed))          # (B, n_filters, 1)
            primary_enc = encoded[:, :self.n_filters, :]
            new_enc_buf = combined[:, :, -self._enc_pad:].detach()

        # ---- TCN with per-block rolling depthwise buffers ----
        h = self.bottleneck_in(encoded)
        new_bufs: List[List[torch.Tensor]] = []
        skip_sum: Optional[torch.Tensor] = None
        block_idx = 0
        for s in range(self.n_stacks):
            stack_bufs: List[torch.Tensor] = []
            for b in range(self.n_blocks):
                block: TCNBlock = self.tcn_blocks[block_idx]
                buf = state.block_bufs[s][b]

                # Pointwise → PReLU → cLN (identical to offline)
                h_p = block.net[0](h)
                h_p = block.net[1](h_p)
                h_p = block.net[2](h_p)

                # Depthwise: rolling buffer instead of causal left-pad
                h_cat = torch.cat([buf, h_p], dim=-1)
                if block._pad > 0:
                    new_buf = h_cat[:, :, -block._pad:].detach()
                else:
                    new_buf = h_p.new_zeros(h_p.shape[0], h_p.shape[1], 0)
                h_dw = block.net[3](h_cat)
                h_dw = block.net[4](h_dw)
                h_dw = block.net[5](h_dw)
                if h_dw.shape[-1] > h.shape[-1]:
                    h_dw = h_dw[..., :h.shape[-1]]

                skip = block.skip_conv(h_dw)
                h    = h + block.res_conv(h_dw)
                skip_sum = skip if skip_sum is None else skip_sum + skip
                stack_bufs.append(new_buf)
                block_idx += 1
            new_bufs.append(stack_bufs)

        # ---- Mask + Decoder with overlap-add ----
        mask    = self.mask_act(self.mask_head(skip_sum))
        masked  = mask * primary_enc
        raw     = self.decoder(masked).squeeze(1)   # (B, encoder_kernel)

        # Overlap-add: prepend previous tail buffer, emit encoder_stride samples
        dec_buf  = state.decoder_buf                # (B, tail_len)
        tail_len = dec_buf.shape[-1]
        if tail_len > 0 and raw.shape[-1] >= tail_len:
            raw = raw.clone()
            raw[:, :tail_len] = raw[:, :tail_len] + dec_buf
        out = raw[:, :self.encoder_stride]          # emit stride samples
        if raw.shape[-1] > self.encoder_stride:
            new_dec_buf = raw[:, self.encoder_stride:].detach()
        else:
            new_dec_buf = dec_buf.new_zeros(dec_buf.shape)

        # Safety trim/pad
        if out.shape[-1] > T_chunk:
            out = out[..., :T_chunk]
        elif out.shape[-1] < T_chunk:
            out = F.pad(out, (0, T_chunk - out.shape[-1]))

        new_state = StreamState(
            block_bufs=new_bufs,
            encoder_buf=new_enc_buf,
            decoder_buf=new_dec_buf,
        )
        return out, new_state

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "CausalConvTasNet":
        """Build from a Config object (cfg.model.convtasnet.*)."""
        mc = cfg.model.convtasnet
        variant = getattr(cfg.model, "variant", "base")
        if variant == "lite":
            lc = mc.lite
            return cls(
                encoder_kernel=int(mc.encoder_kernel),
                encoder_stride=int(mc.encoder_stride),
                n_filters=int(lc.n_filters),
                n_stacks=int(lc.n_stacks),
                n_blocks=int(lc.n_blocks),
                tcn_channels=int(lc.tcn_channels),
                bottleneck_channels=int(lc.bottleneck_channels),
                kernel_size=int(mc.kernel_size),
                mask_activation=str(mc.mask_activation),
                input_fusion=str(cfg.model.input_fusion),
            )
        return cls(
            encoder_kernel=int(mc.encoder_kernel),
            encoder_stride=int(mc.encoder_stride),
            n_filters=int(mc.n_filters),
            n_stacks=int(mc.n_stacks),
            n_blocks=int(mc.n_blocks),
            tcn_channels=int(mc.tcn_channels),
            bottleneck_channels=int(mc.bottleneck_channels),
            kernel_size=int(mc.kernel_size),
            mask_activation=str(mc.mask_activation),
            input_fusion=str(cfg.model.input_fusion),
        )

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"CausalConvTasNet("
            f"fusion={self.input_fusion}, "
            f"stacks={self.n_stacks}×{self.n_blocks}, "
            f"channels={self.tcn_channels}, "
            f"params={self.count_params()/1e6:.2f}M)"
        )
