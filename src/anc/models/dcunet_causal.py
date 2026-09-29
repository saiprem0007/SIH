"""
Causal Complex U-Net (DCUNet) for dual-input speech enhancement.

Architecture
------------
Input  : STFT of primary + reference → stacked as 4 real channels
         (primary_real, primary_imag, reference_real, reference_imag)
Encoder: stack of causal 2-D complex conv layers (frequency × time),
         causal in the time axis (pad only past frames)
Bottleneck: optional bidirectional GRU replaced with causal GRU
Decoder: transposed complex conv (skip connections from encoder)
Output : complex ratio mask (cRM) applied to primary STFT → iSTFT

Causality: only the time axis is padded on the left (past only).
The frequency axis uses symmetric padding (non-causal in freq is fine
since we see the full frequency resolution within each frame).

Reference
---------
Choi et al. (2019) "Phase-aware Speech Enhancement with Deep Complex
U-Net." ICLR 2019.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Complex convolution primitives
# ---------------------------------------------------------------------------

class ComplexConv2d(nn.Module):
    """2-D convolution over complex-valued feature maps.

    A complex-valued map is stored as two real tensors of shape
    (B, C, F, T).  This layer applies:
        real_out = W_rr * real_in - W_ii * imag_in + b_r
        imag_out = W_ri * real_in + W_ir * imag_in + b_i
    which corresponds to complex matrix multiplication.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        bias: bool = True,
    ) -> None:
        super().__init__()
        # Real and imaginary weight matrices
        self.conv_rr = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_ri = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_ir = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_ii = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            real: (B, C, F, T)
            imag: (B, C, F, T)
        Returns:
            (real_out, imag_out): each (B, out_C, F', T')
        """
        real_out = self.conv_rr(real) - self.conv_ii(imag)
        imag_out = self.conv_ri(real) + self.conv_ir(imag)
        return real_out, imag_out


class ComplexConvTranspose2d(nn.Module):
    """Transposed 2-D complex convolution for the decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        output_padding: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        kw = dict(kernel_size=kernel_size, stride=stride, padding=padding,
                  output_padding=output_padding)
        self.conv_rr = nn.ConvTranspose2d(in_channels, out_channels, **kw)
        self.conv_ri = nn.ConvTranspose2d(in_channels, out_channels, **kw)
        self.conv_ir = nn.ConvTranspose2d(in_channels, out_channels, **kw)
        self.conv_ii = nn.ConvTranspose2d(in_channels, out_channels, **kw)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real_out = self.conv_rr(real) - self.conv_ii(imag)
        imag_out = self.conv_ri(real) + self.conv_ir(imag)
        return real_out, imag_out


class ComplexBatchNorm2d(nn.Module):
    """Simplified complex batch norm (independent BN on real and imag)."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bn_r(real), self.bn_i(imag)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class CausalComplexEncoderBlock(nn.Module):
    """Causal encoder block: complex conv → complex BN → ELU.

    Causal: pad only past time steps (left pad along the time axis).
    Frequency axis uses symmetric padding.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_t: int = 3,
        kernel_f: int = 3,
        stride_f: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_t = kernel_t
        self.kernel_f = kernel_f
        self.stride_f = stride_f
        self._causal_pad_t = kernel_t - 1   # left pad along time
        self._sym_pad_f    = (kernel_f - 1) // 2  # symmetric freq pad

        self.cconv = ComplexConv2d(
            in_ch, out_ch,
            kernel_size=(kernel_f, kernel_t),
            stride=(stride_f, 1),
            padding=(0, 0),   # we pad manually
        )
        self.cbn  = ComplexBatchNorm2d(out_ch)
        self.act  = nn.ELU()

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Causal time-axis pad (left only), symmetric freq pad
        real = F.pad(real, (self._causal_pad_t, 0, self._sym_pad_f, self._sym_pad_f))
        imag = F.pad(imag, (self._causal_pad_t, 0, self._sym_pad_f, self._sym_pad_f))
        real, imag = self.cconv(real, imag)
        real, imag = self.cbn(real, imag)
        return self.act(real), self.act(imag)


class CausalComplexDecoderBlock(nn.Module):
    """Decoder block: concat skip + complex transposed conv → complex BN → ELU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_t: int = 3,
        kernel_f: int = 3,
        stride_f: int = 2,
        output_pad_f: int = 0,
        is_last: bool = False,
    ) -> None:
        super().__init__()
        self.is_last = is_last
        self.ctconv = ComplexConvTranspose2d(
            in_ch * 2, out_ch,           # ×2 because of skip concat
            kernel_size=(kernel_f, kernel_t),
            stride=(stride_f, 1),
            padding=(kernel_f // 2, kernel_t - 1),  # freq sym + time causal
            output_padding=(output_pad_f, 0),
        )
        if not is_last:
            self.cbn = ComplexBatchNorm2d(out_ch)
            self.act = nn.ELU()

    def forward(
        self,
        real: torch.Tensor, imag: torch.Tensor,
        skip_r: torch.Tensor, skip_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real = torch.cat([real, skip_r], dim=1)
        imag = torch.cat([imag, skip_i], dim=1)
        real, imag = self.ctconv(real, imag)
        if not self.is_last:
            real, imag = self.cbn(real, imag)
            real, imag = self.act(real), self.act(imag)
        return real, imag


# ---------------------------------------------------------------------------
# DCU-Net
# ---------------------------------------------------------------------------

class CausalDCUNet(nn.Module):
    """Causal Deep Complex U-Net for dual-input speech enhancement.

    Args:
        n_fft             : FFT size
        hop_len           : STFT hop length
        win_len           : STFT window length
        encoder_channels  : list of channel widths for encoder layers
        kernel_t          : time-axis kernel size (causal)
        kernel_f          : frequency-axis kernel size
        use_gru           : whether to include a causal GRU bottleneck
        gru_hidden        : hidden size of the GRU
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_len: int = 160,
        win_len: int = 320,
        encoder_channels: List[int] = None,
        kernel_t: int = 3,
        kernel_f: int = 3,
        use_gru: bool = True,
        gru_hidden: int = 64,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]

        self.n_fft   = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.use_gru = use_gru
        self.gru_hidden = gru_hidden

        # Register window buffer
        window = torch.hann_window(win_len)
        self.register_buffer("window", window)

        n_freq = n_fft // 2 + 1   # 257 for n_fft=512

        # ---- Encoder ----
        # Input: 2 channels (primary + reference), each complex → 4 real
        enc_in = [2] + encoder_channels[:-1]
        enc_out = encoder_channels
        self.encoders = nn.ModuleList()
        for i, (ci, co) in enumerate(zip(enc_in, enc_out)):
            self.encoders.append(
                CausalComplexEncoderBlock(ci, co, kernel_t, kernel_f, stride_f=2)
            )

        # ---- Bottleneck GRU ----
        if use_gru:
            # Estimate frequency after all stride-2 encoders
            f_bot = n_freq // (2 ** len(encoder_channels))
            f_bot = max(f_bot, 1)
            self.gru = nn.GRU(
                input_size=encoder_channels[-1] * f_bot * 2,  # *2 for real+imag
                hidden_size=gru_hidden,
                num_layers=1, batch_first=True,
            )
            self.gru_proj = nn.Linear(gru_hidden, encoder_channels[-1] * f_bot * 2)
            self._f_bot = f_bot

        # ---- Decoder ----
        dec_in  = list(reversed(encoder_channels))
        dec_out = list(reversed(enc_in))
        self.decoders = nn.ModuleList()
        for i, (ci, co) in enumerate(zip(dec_in, dec_out)):
            is_last = (i == len(dec_in) - 1)
            self.decoders.append(
                CausalComplexDecoderBlock(
                    ci, co, kernel_t, kernel_f, stride_f=2,
                    is_last=is_last,
                )
            )

        # ---- Complex Ratio Mask (cRM) projection ----
        # Output: 2 channels (real + imag of mask)
        # Applied to primary STFT
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # STFT helpers
    # ------------------------------------------------------------------
    def _stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT. Returns (real, imag) each (B, F, T)."""
        B, T = x.shape
        X = torch.stft(
            x.reshape(-1, T),
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=True,
            center=False,
        )  # (B, F, T_stft)
        return X.real, X.imag

    def _istft(self, real: torch.Tensor, imag: torch.Tensor, length: int) -> torch.Tensor:
        """Inverse STFT from real/imag. Returns (B, T)."""
        X = torch.complex(real, imag)
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            length=length,
            center=False,
        )
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        primary: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            primary  : (B, T)
            reference: (B, T)
        Returns:
            enhanced : (B, T)
        """
        T = primary.shape[-1]

        # STFT of both channels
        pr, pi = self._stft(primary)    # (B, F, T_s)
        rr, ri = self._stft(reference)

        # Stack as (B, 2, F, T_s) real + imag separately
        real = torch.stack([pr, rr], dim=1)   # (B, 2, F, T_s)
        imag = torch.stack([pi, ri], dim=1)

        # Encode
        skips_r, skips_i = [], []
        for enc in self.encoders:
            skips_r.append(real)
            skips_i.append(imag)
            real, imag = enc(real, imag)

        # GRU bottleneck
        if self.use_gru:
            B, C, F, T_s = real.shape
            # Flatten freq+channels → time sequence
            r_seq = real.permute(0, 3, 1, 2).reshape(B, T_s, -1)
            i_seq = imag.permute(0, 3, 1, 2).reshape(B, T_s, -1)
            seq   = torch.cat([r_seq, i_seq], dim=-1)
            out, _ = self.gru(seq)
            out    = self.gru_proj(out)                        # (B, T_s, C*F*2)
            half   = out.shape[-1] // 2
            r_seq  = out[..., :half].reshape(B, T_s, C, F).permute(0, 2, 3, 1)
            i_seq  = out[..., half:].reshape(B, T_s, C, F).permute(0, 2, 3, 1)
            real   = r_seq
            imag   = i_seq

        # Decode with skip connections
        for i, dec in enumerate(self.decoders):
            skip_r = skips_r[-(i + 1)]
            skip_i = skips_i[-(i + 1)]
            # Trim/pad to match skip shape in time
            if real.shape[-1] != skip_r.shape[-1]:
                tgt = skip_r.shape[-1]
                if real.shape[-1] > tgt:
                    real = real[..., :tgt]
                    imag = imag[..., :tgt]
                else:
                    real = F.pad(real, (0, tgt - real.shape[-1]))
                    imag = F.pad(imag, (0, tgt - imag.shape[-1]))
            if real.shape[2] != skip_r.shape[2]:
                fmin = min(real.shape[2], skip_r.shape[2])
                real = real[:, :, :fmin, :]
                imag = imag[:, :, :fmin, :]
                skip_r = skip_r[:, :, :fmin, :]
                skip_i = skip_i[:, :, :fmin, :]
            real, imag = dec(real, imag, skip_r, skip_i)

        # real, imag: (B, 2, F, T_s) — treat as cRM mask for primary
        # Tanh-bounded complex ratio mask
        mask_r = torch.tanh(real[:, 0, :, :])   # (B, F, T_s)
        mask_i = torch.tanh(imag[:, 0, :, :])

        # Apply mask to primary STFT
        enhanced_r = pr * mask_r - pi * mask_i
        enhanced_i = pr * mask_i + pi * mask_r

        enhanced = self._istft(enhanced_r, enhanced_i, length=T)
        if enhanced.shape[-1] > T:
            enhanced = enhanced[..., :T]
        elif enhanced.shape[-1] < T:
            enhanced = F.pad(enhanced, (0, T - enhanced.shape[-1]))
        return enhanced

    @classmethod
    def from_config(cls, cfg) -> "CausalDCUNet":
        mc = cfg.model.dcunet
        return cls(
            n_fft=int(mc.n_fft),
            hop_len=int(mc.hop_len),
            win_len=int(mc.win_len),
            encoder_channels=list(mc.encoder_channels),
            kernel_t=int(mc.kernel_t),
            kernel_f=int(mc.kernel_f),
            use_gru=bool(mc.use_gru),
            gru_hidden=int(mc.gru_hidden),
        )

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"CausalDCUNet(n_fft={self.n_fft}, "
            f"gru={self.use_gru}, "
            f"params={self.count_params()/1e6:.2f}M)"
        )
