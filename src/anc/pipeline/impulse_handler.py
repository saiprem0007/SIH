"""
Impulse Handler -- real-time burst suppression for the ANC-SIH pipeline.

When an impulse is flagged by the detector, this module:

1. Soft limiter / fast compressor on the burst region
   - Attack: very fast (1-2 ms), release: configurable (50-200 ms)
   - Threshold set relative to the running background level
   - Knee-width for smooth transition (soft-knee)

2. Transient spectral gating (burst duration only)
   - Short-window power spectrum suppression gated to the flagged frames
   - Wiener-style gain floor: does NOT hard-mute, preserves speech below the gate
   - Applied additively on top of the compressor (stacked protection)

3. Reference-mic burst cancellation (optional)
   - When a reference channel is available, the burst in reference is strongly
     correlated with the burst in primary.  A short-window normalised cross-
     correlation gives an alignment delay, and a single-tap scaled subtraction
     reduces the burst energy.
   - Strictly causal: only uses buffered history, no future look-ahead.

4. Cross-fade to avoid clicks
   - Attack/release transitions are windowed with a half-cosine fade,
     preventing audible click artefacts at burst boundaries.

5. Side-channel output: impulse_flag (int 0/1) per frame, available for:
   - Conditioning the AI enhancement model as an extra binary input feature
   - Diagnostic logging

Streaming interface (causal)::

    handler = ImpulseHandler(cfg)
    handler.reset()
    out_primary, flag = handler.process_frame(
        primary_frame, ref_frame, impulse_flag_from_detector)

Full-signal helper::

    out_signal, flags = handler.process_signal(primary, reference, det_flags)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EPS: float = 1e-10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ImpulseHandlerConfig:
    sample_rate: int = 16000
    hop_len: int = 160           # 10 ms

    # -- Compressor --
    comp_threshold_db: float = -20.0   # above background level
    comp_ratio: float = 10.0           # compression ratio (10:1)
    comp_knee_db: float = 6.0          # soft-knee width in dB
    comp_attack_ms: float = 1.0        # 1 ms (very fast)
    comp_release_ms: float = 80.0      # 80 ms

    # -- Spectral gate --
    gate_n_fft: int = 256
    gate_hop: int = 64             # 4 ms (sub-frame resolution)
    gate_threshold_db: float = -15.0   # above local noise floor
    gate_floor: float = 0.05          # minimum gain (Wiener floor), prevents muting speech

    # -- Reference cancellation --
    ref_cancel_enabled: bool = True
    ref_max_delay_samples: int = 32    # max search range for alignment
    ref_alpha: float = 0.95            # smoothing of cross-correlation estimate

    # -- Cross-fade --
    crossfade_ms: float = 2.0          # half-cosine fade length

    # -- Background tracker --
    bg_alpha: float = 0.99             # slow background energy tracker


# ---------------------------------------------------------------------------
# Impulse Handler
# ---------------------------------------------------------------------------

class ImpulseHandler:
    """Causal burst-suppression handler for the ANC-SIH streaming pipeline.

    On each frame call:
      - If flag==1: apply compressor + spectral gate (+ optional ref cancel)
      - If flag==0: pass-through (compressor gain tracks release)
    """

    def __init__(self, cfg: Optional[ImpulseHandlerConfig] = None) -> None:
        self.cfg = cfg or ImpulseHandlerConfig()
        self._reset_state()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        c = self.cfg
        sr = c.sample_rate
        # Compressor state
        self._gain_db: float = 0.0        # current gain (dB), negative = attenuation
        self._bg_energy: float = EPS      # adaptive background energy
        # Precompute time constants
        self._att_coef = float(np.exp(-1.0 / (c.comp_attack_ms  * sr / (1000.0 * c.hop_len))))
        self._rel_coef = float(np.exp(-1.0 / (c.comp_release_ms * sr / (1000.0 * c.hop_len))))
        # Reference cancellation state
        ref_buf = max(c.ref_max_delay_samples, c.hop_len)
        self._ref_buf = np.zeros(ref_buf + c.hop_len, dtype=np.float32)
        self._pri_buf = np.zeros(ref_buf + c.hop_len, dtype=np.float32)
        self._ref_scale: float = 0.0      # adaptive subtraction gain
        # Cross-fade
        self._fade_len = max(1, int(c.crossfade_ms * sr / 1000.0))
        self._fade_buf = np.zeros(self._fade_len, dtype=np.float32)
        self._prev_gain: float = 1.0
        # Spectral gate state
        self._gate_noise_floor: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal DSP helpers
    # ------------------------------------------------------------------

    def _update_bg(self, frame_energy: float, flag: int) -> None:
        if flag == 0:
            self._bg_energy = (self.cfg.bg_alpha * self._bg_energy +
                               (1.0 - self.cfg.bg_alpha) * frame_energy)

    def _compressor_gain(self, frame_rms: float, flag: int) -> float:
        """Return linear gain factor for the frame."""
        c = self.cfg
        bg_rms = float(np.sqrt(self._bg_energy + EPS))
        threshold_lin = bg_rms * (10.0 ** (c.comp_threshold_db / 20.0))

        if frame_rms > threshold_lin and flag == 1:
            # Above threshold: apply compression
            excess_db = 20.0 * np.log10(frame_rms / (threshold_lin + EPS))
            # Soft-knee
            if excess_db < c.comp_knee_db:
                ratio = 1.0 + (c.comp_ratio - 1.0) * (excess_db / c.comp_knee_db)
            else:
                ratio = c.comp_ratio
            target_db = -excess_db / ratio
            target_gain = 10.0 ** (target_db / 20.0)
            # Attack
            self._gain_db = (self._att_coef * self._gain_db +
                             (1.0 - self._att_coef) * target_db)
        else:
            # Release towards unity
            self._gain_db = self._rel_coef * self._gain_db
                            # (approaches 0 dB = unity gain)

        lin_gain = float(10.0 ** (self._gain_db / 20.0))
        return float(np.clip(lin_gain, 0.0, 1.0))

    def _spectral_gate_frame(self, frame: np.ndarray, flag: int) -> np.ndarray:
        """Apply Wiener-style spectral gate to one hop-length frame."""
        c = self.cfg
        if not flag:
            return frame
        n  = c.gate_n_fft
        # Rectangular STFT of this short frame
        padded = np.zeros(n, dtype=np.float32)
        n_copy = min(len(frame), n)
        padded[:n_copy] = frame[:n_copy]
        spec = np.fft.rfft(padded)                         # (n//2+1,)
        power = np.abs(spec) ** 2

        # Update noise floor estimate from quiet frames
        if self._gate_noise_floor is None:
            self._gate_noise_floor = power.copy()
        else:
            if not flag:
                self._gate_noise_floor = (0.9 * self._gate_noise_floor +
                                          0.1 * power)

        thr_lin = self._gate_noise_floor * (
            10.0 ** (c.gate_threshold_db / 10.0))
        # Wiener gain: max(1 - noise/signal, floor)
        wiener = np.maximum(1.0 - thr_lin / (power + EPS), c.gate_floor)
        gated_spec = spec * wiener
        gated = np.fft.irfft(gated_spec, n=n)[:n_copy]
        out = frame.copy()
        out[:n_copy] = gated
        return out

    def _ref_cancel_frame(
        self, primary: np.ndarray, reference: np.ndarray, flag: int
    ) -> np.ndarray:
        """Short-window reference cancellation for the burst region."""
        if not flag or not self.cfg.ref_cancel_enabled:
            return primary
        c = self.cfg
        hop = len(primary)
        d = c.ref_max_delay_samples

        # Update circular buffers
        self._ref_buf = np.roll(self._ref_buf, -hop)
        self._ref_buf[-hop:] = reference
        self._pri_buf = np.roll(self._pri_buf, -hop)
        self._pri_buf[-hop:] = primary

        # Normalised cross-correlation to find best delay
        buf_len = min(len(self._ref_buf), 4 * hop)
        pri_seg = self._pri_buf[-buf_len:]
        ref_seg = self._ref_buf[-buf_len:]
        xcorr = np.correlate(pri_seg, ref_seg, mode='full')
        center = len(xcorr) // 2
        lo = max(0, center - d)
        hi = min(len(xcorr), center + d + 1)
        best_lag = int(np.argmax(np.abs(xcorr[lo:hi]))) - d
        best_lag = int(np.clip(best_lag, 0, d))  # causal: only non-negative lags

        # Adaptive scale: least-squares step
        ref_aligned = np.roll(reference, best_lag)
        ref_aligned[:best_lag] = 0.0
        num = float(np.dot(primary, ref_aligned))
        den = float(np.dot(ref_aligned, ref_aligned) + EPS)
        scale = num / den
        # Smooth the scale estimate
        self._ref_scale = (c.ref_alpha * self._ref_scale +
                           (1.0 - c.ref_alpha) * scale)
        cancelled = primary - self._ref_scale * ref_aligned
        return cancelled.astype(np.float32)

    def _crossfade(self, frame: np.ndarray, gain: float) -> np.ndarray:
        """Apply gain with a cross-fade to avoid discontinuities."""
        fade_len = min(self._fade_len, len(frame))
        out = frame * gain
        if abs(gain - self._prev_gain) > 1e-3 and fade_len > 1:
            # Ramp gain from prev to current
            ramp = np.linspace(self._prev_gain, gain, fade_len).astype(np.float32)
            out[:fade_len] = frame[:fade_len] * ramp
        self._prev_gain = gain
        return out

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        primary: np.ndarray,
        reference: Optional[np.ndarray],
        impulse_flag: int,
    ) -> Tuple[np.ndarray, int]:
        """Process one hop-length frame.

        Args:
            primary      : (hop_len,) primary mic frame
            reference    : (hop_len,) reference mic frame, or None
            impulse_flag : 0 or 1 (from ImpulseDetector)

        Returns:
            (processed_frame, impulse_flag)
        """
        primary = primary.astype(np.float32)
        if reference is not None:
            reference = reference.astype(np.float32)

        fe = float(np.mean(primary ** 2) + EPS)
        self._update_bg(fe, impulse_flag)

        frame = primary.copy()

        # 1. Reference cancellation (most upstream)
        if reference is not None:
            frame = self._ref_cancel_frame(frame, reference, impulse_flag)

        # 2. Spectral gate
        frame = self._spectral_gate_frame(frame, impulse_flag)

        # 3. Compressor + cross-fade
        rms = float(np.sqrt(np.mean(frame ** 2) + EPS))
        gain = self._compressor_gain(rms, impulse_flag)
        frame = self._crossfade(frame, gain)

        return frame, impulse_flag

    # ------------------------------------------------------------------
    def process_signal(
        self,
        primary: np.ndarray,
        reference: Optional[np.ndarray],
        det_flags: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a full signal using precomputed detector flags.

        Args:
            primary   : (N,) float32 primary mic signal
            reference : (N,) float32 reference mic signal, or None
            det_flags : (n_frames,) int8 flags from ImpulseDetector

        Returns:
            (processed_signal, flags)
        """
        hop = self.cfg.hop_len
        self.reset()
        n_frames = len(primary) // hop
        out = np.zeros_like(primary)
        out_flags = np.zeros(n_frames, dtype=np.int8)

        for i in range(n_frames):
            p = primary[i * hop: (i + 1) * hop]
            r = reference[i * hop: (i + 1) * hop] if reference is not None else None
            flag = int(det_flags[i]) if i < len(det_flags) else 0
            frame_out, f_flag = self.process_frame(p, r, flag)
            out[i * hop: (i + 1) * hop] = frame_out
            out_flags[i] = f_flag

        return out, out_flags
