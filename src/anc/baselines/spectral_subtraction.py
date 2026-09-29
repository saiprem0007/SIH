"""
Spectral Subtraction / Wiener Filter baseline.

Estimates noise power spectrum from initial frames or using minimum statistics,
then suppresses noise in the STFT domain.

Formula (over-subtracted spectral subtraction):
    |S_hat(f)|^2 = max(|Y(f)|^2 - alpha * |N_hat(f)|^2, beta * |Y(f)|^2)
    Phase: use original noisy phase
"""
import logging
from typing import List, Optional

import numpy as np
from scipy.signal import get_window

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_SR: int = 16000
_DEFAULT_N_FFT: int = 512
_DEFAULT_HOP_LEN: int = 160
_DEFAULT_WIN_LEN: int = 320
_DEFAULT_ALPHA: float = 2.0        # over-subtraction factor
_DEFAULT_BETA: float = 0.01        # spectral floor factor
_DEFAULT_NOISE_EST_FRAMES: int = 6  # frames used for initial noise estimate
_DEFAULT_SMOOTHING: float = 0.98   # exponential smoothing coefficient


class SpectralSubtractor:
    """Spectral Subtraction / Wiener Filter noise suppressor.

    Operates in the STFT domain using an overlap-add synthesis scheme.
    Noise power spectral density (PSD) is estimated from the first
    ``noise_est_frames`` frames and updated adaptively on low-energy frames
    (minimum-statistics approximation).

    Args:
        sr: Sampling rate in Hz.
        n_fft: FFT size (zero-padding applied if > ``win_len``).
        hop_len: Frame hop in samples (stride).
        win_len: Analysis window length in samples.
        alpha: Over-subtraction factor (>= 1 recommended). Larger values
               remove more noise but may introduce musical noise.
        beta: Spectral floor gain. Prevents complete silence in spectral
              subtraction — keeps ``beta * |Y(f)|^2`` as minimum energy.
        noise_est_frames: Number of initial frames assumed to be noise-only
                          for bootstrapping the noise PSD estimate.
        smoothing: Exponential smoothing coefficient for online noise PSD
                   updates (closer to 1 → slower adaptation).

    Raises:
        ValueError: If parameter combinations are invalid.
    """

    def __init__(
        self,
        sr: int = _DEFAULT_SR,
        n_fft: int = _DEFAULT_N_FFT,
        hop_len: int = _DEFAULT_HOP_LEN,
        win_len: int = _DEFAULT_WIN_LEN,
        alpha: float = _DEFAULT_ALPHA,
        beta: float = _DEFAULT_BETA,
        noise_est_frames: int = _DEFAULT_NOISE_EST_FRAMES,
        smoothing: float = _DEFAULT_SMOOTHING,
    ) -> None:
        if hop_len <= 0:
            raise ValueError(f"hop_len must be > 0, got {hop_len}")
        if win_len <= 0:
            raise ValueError(f"win_len must be > 0, got {win_len}")
        if n_fft < win_len:
            raise ValueError(
                f"n_fft ({n_fft}) must be >= win_len ({win_len}) to allow zero-padding."
            )
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if noise_est_frames <= 0:
            raise ValueError(f"noise_est_frames must be > 0, got {noise_est_frames}")

        self.sr: int = sr
        self.n_fft: int = n_fft
        self.hop_len: int = hop_len
        self.win_len: int = win_len
        self.alpha: float = alpha
        self.beta: float = beta
        self.noise_est_frames: int = noise_est_frames
        self.smoothing: float = smoothing

        # Number of positive-frequency bins (including DC and Nyquist)
        self._n_bins: int = n_fft // 2 + 1

        # Analysis window
        self._window: np.ndarray = get_window("hann", win_len, fftbins=True).astype(
            np.float64
        )

        # State variables — initialised via reset()
        self._noise_psd: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._overlap_buffer: np.ndarray = np.zeros(win_len - hop_len, dtype=np.float64)
        self._noise_frames: List[np.ndarray] = []
        # Lookahead buffer accumulates incoming samples between hops
        self._input_buffer: np.ndarray = np.zeros(win_len, dtype=np.float64)

        logger.debug(
            "SpectralSubtractor initialised — sr=%d, n_fft=%d, hop=%d, win=%d",
            sr,
            n_fft,
            hop_len,
            win_len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state (noise PSD, buffers, frame counter)."""
        self._noise_psd = None
        self._frame_count = 0
        self._overlap_buffer = np.zeros(self.win_len - self.hop_len, dtype=np.float64)
        self._noise_frames = []
        self._input_buffer = np.zeros(self.win_len, dtype=np.float64)
        logger.debug("SpectralSubtractor state reset.")

    def get_state(self) -> dict:
        """Return a snapshot of the current processing state.

        Returns:
            Dictionary containing ``noise_psd``, ``frame_count``,
            ``overlap_buffer``, ``noise_frames``, and ``input_buffer``.
        """
        return {
            "noise_psd": self._noise_psd.copy() if self._noise_psd is not None else None,
            "frame_count": self._frame_count,
            "overlap_buffer": self._overlap_buffer.copy(),
            "noise_frames": [f.copy() for f in self._noise_frames],
            "input_buffer": self._input_buffer.copy(),
        }

    def set_state(self, state: dict) -> None:
        """Restore processing state from a snapshot.

        Args:
            state: Dictionary previously returned by :meth:`get_state`.

        Raises:
            KeyError: If required keys are missing from ``state``.
        """
        self._noise_psd = (
            state["noise_psd"].copy() if state["noise_psd"] is not None else None
        )
        self._frame_count = int(state["frame_count"])
        self._overlap_buffer = np.asarray(state["overlap_buffer"], dtype=np.float64).copy()
        self._noise_frames = [np.asarray(f, dtype=np.float64).copy() for f in state["noise_frames"]]
        self._input_buffer = np.asarray(state["input_buffer"], dtype=np.float64).copy()
        logger.debug("SpectralSubtractor state restored.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one hop-sized audio frame in streaming mode.

        Internally maintains an overlap buffer so that the full analysis
        window is always available. Output is also overlap-added.

        Args:
            frame: New audio samples, shape ``(hop_len,)``.

        Returns:
            Enhanced audio frame, shape ``(hop_len,)`` as ``np.float32``.

        Raises:
            ValueError: If ``frame`` does not have length ``hop_len``.
        """
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape[0] != self.hop_len:
            raise ValueError(
                f"frame must have length hop_len={self.hop_len}, got {frame.shape[0]}"
            )

        # Shift the input buffer left by hop_len and append new samples
        self._input_buffer = np.roll(self._input_buffer, -self.hop_len)
        self._input_buffer[-self.hop_len :] = frame

        # Apply analysis window to the latest win_len samples
        windowed: np.ndarray = self._input_buffer[-self.win_len :] * self._window

        # Zero-pad and take FFT
        padded: np.ndarray = np.zeros(self.n_fft, dtype=np.float64)
        padded[: self.win_len] = windowed
        spectrum: np.ndarray = np.fft.rfft(padded)

        mag: np.ndarray = np.abs(spectrum)
        phase: np.ndarray = np.angle(spectrum)

        # Update noise PSD estimate
        self._estimate_noise_psd(mag)
        self._frame_count += 1

        # If noise PSD not yet bootstrapped, pass through
        if self._noise_psd is None:
            out_samples = frame.copy()
            return out_samples.astype(np.float32)

        # Spectral suppression
        enhanced_spectrum: np.ndarray = self._suppress(mag, phase)

        # iFFT
        enhanced_frame_full: np.ndarray = np.fft.irfft(enhanced_spectrum, n=self.n_fft)
        # Take only win_len samples (discard zero-padded tail)
        enhanced_windowed: np.ndarray = enhanced_frame_full[: self.win_len]

        # Overlap-add synthesis
        overlap_len: int = self.win_len - self.hop_len
        out_hop: np.ndarray = np.empty(self.hop_len, dtype=np.float64)

        # The first hop_len samples combine with stored overlap
        out_hop[:] = enhanced_windowed[: self.hop_len] + self._overlap_buffer[: self.hop_len]

        # Update overlap buffer
        # New overlap = remaining win_len - hop_len samples (shifted + added)
        new_overlap: np.ndarray = np.zeros(overlap_len, dtype=np.float64)
        new_overlap[:] = enhanced_windowed[self.hop_len :]
        # Add carry-over from previous overlap
        carry: int = min(overlap_len - self.hop_len, overlap_len)
        if carry > 0:
            new_overlap[:carry] += self._overlap_buffer[self.hop_len : self.hop_len + carry]
        self._overlap_buffer = new_overlap

        return out_hop.astype(np.float32)

    def process_signal(self, noisy: np.ndarray, reset: bool = True) -> np.ndarray:
        """Process a full signal in offline mode.

        Internally calls :meth:`process_frame` hop-by-hop.

        Args:
            noisy: 1-D noisy audio signal (any dtype, converted to float64).
            reset: If ``True`` (default), reset state before processing.

        Returns:
            Enhanced signal as ``np.float32``, same length as ``noisy``.
        """
        noisy = np.asarray(noisy, dtype=np.float64)
        if noisy.ndim != 1:
            raise ValueError("noisy must be a 1-D array.")

        if reset:
            self.reset()

        n_original: int = noisy.shape[0]

        # Pad to a multiple of hop_len (so every sample is processed)
        pad_len: int = (self.hop_len - n_original % self.hop_len) % self.hop_len
        if pad_len > 0:
            noisy = np.concatenate([noisy, np.zeros(pad_len, dtype=np.float64)])

        n_padded: int = noisy.shape[0]
        n_frames: int = n_padded // self.hop_len

        output_chunks: List[np.ndarray] = []
        for i in range(n_frames):
            hop_samples = noisy[i * self.hop_len : (i + 1) * self.hop_len]
            enhanced_hop = self.process_frame(hop_samples)
            output_chunks.append(enhanced_hop)

        enhanced_full: np.ndarray = np.concatenate(output_chunks).astype(np.float32)

        # Trim to original length
        return enhanced_full[:n_original]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_noise_psd(self, mag_spectrum: np.ndarray) -> None:
        """Update the noise PSD estimate from a magnitude spectrum.

        During the initial ``noise_est_frames`` frames the spectra are
        accumulated. On completion the mean power is set as the initial PSD.
        For subsequent frames, exponential smoothing is applied only when the
        frame energy is below a threshold (minimum-statistics approximation).

        Args:
            mag_spectrum: One-sided magnitude spectrum, shape ``(n_bins,)``.
        """
        if len(self._noise_frames) < self.noise_est_frames:
            # Accumulation phase
            self._noise_frames.append(mag_spectrum.copy())
            if len(self._noise_frames) == self.noise_est_frames:
                # Bootstrap: mean of squared magnitudes across all initial frames
                stacked: np.ndarray = np.stack(self._noise_frames, axis=0)  # (K, n_bins)
                self._noise_psd = np.mean(stacked ** 2, axis=0)
                logger.debug(
                    "Noise PSD initialised from %d frames.", self.noise_est_frames
                )
            return  # noise_psd still None until all initial frames collected

        # Online update phase — minimum-statistics approximation:
        # Only update the noise estimate when the frame energy is low
        # compared to the current noise estimate (i.e., likely a noise-only frame).
        current_power: np.ndarray = mag_spectrum ** 2
        frame_energy: float = float(np.mean(current_power))
        noise_energy: float = float(np.mean(self._noise_psd))

        # Update threshold: frame energy within 3 dB of noise energy → noise frame
        if frame_energy <= noise_energy * 2.0:
            self._noise_psd = (
                self.smoothing * self._noise_psd
                + (1.0 - self.smoothing) * current_power
            )

    def _suppress(self, mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction with noise floor.

        Args:
            mag: Magnitude spectrum (one-sided), shape ``(n_bins,)``.
            phase: Phase spectrum (one-sided), shape ``(n_bins,)``.

        Returns:
            Complex STFT spectrum after suppression, shape ``(n_bins,)``.
        """
        assert self._noise_psd is not None, "Noise PSD must be estimated before suppression."

        mag_sq: np.ndarray = mag ** 2

        # Over-subtracted spectral subtraction
        suppressed_sq: np.ndarray = mag_sq - self.alpha * self._noise_psd

        # Apply spectral floor: never go below beta * |Y|^2
        floor: np.ndarray = self.beta * mag_sq
        suppressed_sq = np.maximum(suppressed_sq, floor)
        suppressed_sq = np.maximum(suppressed_sq, 0.0)  # ensure non-negative

        suppressed_mag: np.ndarray = np.sqrt(suppressed_sq)

        # Reconstruct complex spectrum using original noisy phase
        return suppressed_mag * np.exp(1j * phase)

    @classmethod
    def from_config(cls, cfg, sr: int = _DEFAULT_SR) -> "SpectralSubtractor":
        """Instantiate a :class:`SpectralSubtractor` from a config object.

        Reads parameters from ``cfg.baselines.spectral_sub.*``.

        Args:
            cfg: Configuration object with nested attribute access.
            sr: Sampling rate to use (may override config or fill in default).

        Returns:
            A fully initialised :class:`SpectralSubtractor`.
        """
        ss_cfg = cfg.baselines.spectral_sub
        return cls(
            sr=int(getattr(ss_cfg, "sr", sr)),
            n_fft=int(ss_cfg.n_fft),
            hop_len=int(ss_cfg.hop_len),
            win_len=int(ss_cfg.win_len),
            alpha=float(ss_cfg.alpha),
            beta=float(ss_cfg.beta),
            noise_est_frames=int(ss_cfg.noise_est_frames),
            smoothing=float(ss_cfg.smoothing),
        )
