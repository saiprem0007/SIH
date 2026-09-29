"""Audio augmentation utilities for the ANC-SIH data pipeline.

Provides functions for loading audio, applying room impulse responses,
random gain augmentation, hard clipping, and resampling.  All functions
operate on float32 numpy arrays.
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEAR_SILENCE_THRESHOLD: float = 1e-8
"""Signals whose peak falls below this value are considered silent."""

_DEFAULT_PEAK_TARGET: float = 0.95
"""Default peak amplitude for peak_normalize."""


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(
    path: Union[str, Path],
    sr: int,
    mono: bool = True,
) -> np.ndarray:
    """Load an audio file and return a float32 numpy array.

    Resamples to *sr* using :func:`resample_audio` if the file's native
    sample rate differs.  Converts to mono by averaging channels when
    *mono* is ``True``.

    Args:
        path: Path to the audio file (wav, flac, ogg, mp3 …).
        sr: Target sample rate in Hz.
        mono: If ``True``, collapse channels by averaging.

    Returns:
        Float32 array of shape ``(n_samples,)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If soundfile cannot open the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    # data shape: (n_samples, n_channels)

    if mono:
        data = data.mean(axis=1)  # (n_samples,)
    else:
        data = data[:, 0]  # use first channel only when not merging

    if native_sr != sr:
        data = resample_audio(data, native_sr, sr)

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Peak normalisation
# ---------------------------------------------------------------------------

def peak_normalize(
    signal: np.ndarray,
    target_level: float = _DEFAULT_PEAK_TARGET,
) -> np.ndarray:
    """Scale *signal* so that its peak absolute value equals *target_level*.

    Near-silent signals (peak < :data:`_NEAR_SILENCE_THRESHOLD`) are returned
    unchanged to avoid amplifying numerical noise.

    Args:
        signal: Input signal, any shape.
        target_level: Desired peak amplitude (default 0.95).

    Returns:
        Float32 array, same shape as *signal*.
    """
    peak = float(np.max(np.abs(signal)))
    if peak < _NEAR_SILENCE_THRESHOLD:
        return signal.astype(np.float32)
    return (signal * (target_level / peak)).astype(np.float32)


# ---------------------------------------------------------------------------
# RIR loading and application
# ---------------------------------------------------------------------------

def load_rir(
    rir_dir: Union[str, Path],
    rng: np.random.Generator,
    sr: int,
) -> Optional[np.ndarray]:
    """Load a random room impulse response from *rir_dir*.

    Recursively scans *rir_dir* for ``.wav`` files, picks one at random,
    and returns it as a float32 array.

    Args:
        rir_dir: Directory containing RIR wav files.
        rng: Numpy random generator used for random selection.
        sr: Sample rate to resample the RIR to.

    Returns:
        Float32 array of shape ``(n_samples,)``, or ``None`` if *rir_dir*
        does not exist or contains no wav files.
    """
    rir_dir = Path(rir_dir)
    if not rir_dir.exists():
        logger.debug("RIR directory does not exist: %s", rir_dir)
        return None

    wav_files = list(rir_dir.rglob("*.wav"))
    if not wav_files:
        logger.debug("No .wav files found in RIR directory: %s", rir_dir)
        return None

    chosen = wav_files[rng.integers(0, len(wav_files))]
    try:
        rir = load_audio(chosen, sr=sr, mono=True)
    except Exception as exc:
        logger.warning("Failed to load RIR %s: %s", chosen, exc)
        return None

    return rir


def apply_rir(
    signal: np.ndarray,
    rir: np.ndarray,
    mode: str = "causal",
) -> np.ndarray:
    """Convolve *signal* with a room impulse response *rir*.

    The convolved output is trimmed to the same length as *signal* and then
    RMS-normalised to match the input signal's original RMS.

    Args:
        signal: Input signal, shape ``(N,)``.
        rir: Room impulse response, shape ``(M,)``.
        mode: ``'causal'`` (keep first N samples) or ``'full'`` (keep last N
            samples of the full convolution — not recommended for ANC).

    Returns:
        Float32 array of shape ``(N,)``.
    """
    n = len(signal)
    orig_rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) + 1e-10

    convolved = scipy_signal.fftconvolve(signal.astype(np.float64), rir.astype(np.float64))

    if mode == "causal":
        out = convolved[:n]
    else:
        # 'full' mode: trim from the tail
        out = convolved[-n:]

    # Pad if convolved output is shorter than n (edge case for very short signals)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))

    # Preserve RMS
    out_rms = float(np.sqrt(np.mean(out ** 2))) + 1e-10
    out = out * (orig_rms / out_rms)

    return out[:n].astype(np.float32)


# ---------------------------------------------------------------------------
# Gain and clipping augmentation
# ---------------------------------------------------------------------------

def random_gain(
    signal: np.ndarray,
    min_db: float,
    max_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a random gain in decibels to *signal*.

    The gain is drawn uniformly from [*min_db*, *max_db*] dB.

    Args:
        signal: Input signal, shape ``(N,)``.
        min_db: Minimum gain in dB (can be negative).
        max_db: Maximum gain in dB.
        rng: Numpy random generator.

    Returns:
        Float32 array of the same shape as *signal*.
    """
    gain_db = rng.uniform(min_db, max_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (signal * gain_linear).astype(np.float32)


def hard_clip(
    signal: np.ndarray,
    level: float = 0.95,
) -> np.ndarray:
    """Clip *signal* symmetrically to ``[-level, +level]``.

    Args:
        signal: Input signal, any shape.
        level: Clip threshold (default 0.95).

    Returns:
        Float32 array, same shape as *signal*, with values in
        ``[-level, level]``.
    """
    return np.clip(signal, -level, level).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_audio(
    signal: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample *signal* from *orig_sr* to *target_sr*.

    Uses :func:`scipy.signal.resample_poly` with the GCD-reduced up/down
    ratio for efficiency.

    Args:
        signal: Input signal, shape ``(N,)``.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        Float32 array resampled to ``target_sr``.
    """
    if orig_sr == target_sr:
        return signal.astype(np.float32)

    common = gcd(orig_sr, target_sr)
    up = target_sr // common
    down = orig_sr // common

    resampled = scipy_signal.resample_poly(signal.astype(np.float64), up, down)
    return resampled.astype(np.float32)
