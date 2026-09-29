"""Synthetic noise generators for the ANC-SIH data pipeline.

Each generator returns a float32 numpy array at the requested sample rate,
peak-normalised to approximately 0.9.  All randomness is driven by a caller-
supplied ``numpy.random.Generator`` so that results are fully reproducible.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional

import numpy as np
from scipy import signal as scipy_signal

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PEAK_TARGET: float = 0.90
"""Target peak amplitude for all synthetic generators."""

_PINK_ALPHA: float = 1.0
"""Spectral exponent for pink noise (power ∝ 1/f^alpha)."""

_BLAST_DECAY_MIN: float = 20.0
_BLAST_DECAY_MAX: float = 100.0

_SIREN_F_LO: float = 300.0   # Hz
_SIREN_F_HI: float = 3000.0  # Hz
_SIREN_SWEEP_MIN: float = 0.5  # seconds
_SIREN_SWEEP_MAX: float = 2.0  # seconds
_SIREN_AM_DEPTH: float = 0.10

_WIND_LOWPASS_FC: float = 800.0  # Hz
_WIND_LOWPASS_ORDER: int = 4

_ROTOR_F_MIN: float = 10.0   # Hz
_ROTOR_F_MAX: float = 25.0   # Hz
_ROTOR_N_HARMONICS: int = 12
_ROTOR_NOISE_FLOOR_DB: float = -20.0  # dB relative to fundamental

_CROWD_BP_LO: float = 300.0   # Hz
_CROWD_BP_HI: float = 4000.0  # Hz
_CROWD_BP_ORDER: int = 4
_CROWD_AM_RATE_MIN: float = 0.5   # Hz
_CROWD_AM_RATE_MAX: float = 3.0   # Hz
_CROWD_BURST_RATE: float = 2.0    # bursts per second on average
_CROWD_BURST_DUR_MIN: float = 0.05  # seconds
_CROWD_BURST_DUR_MAX: float = 0.25  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_normalise(sig: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    """Peak-normalise *sig* to *target*.

    Args:
        sig: Input signal, any shape.
        target: Desired peak amplitude.

    Returns:
        Normalised float32 array.
    """
    peak = np.max(np.abs(sig))
    if peak < 1e-8:
        return sig.astype(np.float32)
    return (sig * (target / peak)).astype(np.float32)


def _butter_bandpass(lo: float, hi: float, sr: int, order: int = 4) -> tuple:
    """Return (b, a) Butterworth bandpass coefficients.

    Args:
        lo: Lower cutoff frequency in Hz.
        hi: Upper cutoff frequency in Hz.
        sr: Sample rate in Hz.
        order: Filter order (applied to each band, so effective order doubles).

    Returns:
        Tuple (b, a) of filter coefficients.
    """
    nyq = sr / 2.0
    low = np.clip(lo / nyq, 1e-4, 1.0 - 1e-4)
    high = np.clip(hi / nyq, 1e-4, 1.0 - 1e-4)
    return scipy_signal.butter(order, [low, high], btype='band')


def _butter_lowpass(fc: float, sr: int, order: int = 4) -> tuple:
    """Return (b, a) Butterworth low-pass coefficients.

    Args:
        fc: Cutoff frequency in Hz.
        sr: Sample rate in Hz.
        order: Filter order.

    Returns:
        Tuple (b, a) of filter coefficients.
    """
    nyq = sr / 2.0
    wn = np.clip(fc / nyq, 1e-4, 1.0 - 1e-4)
    return scipy_signal.butter(order, wn, btype='low')


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink (1/f) noise via spectral shaping.

    White noise is FFT-transformed, each bin scaled by 1/sqrt(f), then
    IFFT-transformed back.  The DC bin is zeroed.

    Args:
        n_samples: Number of output samples.
        rng: Numpy random generator instance.

    Returns:
        Float64 array of length *n_samples*.
    """
    white = rng.standard_normal(n_samples)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = np.inf  # avoid divide-by-zero; DC → 0 below
    scale = 1.0 / np.sqrt(freqs)
    scale[0] = 0.0  # zero DC
    pink_fft = fft_vals * scale
    pink = np.fft.irfft(pink_fft, n=n_samples)
    return pink.astype(np.float64)


# ---------------------------------------------------------------------------
# Individual generators
# ---------------------------------------------------------------------------

def gen_synthetic_blast(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an exponentially decaying broadband noise burst (blast/gunshot).

    The signal is Gaussian white noise multiplied by an exponential decay
    envelope: ``n(t) * exp(-decay_rate * t)`` where ``decay_rate`` is drawn
    uniformly from [20, 100].

    Metadata tags: category='impulsive', source='synthetic'

    Args:
        duration_s: Duration of the output in seconds.
        sr: Sample rate in Hz.
        rng: Numpy random generator for reproducibility.

    Returns:
        Float32 array of shape ``(n_samples,)`` peak-normalised to ~0.90.
    """
    n_samples = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False, dtype=np.float64)

    decay_rate = rng.uniform(_BLAST_DECAY_MIN, _BLAST_DECAY_MAX)
    envelope = np.exp(-decay_rate * t)

    # Burst onset: place it somewhere in the first 20% of the clip
    onset_frac = rng.uniform(0.0, 0.2)
    onset_idx = int(onset_frac * n_samples)
    noise_burst = rng.standard_normal(n_samples) * envelope
    noise_burst[:onset_idx] = 0.0  # silence before onset

    return _peak_normalise(noise_burst)


def gen_synthetic_siren(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a swept sinusoidal siren tone with mild amplitude modulation.

    The instantaneous frequency sweeps sinusoidally between ``_SIREN_F_LO``
    and ``_SIREN_F_HI`` Hz with a sweep period drawn from
    [_SIREN_SWEEP_MIN, _SIREN_SWEEP_MAX] seconds.  A 10 % AM depth is applied.

    Metadata tags: category='siren', source='synthetic'

    Args:
        duration_s: Duration of the output in seconds.
        sr: Sample rate in Hz.
        rng: Numpy random generator for reproducibility.

    Returns:
        Float32 array of shape ``(n_samples,)`` peak-normalised to ~0.90.
    """
    n_samples = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False, dtype=np.float64)

    sweep_period = rng.uniform(_SIREN_SWEEP_MIN, _SIREN_SWEEP_MAX)
    f_mid = (_SIREN_F_LO + _SIREN_F_HI) / 2.0
    f_dev = (_SIREN_F_HI - _SIREN_F_LO) / 2.0

    # Instantaneous frequency (sinusoidal FM)
    inst_freq = f_mid + f_dev * np.sin(2.0 * np.pi * t / sweep_period)

    # Phase by numerical integration of inst_freq
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr

    # FM tone
    tone = np.sin(phase)

    # Amplitude modulation (slight wobble)
    am = 1.0 - _SIREN_AM_DEPTH * np.sin(2.0 * np.pi * t / sweep_period)
    siren = tone * am

    return _peak_normalise(siren)


def gen_synthetic_wind(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate wind-like noise as low-passed pink noise.

    Pink noise is generated via spectral shaping (FFT domain), then passed
    through a Butterworth low-pass filter at ``_WIND_LOWPASS_FC`` Hz.

    Metadata tags: category='wind', source='synthetic'

    Args:
        duration_s: Duration of the output in seconds.
        sr: Sample rate in Hz.
        rng: Numpy random generator for reproducibility.

    Returns:
        Float32 array of shape ``(n_samples,)`` peak-normalised to ~0.90.
    """
    n_samples = int(duration_s * sr)
    pink = _pink_noise(n_samples, rng)

    b, a = _butter_lowpass(_WIND_LOWPASS_FC, sr, order=_WIND_LOWPASS_ORDER)
    filtered = scipy_signal.lfilter(b, a, pink)

    return _peak_normalise(filtered)


def gen_synthetic_rotor(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate helicopter/drone rotor noise.

    Models the blade passage frequency (BPF) as a harmonic series:
    - Fundamental f0 ~ U(10, 25) Hz
    - k-th harmonic amplitude = 1/k  (k = 1 … 12)
    - Broadband noise floor added at ``_ROTOR_NOISE_FLOOR_DB`` dB relative to
      the fundamental's RMS.

    Metadata tags: category='rotor', source='synthetic'

    Args:
        duration_s: Duration of the output in seconds.
        sr: Sample rate in Hz.
        rng: Numpy random generator for reproducibility.

    Returns:
        Float32 array of shape ``(n_samples,)`` peak-normalised to ~0.90.
    """
    n_samples = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False, dtype=np.float64)

    f0 = rng.uniform(_ROTOR_F_MIN, _ROTOR_F_MAX)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=_ROTOR_N_HARMONICS)

    harmonic_sum = np.zeros(n_samples, dtype=np.float64)
    for k in range(1, _ROTOR_N_HARMONICS + 1):
        freq = k * f0
        if freq >= sr / 2.0:
            break
        harmonic_sum += (1.0 / k) * np.sin(2.0 * np.pi * freq * t + phases[k - 1])

    # Broadband noise floor
    fund_rms = np.sqrt(np.mean(harmonic_sum ** 2)) + 1e-10
    noise_gain = fund_rms * 10.0 ** (_ROTOR_NOISE_FLOOR_DB / 20.0)
    broadband = rng.standard_normal(n_samples) * noise_gain

    rotor = harmonic_sum + broadband
    return _peak_normalise(rotor)


def gen_synthetic_crowd(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate crowd noise: bandpass-filtered noise with AM and burst events.

    Base layer is band-pass filtered white noise (300–4000 Hz) with slow
    amplitude modulation.  Short burst events (simulating shouts / claps) are
    inserted at Poisson-distributed times.

    Metadata tags: category='crowd', source='synthetic'

    Args:
        duration_s: Duration of the output in seconds.
        sr: Sample rate in Hz.
        rng: Numpy random generator for reproducibility.

    Returns:
        Float32 array of shape ``(n_samples,)`` peak-normalised to ~0.90.
    """
    n_samples = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False, dtype=np.float64)

    # Base bandpass noise
    white = rng.standard_normal(n_samples)
    b, a = _butter_bandpass(_CROWD_BP_LO, _CROWD_BP_HI, sr, order=_CROWD_BP_ORDER)
    base_noise = scipy_signal.lfilter(b, a, white)

    # Slow amplitude modulation
    am_rate = rng.uniform(_CROWD_AM_RATE_MIN, _CROWD_AM_RATE_MAX)
    am_phase = rng.uniform(0.0, 2.0 * np.pi)
    am = 0.7 + 0.3 * np.sin(2.0 * np.pi * am_rate * t + am_phase)
    crowd = base_noise * am

    # Burst events (Poisson arrivals)
    n_bursts = rng.poisson(_CROWD_BURST_RATE * duration_s)
    for _ in range(n_bursts):
        burst_dur = rng.uniform(_CROWD_BURST_DUR_MIN, _CROWD_BURST_DUR_MAX)
        burst_len = min(int(burst_dur * sr), n_samples)
        onset = rng.integers(0, max(1, n_samples - burst_len))
        burst = rng.standard_normal(burst_len)
        # Bandpass the burst as well
        if burst_len > 30:
            burst = scipy_signal.lfilter(b, a, burst)
        # Taper with a Hann window
        burst *= np.hanning(burst_len)
        crowd[onset: onset + burst_len] += burst * rng.uniform(0.5, 2.0)

    return _peak_normalise(crowd)


# ---------------------------------------------------------------------------
# Registry and dispatcher
# ---------------------------------------------------------------------------

SYNTHETIC_GENERATORS: Dict[str, Callable[[float, int, np.random.Generator], np.ndarray]] = {
    "impulsive": gen_synthetic_blast,
    "siren":     gen_synthetic_siren,
    "wind":      gen_synthetic_wind,
    "rotor":     gen_synthetic_rotor,
    "crowd":     gen_synthetic_crowd,
}
"""Mapping from noise category name to its generator function."""


def gen_synthetic_noise(
    category: str,
    duration_s: float,
    sr: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic noise for the requested category.

    Dispatches to the appropriate generator function.  If *category* is not
    found in :data:`SYNTHETIC_GENERATORS`, pink noise is returned and a
    warning is emitted.

    Args:
        category: Noise category, e.g. ``'impulsive'``, ``'wind'``.
        duration_s: Duration in seconds.
        sr: Sample rate in Hz.
        seed: Optional integer seed for reproducibility.  If ``None``, the
            generator is seeded from OS entropy.

    Returns:
        Float32 array of shape ``(n_samples,)`` peak-normalised to ~0.90.
    """
    rng = np.random.default_rng(seed)

    if category in SYNTHETIC_GENERATORS:
        return SYNTHETIC_GENERATORS[category](duration_s, sr, rng)

    warnings.warn(
        f"Unknown synthetic noise category '{category}'. "
        "Falling back to pink noise.",
        stacklevel=2,
    )
    n_samples = int(duration_s * sr)
    pink = _pink_noise(n_samples, rng)
    return _peak_normalise(pink)
