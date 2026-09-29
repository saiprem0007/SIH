"""Reference microphone simulation for the ANC-SIH data pipeline.

The reference microphone captures the noise field from a different spatial
position and through a different acoustic transfer function than the primary
microphone.  This module simulates that difference by:

1. Applying a random short FIR filter (different transfer function).
2. Introducing a small integer delay.
3. Adding speech leakage at a configurable SNR.
4. Adding independent sensor (thermal) noise.
5. Optionally replacing the output with synthetic wind noise to simulate a
   degraded / obstructed reference sensor.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from anc.data.synthetic_noise import gen_synthetic_wind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEAR_SILENCE_THRESHOLD: float = 1e-8
"""Threshold below which a signal is considered silent."""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_reference_filter(n_taps: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random normalised FIR filter of length *n_taps*.

    Tap coefficients are drawn from a standard normal distribution and then
    normalised so that the filter has unit energy.

    Args:
        n_taps: Number of filter taps (>= 1).
        rng: Numpy random generator for reproducibility.

    Returns:
        Float32 array of shape ``(n_taps,)``.
    """
    taps = rng.standard_normal(n_taps).astype(np.float32)
    norm = float(np.sqrt(np.sum(taps ** 2)))
    if norm > _NEAR_SILENCE_THRESHOLD:
        taps /= norm
    return taps


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def simulate_reference_mic(
    noise: np.ndarray,
    speech: np.ndarray,
    sr: int,
    cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate the signal received by a reference (error) microphone.

    The reference microphone is modelled as seeing the same acoustic noise
    field as the primary microphone, but through a different transfer function,
    with a small propagation delay, speech leakage, and independent sensor noise.

    Steps:

    1. Apply a random short FIR filter to the noise signal.
    2. Apply a random integer delay (propagation path difference).
    3. Add speech leakage at a configurable level.
    4. Add independent Gaussian sensor noise.
    5. (Optional) Replace output with synthetic wind noise with probability
       ``cfg.mixing.ref_degraded_prob`` to simulate a degraded sensor.

    Args:
        noise: Noise signal at the primary mic, shape ``(N,)``.
        speech: Clean speech signal, shape ``(N,)``.
        sr: Sample rate in Hz.
        cfg: Configuration object with ``mixing.*`` attributes:

            - ``mixing.ref_filter_taps_min`` / ``mixing.ref_filter_taps_max``
            - ``mixing.ref_delay_max_ms``
            - ``mixing.ref_speech_leak_min_db`` / ``mixing.ref_speech_leak_max_db``
            - ``mixing.ref_sensor_noise_db``
            - ``mixing.ref_degraded_prob``

        rng: Numpy random generator for reproducibility.

    Returns:
        Float32 reference signal array of shape ``(N,)``.
    """
    n_samples = len(noise)

    # ------------------------------------------------------------------
    # 1. Random FIR filter (different acoustic transfer function)
    # ------------------------------------------------------------------
    taps_min = int(cfg.mixing.ref_filter_taps_min)
    taps_max = int(cfg.mixing.ref_filter_taps_max)
    n_taps = int(rng.integers(taps_min, taps_max + 1))
    fir = get_reference_filter(n_taps, rng)

    ref = scipy_signal.lfilter(fir, [1.0], noise.astype(np.float64)).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Random integer propagation delay
    # ------------------------------------------------------------------
    max_delay_samples = int(cfg.mixing.ref_delay_max_ms * sr / 1000.0)
    delay_samples = int(rng.integers(0, max(1, max_delay_samples + 1)))

    if delay_samples > 0:
        ref = np.roll(ref, delay_samples)
        ref[:delay_samples] = 0.0  # zero-pad the front (causal)

    # ------------------------------------------------------------------
    # 3. Speech leakage
    # ------------------------------------------------------------------
    leak_db = rng.uniform(
        float(cfg.mixing.ref_speech_leak_min_db),
        float(cfg.mixing.ref_speech_leak_max_db),
    )
    leak_gain = 10.0 ** (leak_db / 20.0)

    # Align speech length with reference
    leak_speech = speech[:n_samples] if len(speech) >= n_samples else np.pad(
        speech, (0, n_samples - len(speech))
    )
    ref = ref + (leak_gain * leak_speech).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Independent sensor (thermal) noise
    # ------------------------------------------------------------------
    sensor_snr_db = float(cfg.mixing.ref_sensor_noise_db)
    ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    if ref_rms < _NEAR_SILENCE_THRESHOLD:
        ref_rms = _NEAR_SILENCE_THRESHOLD

    # sensor_snr_db is typically negative (e.g. -50 dB) meaning sensor noise
    # is 50 dB below the reference signal.
    sensor_std = ref_rms * (10.0 ** (sensor_snr_db / 20.0))
    sensor_noise = rng.standard_normal(n_samples).astype(np.float32) * float(sensor_std)
    ref = ref + sensor_noise

    # ------------------------------------------------------------------
    # 5. Degraded mode (reference mic replaced by synthetic wind)
    # ------------------------------------------------------------------
    degraded_prob = float(cfg.mixing.ref_degraded_prob)
    if rng.random() < degraded_prob:
        logger.debug("Reference mic in degraded mode – replacing with synthetic wind noise.")
        duration_s = n_samples / float(sr)
        wind = gen_synthetic_wind(duration_s, sr, rng)
        # Scale wind to similar RMS as ref so it is plausibly realistic
        wind_rms = float(np.sqrt(np.mean(wind.astype(np.float64) ** 2))) + _NEAR_SILENCE_THRESHOLD
        target_rms = ref_rms
        wind = (wind * (target_rms / wind_rms)).astype(np.float32)
        ref = wind

    return ref[:n_samples].astype(np.float32)
