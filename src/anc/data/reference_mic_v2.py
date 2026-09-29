"""
Extended reference microphone simulator (Phase 4).

Modes
-----
filter      : Original Phase 0 FIR-filter simulation (fast, no extra deps)
demand_pair : Use two channels of the same DEMAND multi-channel recording
              as primary-noise and reference-noise (real spatial decorrelation)
pyroom      : Simulate with pyroomacoustics (requires: pip install pyroomacoustics)

All modes:
  - Add speech leakage to reference
  - Add independent sensor noise
  - Degrade with probability cfg.mixing.ref_degraded_prob

RIR caching
-----------
For ``pyroom`` mode the room impulse responses are cached on disk in
``cache_dir`` (default: ``data/rir_cache/``), keyed by a deterministic hash
of the room parameters.  Subsequent calls with identical room geometry re-use
the cached RIRs, making repeated dataset generation much faster.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

# Import the original Phase 0 simulator
from anc.data.reference_mic import simulate_reference_mic  # noqa: F401
from anc.data.synthetic_noise import gen_synthetic_wind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEAR_SILENCE: float = 1e-8
_VALID_MODES: List[str] = ["filter", "demand_pair", "pyroom"]

# Default shoebox room parameters for pyroom mode
_ROOM_DIM: Tuple[float, float, float] = (3.0, 3.0, 2.5)       # metres
_ROOM_ABSORPTION: float = 0.3
_PRIMARY_MIC: Tuple[float, float] = (0.5, 1.5)                # [x, y]
_REFERENCE_MIC: Tuple[float, float] = (2.5, 1.5)
_SPEECH_SRC: Tuple[float, float] = (0.6, 1.5)
_NOISE_SRC: Tuple[float, float] = (1.5, 0.5)


# ---------------------------------------------------------------------------
# RIR cache helpers
# ---------------------------------------------------------------------------

def _rir_cache_key(sr: int, room_dim: tuple, absorption: float, seed: int) -> str:
    """Build a deterministic cache key from room parameters.

    Args:
        sr:         Sample rate in Hz.
        room_dim:   Room dimensions as (x, y, z) metres.
        absorption: Wall absorption coefficient.
        seed:       RNG seed used for the simulation.

    Returns:
        8-character hex string suitable for use as a filename stem.
    """
    payload = json.dumps(
        {"sr": sr, "room_dim": room_dim, "absorption": absorption, "seed": seed},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def _load_rir_cache(
    cache_dir: Path, key: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load cached RIRs from disk if available.

    Args:
        cache_dir: Directory that holds cached ``.npy`` files.
        key:       Cache key returned by :func:`_rir_cache_key`.

    Returns:
        Tuple ``(rir_primary, rir_reference)`` or ``None`` on cache miss.
    """
    p0 = cache_dir / f"{key}_primary.npy"
    p1 = cache_dir / f"{key}_reference.npy"
    if p0.exists() and p1.exists():
        try:
            return np.load(str(p0)), np.load(str(p1))
        except Exception as exc:  # noqa: BLE001
            logger.warning("RIR cache load failed for key=%s: %s", key, exc)
    return None


def _save_rir_cache(
    cache_dir: Path,
    key: str,
    rir_primary: np.ndarray,
    rir_reference: np.ndarray,
) -> None:
    """Save RIRs to disk for future reuse.

    Args:
        cache_dir:     Directory to write ``.npy`` files to (created if needed).
        key:           Cache key.
        rir_primary:   RIR for the primary microphone.
        rir_reference: RIR for the reference microphone.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_dir / f"{key}_primary.npy"), rir_primary)
        np.save(str(cache_dir / f"{key}_reference.npy"), rir_reference)
        logger.debug("RIR cache saved: key=%s in %s", key, cache_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RIR cache save failed for key=%s: %s", key, exc)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReferenceSimulator:
    """Simulate a reference microphone signal with multiple acoustic modes.

    Supports three modes:

    - ``'filter'``: Fast FIR-filter simulation (Phase 0 behaviour).
    - ``'demand_pair'``: Two spatially separated channels from a DEMAND
      multi-channel recording.
    - ``'pyroom'``: Full pyroomacoustics shoebox simulation with RIR caching.

    Args:
        mode:      Simulation mode.  One of ``'filter'``, ``'demand_pair'``,
                   ``'pyroom'``.
        cfg:       ANC-SIH Config object (provides ``mixing.*`` attributes).
                   If ``None``, sensible defaults are used.
        sr:        Sample rate in Hz (default 16 000).
        seed:      Base random seed for RIR cache keys (default 42).
        cache_dir: Directory for pyroomacoustics RIR cache.
    """

    def __init__(
        self,
        mode: str = "filter",
        cfg=None,
        sr: int = 16000,
        seed: int = 42,
        cache_dir: str = "data/rir_cache",
    ) -> None:
        """Initialise the simulator."""
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes: {_VALID_MODES}"
            )
        self._mode = mode
        self._cfg = cfg
        self._sr = sr
        self._seed = seed
        self._cache_dir = Path(cache_dir)

        # Fallback default mixing params (used when cfg is None)
        self._defaults: Dict = {
            "ref_filter_taps_min":    4,
            "ref_filter_taps_max":    32,
            "ref_delay_max_ms":       5.0,
            "ref_speech_leak_min_db": -30.0,
            "ref_speech_leak_max_db": -15.0,
            "ref_sensor_noise_db":    -50.0,
            "ref_degraded_prob":      0.05,
        }

        if mode == "pyroom":
            self._pyroom_available = self._check_pyroom()
            if not self._pyroom_available:
                logger.warning(
                    "pyroomacoustics not installed -- falling back to 'filter' mode. "
                    "To enable pyroom mode, run: pip install pyroomacoustics"
                )
                self._mode = "filter"

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def available_modes(cls) -> List[str]:
        """Return the list of valid simulation mode strings.

        Returns:
            List of mode name strings.
        """
        return list(_VALID_MODES)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate(
        self,
        noise: np.ndarray,
        speech: np.ndarray,
        rng: np.random.Generator,
        demand_channels: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Simulate the reference microphone signal.

        Args:
            noise:           Primary mic noise signal, shape ``(N,)`` float32.
            speech:          Clean speech signal, shape ``(N,)`` float32.
            rng:             Numpy random generator for reproducibility.
            demand_channels: ``(ch1, ch2)`` tuple of DEMAND recording channels
                             (required only when ``mode='demand_pair'``).

        Returns:
            Reference signal, shape ``(N,)`` float32.

        Raises:
            ValueError: If ``mode='demand_pair'`` but ``demand_channels`` is
                        ``None``.
        """
        if self._mode == "filter":
            return self._simulate_filter(noise, speech, rng)
        if self._mode == "demand_pair":
            if demand_channels is None:
                raise ValueError(
                    "demand_channels=(ch1, ch2) must be supplied when mode='demand_pair'."
                )
            return self._simulate_demand(noise, speech, rng, demand_channels)
        if self._mode == "pyroom":
            return self._simulate_pyroom(noise, speech, rng)
        # Should never reach here
        return self._simulate_filter(noise, speech, rng)

    def ablation_report(
        self,
        noise: np.ndarray,
        speech: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict[str, Dict[str, float]]:
        """Run all available modes and return SNR / leakage metrics.

        Useful for ablation studies comparing the reference signal quality
        produced by different simulation modes.

        Args:
            noise:  Primary mic noise signal, shape ``(N,)`` float32.
            speech: Clean speech signal, shape ``(N,)`` float32.
            rng:    Numpy random generator.

        Returns:
            Dict mapping mode name -> ``{'ref_snr_db', 'speech_leakage_db'}``.
        """
        report: Dict[str, Dict[str, float]] = {}
        for mode_name in _VALID_MODES:
            try:
                sim = ReferenceSimulator(
                    mode=mode_name,
                    cfg=self._cfg,
                    sr=self._sr,
                    seed=self._seed,
                    cache_dir=str(self._cache_dir),
                )
                ref = sim.simulate(noise, speech, rng)
                report[mode_name] = {
                    "ref_snr_db":       _snr_db(noise, ref),
                    "speech_leakage_db": _snr_db(ref, speech),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ablation_report: mode=%s failed: %s", mode_name, exc
                )
                report[mode_name] = {"ref_snr_db": float("nan"), "speech_leakage_db": float("nan")}
        return report

    # ------------------------------------------------------------------
    # Private: mode implementations
    # ------------------------------------------------------------------

    def _get_mixing_attr(self, attr: str, default=None):
        """Safely retrieve a mixing config attribute.

        Args:
            attr:    Attribute name under ``cfg.mixing``.
            default: Fallback value if cfg is None or attribute is absent.

        Returns:
            The config value or the default.
        """
        if self._cfg is not None:
            try:
                return getattr(self._cfg.mixing, attr)
            except AttributeError:
                pass
        return self._defaults.get(attr, default)

    def _add_leakage_and_sensor_noise(
        self,
        ref: np.ndarray,
        speech: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add speech leakage and sensor noise to a reference signal.

        Args:
            ref:    Reference noise signal, shape ``(N,)`` float32.
            speech: Clean speech, shape ``(>=N,)`` float32.
            rng:    Numpy random generator.

        Returns:
            Modified reference signal, shape ``(N,)`` float32.
        """
        n = len(ref)

        # Speech leakage
        leak_db = float(rng.uniform(
            float(self._get_mixing_attr("ref_speech_leak_min_db", -30.0)),
            float(self._get_mixing_attr("ref_speech_leak_max_db", -15.0)),
        ))
        leak_gain = 10.0 ** (leak_db / 20.0)
        leak_speech = speech[:n] if len(speech) >= n else np.pad(speech, (0, n - len(speech)))
        ref = ref + (leak_gain * leak_speech).astype(np.float32)

        # Sensor noise
        sensor_snr_db = float(self._get_mixing_attr("ref_sensor_noise_db", -50.0))
        ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
        if ref_rms < _NEAR_SILENCE:
            ref_rms = _NEAR_SILENCE
        sensor_std = ref_rms * (10.0 ** (sensor_snr_db / 20.0))
        ref = ref + rng.standard_normal(n).astype(np.float32) * float(sensor_std)

        return ref

    def _maybe_degrade(
        self,
        ref: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Randomly replace reference with wind noise (degraded sensor).

        Args:
            ref: Reference signal, shape ``(N,)`` float32.
            rng: Numpy random generator.

        Returns:
            Possibly replaced reference signal.
        """
        prob = float(self._get_mixing_attr("ref_degraded_prob", 0.05))
        if rng.random() < prob:
            n = len(ref)
            duration_s = n / float(self._sr)
            wind = gen_synthetic_wind(duration_s, self._sr, rng)
            ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))) + _NEAR_SILENCE
            wind_rms = float(np.sqrt(np.mean(wind.astype(np.float64) ** 2))) + _NEAR_SILENCE
            wind = (wind * (ref_rms / wind_rms)).astype(np.float32)
            logger.debug("Reference mic degraded: replaced with synthetic wind.")
            return wind[:n]
        return ref

    def _simulate_filter(
        self,
        noise: np.ndarray,
        speech: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Phase 0 FIR-filter reference simulation.

        Args:
            noise:  Primary noise signal.
            speech: Clean speech signal.
            rng:    Numpy random generator.

        Returns:
            Simulated reference signal, float32.
        """
        if self._cfg is None:
            # Build a minimal mock config so simulate_reference_mic works
            cfg = _MinimalMixingCfg(self._defaults)
        else:
            cfg = self._cfg

        return simulate_reference_mic(noise, speech, self._sr, cfg, rng)

    def _simulate_demand(
        self,
        noise: np.ndarray,
        speech: np.ndarray,
        rng: np.random.Generator,
        demand_channels: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """DEMAND two-channel reference simulation.

        Uses the second channel of a DEMAND multi-microphone recording as
        the reference noise (real spatial decorrelation included).

        Args:
            noise:           Primary noise (first channel, for length reference).
            speech:          Clean speech signal.
            rng:             Numpy random generator.
            demand_channels: ``(primary_ch, reference_ch)`` both float arrays.

        Returns:
            Simulated reference signal, float32.
        """
        n = len(noise)
        _, ref_noise_ch = demand_channels

        # Trim or pad reference channel to match primary length
        ref_noise_ch = ref_noise_ch.astype(np.float32)
        if len(ref_noise_ch) >= n:
            ref = ref_noise_ch[:n].copy()
        else:
            ref = np.pad(ref_noise_ch, (0, n - len(ref_noise_ch))).astype(np.float32)

        ref = self._add_leakage_and_sensor_noise(ref, speech, rng)
        ref = self._maybe_degrade(ref, rng)
        return ref[:n].astype(np.float32)

    def _simulate_pyroom(
        self,
        noise: np.ndarray,
        speech: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Pyroomacoustics shoebox simulation.

        Room layout (all in metres)::

            Reference mic [2.5, 1.5] ........ Noise src [1.5, 0.5]
                                           |
            Primary mic  [0.5, 1.5] ........ Speech src [0.6, 1.5]

        RIRs are cached on disk; subsequent calls with the same room
        parameters are ~instantaneous.

        Args:
            noise:  Primary noise signal (used for duration and as fallback).
            speech: Clean speech signal.
            rng:    Numpy random generator.

        Returns:
            Simulated reference signal, float32.
        """
        import pyroomacoustics as pra  # checked in __init__ already

        n = len(noise)
        duration_s = n / float(self._sr)

        cache_key = _rir_cache_key(
            self._sr, _ROOM_DIM, _ROOM_ABSORPTION, self._seed
        )
        cached = _load_rir_cache(self._cache_dir, cache_key)

        if cached is not None:
            rir_primary, rir_reference = cached
            logger.debug("pyroom: using cached RIRs (key=%s)", cache_key)
        else:
            logger.debug("pyroom: computing RIRs (key=%s) ...", cache_key)
            try:
                materials = pra.Material(_ROOM_ABSORPTION)
                room = pra.ShoeBox(
                    list(_ROOM_DIM),
                    fs=self._sr,
                    materials=materials,
                    max_order=10,
                )
                # Add microphones
                mic_array = np.array([
                    [_PRIMARY_MIC[0],   _REFERENCE_MIC[0]],
                    [_PRIMARY_MIC[1],   _REFERENCE_MIC[1]],
                    [_ROOM_DIM[2] / 2,  _ROOM_DIM[2] / 2],   # z = mid-height
                ], dtype=np.float64)
                room.add_microphone(mic_array)

                # Add sources (placeholder: 1-sample impulse, actual signal applied via conv)
                impulse = np.zeros(int(self._sr * 0.1))
                impulse[0] = 1.0
                room.add_source(
                    [_SPEECH_SRC[0], _SPEECH_SRC[1], _ROOM_DIM[2] / 2],
                    signal=impulse,
                )
                room.add_source(
                    [_NOISE_SRC[0], _NOISE_SRC[1], _ROOM_DIM[2] / 2],
                    signal=impulse,
                )

                room.simulate()
                # RIRs: room.rir[mic_idx][src_idx]
                rir_primary   = room.rir[0][1].astype(np.float32)   # mic0 <- noise src
                rir_reference = room.rir[1][1].astype(np.float32)   # mic1 <- noise src

                _save_rir_cache(self._cache_dir, cache_key, rir_primary, rir_reference)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "pyroom simulation failed: %s. Falling back to filter mode.", exc
                )
                return self._simulate_filter(noise, speech, rng)

        # Convolve noise through reference RIR
        ref_noise = scipy_signal.fftconvolve(
            noise.astype(np.float64),
            rir_reference.astype(np.float64),
        ).astype(np.float32)

        # Trim to original length
        if len(ref_noise) >= n:
            ref = ref_noise[:n]
        else:
            ref = np.pad(ref_noise, (0, n - len(ref_noise))).astype(np.float32)

        ref = self._add_leakage_and_sensor_noise(ref, speech, rng)
        ref = self._maybe_degrade(ref, rng)
        return ref[:n].astype(np.float32)

    # ------------------------------------------------------------------
    # Private: pyroomacoustics availability check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_pyroom() -> bool:
        """Check if pyroomacoustics is importable.

        Returns:
            True if available, False otherwise.
        """
        try:
            import pyroomacoustics  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Minimal config shim (used when cfg is None)
# ---------------------------------------------------------------------------

class _MinimalMixingCfg:
    """Minimal mixing config shim so simulate_reference_mic() works without YAML.

    Args:
        defaults: Dict of mixing parameter values.
    """

    def __init__(self, defaults: Dict) -> None:
        class _Mixing:
            pass

        m = _Mixing()
        for k, v in defaults.items():
            setattr(m, k, v)
        self.mixing = m


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute SNR in dB between *signal* and *noise*.

    Args:
        signal: Signal array.
        noise:  Noise / interference array (same length).

    Returns:
        SNR in dB (float).  Returns NaN if either RMS is zero.
    """
    sig_rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    noi_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    if sig_rms < _NEAR_SILENCE or noi_rms < _NEAR_SILENCE:
        return float("nan")
    return float(20.0 * np.log10(sig_rms / noi_rms))
