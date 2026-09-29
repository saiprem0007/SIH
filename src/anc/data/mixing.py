"""
Mixing engine for the ANC-SIH data pipeline.

Generates (primary mic, reference mic, clean speech) triplets at a target SNR
by mixing clean speech with defence noise, optionally inserting impulsive events,
applying room impulse responses, and simulating a reference microphone channel.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from anc.data.augmentation import (
    apply_rir,
    hard_clip,
    load_audio,
    load_rir,
    peak_normalize,
    random_gain,
)
from anc.data.reference_mic import simulate_reference_mic
from anc.data.synthetic_noise import SYNTHETIC_GENERATORS, gen_synthetic_noise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NOISE_CATEGORIES: Dict[str, List[str]] = {
    "impulsive": ["gunshot", "gun_shot", "explosion", "blast", "artillery", "shot", "firearm"],
    "rotor":     ["helicopter", "drone", "rotor", "propeller"],
    "vehicle":   ["engine", "vehicle", "car", "truck", "idling", "engine_idling"],
    "siren":     ["siren", "alarm", "emergency", "horn"],
    "wind":      ["wind", "air", "breeze"],
    "crowd":     ["crowd", "urban", "street", "jackhammer", "drilling", "construction"],
    "misc":      [],
}

SNR_TIERS: Dict[str, float] = {
    "Easy":     10.0,
    "Moderate":  5.0,
    "Hard":      0.0,
    "VeryHard": -5.0,
    "Extreme":  -10.0,
}

_VAD_FRAME_LEN: int = 160          # 10 ms at 16 kHz
_VAD_THRESHOLD_DB: float = -40.0   # frames 40 dB below global energy are silence
_MIN_RMS: float = 1e-10


# ---------------------------------------------------------------------------
# DSP utilities
# ---------------------------------------------------------------------------

def compute_rms(
    signal: np.ndarray,
    use_vad: bool = True,
    vad_threshold_db: float = _VAD_THRESHOLD_DB,
) -> float:
    """Compute the RMS of a signal, optionally over active (voiced) frames only.

    Args:
        signal: 1-D float array.
        use_vad: If True, exclude near-silent frames before computing RMS.
        vad_threshold_db: Frames whose energy is more than this many dB below
            the global energy are treated as silence (inactive).

    Returns:
        RMS value (always >= _MIN_RMS).
    """
    signal = np.asarray(signal, dtype=np.float64)
    global_energy = float(np.mean(signal ** 2))

    if not use_vad or global_energy < 1e-20:
        return float(max(np.sqrt(global_energy), _MIN_RMS))

    # Trim to whole frames
    n_frames = len(signal) // _VAD_FRAME_LEN
    if n_frames == 0:
        return float(max(np.sqrt(global_energy), _MIN_RMS))

    frames = signal[: n_frames * _VAD_FRAME_LEN].reshape(n_frames, _VAD_FRAME_LEN)
    frame_energy = np.mean(frames ** 2, axis=1)
    threshold = global_energy * (10.0 ** (vad_threshold_db / 10.0))
    active = frames[frame_energy > max(threshold, 1e-20)]

    if len(active) == 0:
        active = frames  # fall back to all frames

    return float(max(np.sqrt(np.mean(active ** 2)), _MIN_RMS))


def compute_snr_db(
    speech: np.ndarray,
    noise: np.ndarray,
    use_vad: bool = True,
) -> float:
    """Compute SNR in dB between speech and noise.

    Args:
        speech: Clean speech signal.
        noise:  Noise signal (same length).
        use_vad: Use VAD-gated RMS for speech.

    Returns:
        SNR in dB, or +inf if noise is near-zero.
    """
    s_rms = compute_rms(speech, use_vad=use_vad)
    n_rms = compute_rms(noise, use_vad=False)
    if n_rms < _MIN_RMS * 10:
        return float("inf")
    return float(20.0 * np.log10(s_rms / n_rms))


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
    use_vad: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale noise to achieve the target SNR then add to speech.

    Args:
        speech: Clean speech signal, shape (N,).
        noise:  Noise signal, shape (N,).
        target_snr_db: Desired SNR in dB.
        use_vad: Use VAD-gated RMS for speech level estimation.

    Returns:
        Tuple of:
            noisy      -- speech + scaled_noise, shape (N,)
            scaled_noise -- the noise portion only, shape (N,)
            actual_snr_db -- measured SNR after mixing
    """
    s_rms = compute_rms(speech, use_vad=use_vad)
    n_rms = compute_rms(noise, use_vad=False)

    if n_rms < _MIN_RMS * 10:
        logger.warning("Noise RMS near zero — returning speech unchanged.")
        return speech.copy().astype(np.float32), noise.copy().astype(np.float32), float("inf")

    desired_n_rms = s_rms / (10.0 ** (target_snr_db / 20.0))
    scale = desired_n_rms / n_rms
    scaled_noise = (noise * scale).astype(np.float32)
    noisy = (speech.astype(np.float32) + scaled_noise)
    actual_snr = compute_snr_db(speech, scaled_noise, use_vad=use_vad)

    if abs(actual_snr - target_snr_db) > 0.5:
        logger.warning(
            "SNR mismatch: target=%.1f dB, actual=%.1f dB", target_snr_db, actual_snr
        )
    return noisy, scaled_noise, actual_snr


def loop_with_crossfade(
    noise: np.ndarray,
    target_len: int,
    crossfade_samples: int = 160,
) -> np.ndarray:
    """Loop *noise* to *target_len* samples, using a linear crossfade at boundaries.

    Args:
        noise: Input noise array, 1-D float.
        target_len: Desired output length in samples.
        crossfade_samples: Length of the crossfade region (samples).

    Returns:
        Looped noise of shape (target_len,) as float32.
    """
    if len(noise) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(noise) >= target_len:
        return noise[:target_len].astype(np.float32)

    n = len(noise)
    cf = min(crossfade_samples, n // 4, 1)  # at least 1
    out = np.zeros(target_len, dtype=np.float64)
    pos = 0

    while pos < target_len:
        remaining = target_len - pos
        to_copy = min(n, remaining)
        out[pos: pos + to_copy] += noise[:to_copy].astype(np.float64)
        # Crossfade overlap with previous loop end
        if pos > 0 and cf > 1:
            seg_cf = min(cf, to_copy)
            fade_in = np.linspace(0.0, 1.0, seg_cf)
            fade_out = np.linspace(1.0, 0.0, seg_cf)
            # The last `seg_cf` samples already written from previous loop
            prev_start = pos - seg_cf
            if prev_start >= 0:
                out[pos: pos + seg_cf] = (
                    out[pos: pos + seg_cf] * fade_in
                    + out[prev_start: prev_start + seg_cf] * fade_out
                )
        pos += to_copy

    return out[:target_len].astype(np.float32)


def compute_frame_impulse_labels(
    impulse_signal: np.ndarray,
    total_signal_len: int,
    hop_len: int,
    peak_threshold_factor: float = 5.0,
) -> np.ndarray:
    """Compute per-frame binary impulse labels.

    A frame is labelled 1 if its peak amplitude exceeds
    ``peak_threshold_factor * global_rms`` of the impulse signal.

    Args:
        impulse_signal: The impulse-only component, shape (total_signal_len,).
            May be shorter than total_signal_len; zero-padded internally.
        total_signal_len: Total length of the mixture signal.
        hop_len: Frame hop length in samples.
        peak_threshold_factor: Peak-to-RMS ratio threshold.

    Returns:
        Binary label array of shape (n_frames,) as float32.
    """
    n_frames = total_signal_len // hop_len
    labels = np.zeros(n_frames, dtype=np.float32)

    if impulse_signal is None or len(impulse_signal) == 0:
        return labels

    # Pad/trim to total_signal_len
    if len(impulse_signal) < total_signal_len:
        imp = np.zeros(total_signal_len, dtype=np.float64)
        imp[: len(impulse_signal)] = impulse_signal
    else:
        imp = impulse_signal[:total_signal_len].astype(np.float64)

    rms = float(np.sqrt(np.mean(imp ** 2)) + 1e-10)
    threshold = peak_threshold_factor * rms

    for i in range(n_frames):
        frame = imp[i * hop_len: (i + 1) * hop_len]
        if np.max(np.abs(frame)) > threshold:
            labels[i] = 1.0

    return labels


def snr_tier_name(snr_db: float) -> str:
    """Return the name of the nearest SNR tier.

    Args:
        snr_db: SNR value in dB.

    Returns:
        One of 'Easy', 'Moderate', 'Hard', 'VeryHard', 'Extreme'.
    """
    closest = min(SNR_TIERS, key=lambda k: abs(SNR_TIERS[k] - snr_db))
    return closest


# ---------------------------------------------------------------------------
# MixingEngine
# ---------------------------------------------------------------------------

class MixingEngine:
    """Generates (primary, reference, clean) triplets on-the-fly.

    The engine draws speech and noise files at random, mixes them at a target
    SNR, optionally stacks multiple noise sources, inserts impulsive events,
    applies RIR augmentation, and simulates the reference microphone channel.

    Args:
        cfg: Config object with audio.*, data.*, and mixing.* sections.
        speech_files: List of clean speech file paths.
        noise_files: List of noise file paths.
        rir_files: Optional list of room impulse response file paths.
        impulse_files: Optional list of impulsive noise file paths.
            If None or empty, synthetic blasts are used as fallback.
    """

    def __init__(
        self,
        cfg,
        speech_files: List[Path],
        noise_files: List[Path],
        rir_files: Optional[List[Path]] = None,
        impulse_files: Optional[List[Path]] = None,
    ) -> None:
        if not speech_files:
            raise ValueError("speech_files must not be empty.")
        if not noise_files:
            raise ValueError("noise_files must not be empty.")

        self.cfg = cfg
        self.speech_files = [Path(p) for p in speech_files]
        self.noise_files = [Path(p) for p in noise_files]
        self.rir_files = [Path(p) for p in rir_files] if rir_files else []
        self.impulse_files = [Path(p) for p in impulse_files] if impulse_files else []

        self.sr: int = int(cfg.audio.sample_rate)
        self.seg_len: int = int(float(cfg.data.segment_dur) * self.sr)
        self.hop_len: int = int(cfg.audio.hop_len)

        logger.info(
            "MixingEngine ready — %d speech, %d noise, %d RIR, %d impulse files.",
            len(self.speech_files),
            len(self.noise_files),
            len(self.rir_files),
            len(self.impulse_files),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_crop(self, path: Path, rng: np.random.Generator) -> np.ndarray:
        """Load audio file and random-crop to self.seg_len."""
        audio = load_audio(path, self.sr)
        if len(audio) >= self.seg_len:
            start = rng.integers(0, len(audio) - self.seg_len + 1)
            return audio[start: start + self.seg_len].astype(np.float32)
        # Shorter than seg_len: loop with crossfade
        return loop_with_crossfade(audio, self.seg_len, crossfade_samples=int(self.cfg.mixing.crossfade_ms * self.sr / 1000))

    def _pick_noise_file(
        self, rng: np.random.Generator, category: Optional[str] = None
    ) -> Tuple[Path, str]:
        """Pick a random noise file, optionally filtered by category.

        Returns:
            (path, detected_category)
        """
        if category is not None:
            keywords = NOISE_CATEGORIES.get(category, [])
            filtered = [
                p for p in self.noise_files
                if any(kw in p.stem.lower() or kw in str(p).lower() for kw in keywords)
            ] if keywords else []
            candidates = filtered if filtered else self.noise_files
        else:
            candidates = self.noise_files

        path = candidates[rng.integers(0, len(candidates))]
        # Detect category from filename
        p_str = str(path).lower()
        detected = "misc"
        for cat, kws in NOISE_CATEGORIES.items():
            if any(kw in p_str for kw in kws):
                detected = cat
                break
        return path, detected

    def _build_noise(
        self,
        rng: np.random.Generator,
        category: Optional[str],
    ) -> Tuple[np.ndarray, str, str]:
        """Build the noise signal (possibly stacked from 2 sources).

        Returns:
            (noise_array, noise_class, noise_path_str)
        """
        noise_path, noise_class = self._pick_noise_file(rng, category)
        noise = self._load_crop(noise_path, rng)

        # Noise stacking
        if rng.random() < float(self.cfg.mixing.stack_prob):
            noise2_path, _ = self._pick_noise_file(rng, category=None)
            noise2 = self._load_crop(noise2_path, rng)
            offset_db = float(self.cfg.mixing.stack_snr_offset_db)
            scale2 = 10.0 ** (-offset_db / 20.0)
            noise = noise + (noise2 * scale2).astype(np.float32)

        return noise, noise_class, str(noise_path)

    def _insert_impulse_events(
        self,
        base_noise: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Insert 0–max_impulse_events impulsive clips into the noise.

        Returns:
            (noise_with_impulses, impulse_only_signal)
        """
        n_events = int(rng.integers(
            int(self.cfg.mixing.min_impulse_events),
            int(self.cfg.mixing.max_impulse_events) + 1,
        ))
        impulse_track = np.zeros(self.seg_len, dtype=np.float32)

        for _ in range(n_events):
            if self.impulse_files:
                imp_path = self.impulse_files[rng.integers(0, len(self.impulse_files))]
                imp = self._load_crop(imp_path, rng)
            else:
                imp = gen_synthetic_noise("impulsive", float(self.cfg.data.segment_dur), self.sr, seed=int(rng.integers(0, 2**31)))

            # Pick random placement
            imp_len = min(len(imp), self.seg_len)
            onset = int(rng.integers(0, max(1, self.seg_len - imp_len)))

            # Scale impulse
            imp_snr = rng.uniform(
                float(self.cfg.mixing.impulse_snr_min),
                float(self.cfg.mixing.impulse_snr_max),
            )
            s_rms = compute_rms(base_noise, use_vad=False)
            i_rms = compute_rms(imp[:imp_len], use_vad=False)
            if i_rms > _MIN_RMS:
                desired_rms = s_rms * (10.0 ** (imp_snr / 20.0))
                scale = desired_rms / i_rms
                imp = (imp * scale).astype(np.float32)

            base_noise[onset: onset + imp_len] += imp[:imp_len]
            impulse_track[onset: onset + imp_len] += imp[:imp_len]

        return base_noise, impulse_track

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mix(
        self,
        target_snr_db: Optional[float] = None,
        noise_category: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate one (primary, reference, clean) mixture.

        Args:
            target_snr_db: Target SNR in dB. Sampled from config range if None.
            noise_category: Force a specific noise category, or None for random.
            seed: Integer seed for full reproducibility. Uses OS entropy if None.

        Returns:
            Dictionary with keys:
                primary (np.ndarray): Noisy primary mic, shape (N,).
                reference (np.ndarray): Reference mic, shape (N,).
                clean (np.ndarray): Clean speech target, shape (N,).
                scaled_noise (np.ndarray): Noise component in primary, shape (N,).
                snr_target (float): The target SNR used.
                snr_actual (float): Measured SNR of the mixture.
                snr_tier (str): Nearest SNR tier name.
                noise_class (str): Noise category detected.
                noise_path (str): Path of noise file used.
                speech_path (str): Path of speech file used.
                impulse_labels (np.ndarray): Binary frame labels, shape (n_frames,).
                rir_applied (bool): Whether RIR was applied.
                sample_rate (int): 16000.
        """
        rng = np.random.default_rng(seed)

        # 1. Sample target SNR
        if target_snr_db is None:
            target_snr_db = float(rng.uniform(
                float(self.cfg.data.snr_min), float(self.cfg.data.snr_max)
            ))

        # 2. Load speech
        speech_path = self.speech_files[rng.integers(0, len(self.speech_files))]
        clean = self._load_crop(speech_path, rng)

        # 3. Build noise (possibly stacked)
        noise, noise_class, noise_path = self._build_noise(rng, noise_category)

        # 4. Apply RIR to speech
        rir_applied = False
        if self.rir_files and rng.random() < float(self.cfg.mixing.rir_prob):
            rir = self.rir_files[rng.integers(0, len(self.rir_files))]
            rir_wav = load_audio(rir, self.sr)
            clean = apply_rir(clean, rir_wav)
            rir_applied = True
        # Optionally apply RIR to noise
        if self.rir_files and rng.random() < float(self.cfg.mixing.noise_rir_prob):
            rir = self.rir_files[rng.integers(0, len(self.rir_files))]
            rir_wav = load_audio(rir, self.sr)
            noise = apply_rir(noise, rir_wav)

        # 5. Insert impulsive events
        impulse_track = np.zeros(self.seg_len, dtype=np.float32)
        if rng.random() < float(self.cfg.mixing.impulse_prob):
            noise, impulse_track = self._insert_impulse_events(noise, rng)

        # 6. Mix at target SNR
        primary, scaled_noise, snr_actual = mix_at_snr(clean, noise, target_snr_db)

        # 7. Gain augmentation
        if rng.random() < float(self.cfg.mixing.gain_aug_prob):
            primary = random_gain(
                primary,
                float(self.cfg.mixing.gain_min_db),
                float(self.cfg.mixing.gain_max_db),
                rng,
            )

        # 8. Hard clipping (mic saturation)
        if rng.random() < float(self.cfg.mixing.clip_prob):
            primary = hard_clip(primary, float(self.cfg.mixing.clip_level))

        # 9. Simulate reference microphone
        reference = simulate_reference_mic(noise, clean, self.sr, self.cfg, rng)

        # 10. Ensure all arrays are the same length
        primary    = primary[:self.seg_len].astype(np.float32)
        reference  = reference[:self.seg_len].astype(np.float32)
        clean      = clean[:self.seg_len].astype(np.float32)
        scaled_noise = scaled_noise[:self.seg_len].astype(np.float32)

        # 11. Compute frame-level impulse labels
        impulse_labels = compute_frame_impulse_labels(
            impulse_track,
            self.seg_len,
            self.hop_len,
            float(self.cfg.mixing.impulse_peak_threshold_factor),
        )

        return {
            "primary":       primary,
            "reference":     reference,
            "clean":         clean,
            "scaled_noise":  scaled_noise,
            "snr_target":    float(target_snr_db),
            "snr_actual":    float(snr_actual),
            "snr_tier":      snr_tier_name(target_snr_db),
            "noise_class":   noise_class,
            "noise_path":    noise_path,
            "speech_path":   str(speech_path),
            "impulse_labels": impulse_labels,
            "rir_applied":   rir_applied,
            "sample_rate":   self.sr,
        }
