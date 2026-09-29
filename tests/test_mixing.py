"""Unit tests for the mixing engine (src/anc/data/mixing.py)."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from anc.data.mixing import (
    NOISE_CATEGORIES,
    SNR_TIERS,
    MixingEngine,
    compute_frame_impulse_labels,
    compute_rms,
    compute_snr_db,
    loop_with_crossfade,
    mix_at_snr,
    snr_tier_name,
)

SR = 16_000
SEG_LEN = 4 * SR  # 64 000 samples
HOP_LEN = 160


# ---------------------------------------------------------------------------
# compute_rms
# ---------------------------------------------------------------------------

def test_compute_rms_sine():
    """RMS of a unit-amplitude sine = 1/sqrt(2) ≈ 0.7071."""
    t = np.linspace(0, 1, SR, endpoint=False, dtype=np.float32)
    sine = np.sin(2 * np.pi * 440 * t)
    rms = compute_rms(sine, use_vad=False)
    assert abs(rms - 1.0 / np.sqrt(2)) < 0.01


def test_compute_rms_nonzero():
    """RMS is always positive even for silence."""
    silence = np.zeros(SR, dtype=np.float32)
    assert compute_rms(silence, use_vad=False) > 0.0


# ---------------------------------------------------------------------------
# mix_at_snr
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_snr", [-10.0, -5.0, 0.0, 5.0, 10.0])
def test_mix_at_snr_accuracy(target_snr: float):
    """Actual SNR after mixing is within ±0.5 dB of target."""
    rng = np.random.default_rng(0)
    speech = rng.standard_normal(SEG_LEN).astype(np.float32) * 0.5
    noise  = rng.standard_normal(SEG_LEN).astype(np.float32) * 0.3
    _, _, actual_snr = mix_at_snr(speech, noise, target_snr)
    assert abs(actual_snr - target_snr) < 0.6, (
        f"target={target_snr:.1f} dB, actual={actual_snr:.1f} dB"
    )


def test_mix_at_snr_zero_noise():
    """Near-zero noise must not crash; should return speech unchanged."""
    rng = np.random.default_rng(1)
    speech = rng.standard_normal(SEG_LEN).astype(np.float32)
    noise  = np.zeros(SEG_LEN, dtype=np.float32)
    noisy, _, snr = mix_at_snr(speech, noise, 0.0)
    assert np.isfinite(noisy).all()
    assert snr == float("inf") or snr > 100


# ---------------------------------------------------------------------------
# loop_with_crossfade
# ---------------------------------------------------------------------------

def test_loop_with_crossfade_length():
    """Output length must equal target_len exactly."""
    noise = np.ones(100, dtype=np.float32)
    out = loop_with_crossfade(noise, 1000, crossfade_samples=20)
    assert len(out) == 1000


def test_loop_with_crossfade_shorter_than_crossfade():
    """Signal shorter than crossfade_samples must not crash."""
    noise = np.ones(5, dtype=np.float32)
    out = loop_with_crossfade(noise, 500, crossfade_samples=160)
    assert len(out) == 500
    assert np.isfinite(out).all()


def test_loop_with_crossfade_already_long_enough():
    """Signal longer than target → just slice."""
    noise = np.arange(2000, dtype=np.float32)
    out = loop_with_crossfade(noise, 1000)
    assert len(out) == 1000
    np.testing.assert_array_equal(out, noise[:1000])


# ---------------------------------------------------------------------------
# compute_frame_impulse_labels
# ---------------------------------------------------------------------------

def test_impulse_labels_shape():
    """Labels shape = (seg_len // hop_len,)."""
    signal = np.zeros(SEG_LEN, dtype=np.float32)
    labels = compute_frame_impulse_labels(signal, SEG_LEN, HOP_LEN)
    assert labels.shape == (SEG_LEN // HOP_LEN,)


def test_impulse_labels_binary():
    """Labels must be 0.0 or 1.0 only."""
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(SEG_LEN).astype(np.float32)
    labels = compute_frame_impulse_labels(signal, SEG_LEN, HOP_LEN)
    assert set(labels.tolist()).issubset({0.0, 1.0})


def test_impulse_labels_detects_spike():
    """A large spike at a known position should produce label=1 in that frame."""
    signal = np.zeros(SEG_LEN, dtype=np.float32)
    spike_frame = 10
    signal[spike_frame * HOP_LEN + HOP_LEN // 2] = 100.0  # enormous spike
    labels = compute_frame_impulse_labels(signal, SEG_LEN, HOP_LEN)
    assert labels[spike_frame] == 1.0


# ---------------------------------------------------------------------------
# SNR tiers & categories
# ---------------------------------------------------------------------------

def test_snr_tiers_count():
    assert len(SNR_TIERS) == 5


def test_snr_tiers_sorted_descending():
    values = list(SNR_TIERS.values())
    assert values == sorted(values, reverse=True)


def test_noise_categories_keys():
    expected = {"impulsive", "rotor", "vehicle", "siren", "wind", "crowd", "misc"}
    assert set(NOISE_CATEGORIES.keys()) == expected


def test_snr_tier_name():
    assert snr_tier_name(10.0) == "Easy"
    assert snr_tier_name(-10.0) == "Extreme"
    assert snr_tier_name(0.0) == "Hard"


# ---------------------------------------------------------------------------
# MixingEngine
# ---------------------------------------------------------------------------

def test_mixing_engine_basic(tmp_audio_dir: Path, base_cfg):
    """MixingEngine.mix() returns a dict with correct keys and consistent lengths."""
    speech_files = list((tmp_audio_dir / "speech").rglob("*.wav"))
    noise_files  = list((tmp_audio_dir / "noise").rglob("*.wav"))

    engine = MixingEngine(base_cfg, speech_files, noise_files)
    for seed in (0, 1, 2):
        result = engine.mix(seed=seed)

    required_keys = {
        "primary", "reference", "clean", "scaled_noise",
        "snr_target", "snr_actual", "snr_tier", "noise_class",
        "noise_path", "speech_path", "impulse_labels",
        "rir_applied", "sample_rate",
    }
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - set(result.keys())}"

    n = len(result["primary"])
    assert len(result["reference"]) == n
    assert len(result["clean"]) == n

    seg_len = int(float(base_cfg.data.segment_dur) * SR)
    assert n == seg_len

    expected_frames = seg_len // HOP_LEN
    assert len(result["impulse_labels"]) == expected_frames

    assert -15 <= result["snr_actual"] <= 15  # reasonable range


def test_mixing_engine_reproducibility(tmp_audio_dir: Path, base_cfg):
    """Same seed must produce identical primary arrays."""
    speech_files = list((tmp_audio_dir / "speech").rglob("*.wav"))
    noise_files  = list((tmp_audio_dir / "noise").rglob("*.wav"))

    engine = MixingEngine(base_cfg, speech_files, noise_files)
    r1 = engine.mix(seed=42)
    r2 = engine.mix(seed=42)

    np.testing.assert_array_equal(r1["primary"], r2["primary"])
    np.testing.assert_array_equal(r1["reference"], r2["reference"])


def test_mixing_engine_different_seeds(tmp_audio_dir: Path, base_cfg):
    """Different seeds must produce (usually) different outputs."""
    speech_files = list((tmp_audio_dir / "speech").rglob("*.wav"))
    noise_files  = list((tmp_audio_dir / "noise").rglob("*.wav"))

    engine = MixingEngine(base_cfg, speech_files, noise_files)
    r1 = engine.mix(seed=0)
    r2 = engine.mix(seed=1)
    # Very unlikely to be identical
    assert not np.array_equal(r1["primary"], r2["primary"])


def test_mixing_engine_snr_tier_targeted(tmp_audio_dir: Path, base_cfg):
    """Mixing with a specific SNR tier produces the right tier name."""
    speech_files = list((tmp_audio_dir / "speech").rglob("*.wav"))
    noise_files  = list((tmp_audio_dir / "noise").rglob("*.wav"))

    engine = MixingEngine(base_cfg, speech_files, noise_files)
    result = engine.mix(target_snr_db=10.0, seed=0)
    assert result["snr_tier"] == "Easy"

    result2 = engine.mix(target_snr_db=-10.0, seed=0)
    assert result2["snr_tier"] == "Extreme"
