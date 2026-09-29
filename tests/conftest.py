"""
Shared pytest fixtures for ANC-SIH unit tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, lfilter

SR = 16000
FRAME_LEN = 320   # 20 ms
HOP_LEN = 160     # 10 ms
SEG_LEN = 4 * SR  # 4 seconds = 64 000 samples


# ---------------------------------------------------------------------------
# Basic signal fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def clean_speech(rng: np.random.Generator) -> np.ndarray:
    """Synthetic speech-like signal: bandpass-filtered noise (300–3400 Hz)."""
    white = rng.standard_normal(SEG_LEN).astype(np.float64)
    nyq = SR / 2.0
    b, a = butter(4, [300 / nyq, 3400 / nyq], btype="band")
    filtered = lfilter(b, a, white).astype(np.float32)
    peak = np.max(np.abs(filtered))
    if peak > 1e-8:
        filtered = (filtered * (0.9 / peak))
    return filtered.astype(np.float32)


@pytest.fixture
def noise_signal(rng: np.random.Generator) -> np.ndarray:
    """Synthetic white noise signal, same length as clean_speech."""
    noise = rng.standard_normal(SEG_LEN).astype(np.float32)
    peak = np.max(np.abs(noise))
    return (noise * (0.5 / peak)).astype(np.float32)


@pytest.fixture
def noisy_mixture(clean_speech: np.ndarray, noise_signal: np.ndarray):
    """0 dB SNR mixture."""
    from anc.data.mixing import mix_at_snr
    noisy, scaled_noise, actual_snr = mix_at_snr(clean_speech, noise_signal, 0.0)
    return noisy, clean_speech, scaled_noise, actual_snr


# ---------------------------------------------------------------------------
# Temporary audio directory (5 speech + 5 noise files)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_audio_dir(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Temp directory with 5 speech and 5 noise wav files."""
    speech_dir = tmp_path / "speech" / "spk001"
    noise_dir  = tmp_path / "noise"
    speech_dir.mkdir(parents=True)
    noise_dir.mkdir(parents=True)

    nyq = SR / 2.0
    b, a = butter(4, [300 / nyq, 3400 / nyq], btype="band")

    for i in range(5):
        # Speech-like
        sig = lfilter(b, a, rng.standard_normal(SEG_LEN)).astype(np.float32)
        sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.8
        sf.write(str(speech_dir / f"utt{i:04d}.wav"), sig, SR, subtype="PCM_16")

        # Noise (engine-like label in filename)
        nsig = rng.standard_normal(SEG_LEN).astype(np.float32) * 0.5
        sf.write(str(noise_dir / f"engine_{i:04d}.wav"), nsig, SR, subtype="PCM_16")

    return tmp_path


# ---------------------------------------------------------------------------
# Base config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def base_cfg():
    """Load the base YAML config."""
    from anc.utils.config import load_config
    cfg_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    return load_config(cfg_path)
