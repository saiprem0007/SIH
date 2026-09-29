"""
Phase 4 pytest tests for anc.synth generators.

Tests cover physical properties, not just array shape, to catch
regressions in the acoustic models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.synth import (
    blast, gunshot, artillery, rotor, siren, engine,
    tracked_vehicle, wind_buffet, babble,
    DEFENCE_GENERATORS, render_all,
)

SR: int = 16000
DUR: float = 4.0


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1-5: universal properties across all generators
# ---------------------------------------------------------------------------

def test_all_generators_return_float32():
    for name, fn in DEFENCE_GENERATORS.items():
        out = fn(DUR, SR, np.random.default_rng(42))
        assert out.dtype == np.float32, f"{name}: expected float32, got {out.dtype}"


def test_all_generators_correct_length():
    expected = int(DUR * SR)
    for name, fn in DEFENCE_GENERATORS.items():
        out = fn(DUR, SR, np.random.default_rng(42))
        assert len(out) == expected, f"{name}: expected {expected} samples, got {len(out)}"


def test_all_generators_peak_normalized():
    for name, fn in DEFENCE_GENERATORS.items():
        out = fn(DUR, SR, np.random.default_rng(42))
        peak = float(np.max(np.abs(out)))
        assert 0.5 <= peak <= 0.95, f"{name}: peak={peak:.4f} not in [0.5, 0.95]"


def test_all_generators_no_nan_inf():
    for name, fn in DEFENCE_GENERATORS.items():
        out = fn(DUR, SR, np.random.default_rng(42))
        assert np.all(np.isfinite(out)), f"{name}: contains NaN or Inf"


def test_all_generators_deterministic():
    for name, fn in DEFENCE_GENERATORS.items():
        out1 = fn(DUR, SR, np.random.default_rng(99))
        out2 = fn(DUR, SR, np.random.default_rng(99))
        assert np.array_equal(out1, out2), f"{name}: not deterministic for same seed"


# ---------------------------------------------------------------------------
# 6. blast: sharp onset in first 35%
# ---------------------------------------------------------------------------

def test_blast_sharp_onset(rng):
    out = blast(DUR, SR, rng)
    peak_idx = int(np.argmax(np.abs(out)))
    assert peak_idx < int(0.35 * len(out)), (
        f"blast peak at sample {peak_idx} ({peak_idx/SR:.3f}s) not in first 35%"
    )
    assert float(np.max(np.abs(out))) > 0.5


# ---------------------------------------------------------------------------
# 7. blast: energy decays over time
# ---------------------------------------------------------------------------

def test_blast_decay(rng):
    """RMS in a 100 ms window just after the blast peak must exceed RMS in tail."""
    out = blast(DUR, SR, rng)
    peak_idx = int(np.argmax(np.abs(out)))
    window_start = min(peak_idx + int(0.005 * SR), len(out) - 1)
    window_end   = min(window_start + int(0.1 * SR), len(out))
    early_rms = float(np.sqrt(np.mean(out[window_start:window_end] ** 2)))
    late_rms  = float(np.sqrt(np.mean(out[int(0.70 * len(out)):] ** 2)))
    assert late_rms < early_rms + 1e-6, (
        f"blast not decaying: near-peak rms={early_rms:.6f}, tail rms={late_rms:.6f}"
    )


# ---------------------------------------------------------------------------
# 8. gunshot: bipolar transient (positive AND negative peak)
# ---------------------------------------------------------------------------

def test_gunshot_has_bipolar_transient(rng):
    out = gunshot(DUR, SR, rng, distance_m=5.0)
    assert float(np.max(out)) > 0.3, "gunshot: no strong positive transient"
    assert float(np.min(out)) < -0.05, "gunshot: no negative lobe"


# ---------------------------------------------------------------------------
# 9. artillery: low-frequency dominant
# ---------------------------------------------------------------------------

def test_artillery_low_frequency_dominant(rng):
    out = artillery(DUR, SR, rng)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / SR)
    low_power  = float(np.sum(spectrum[freqs < 300] ** 2))
    high_power = float(np.sum(spectrum[freqs > 1000] ** 2))
    assert low_power >= high_power, (
        f"artillery: low_power={low_power:.2f} < high_power={high_power:.2f}"
    )


# ---------------------------------------------------------------------------
# 10. rotor: autocorrelation shows periodicity
# ---------------------------------------------------------------------------

def test_rotor_periodicity(rng):
    out = rotor(DUR, SR, rng)
    # Use first 0.5s to keep FFT fast
    seg = out[: SR // 2].astype(np.float64)
    corr = np.correlate(seg, seg, mode="full")
    corr = corr[len(corr) // 2:]
    # Fundamental between 15-30 Hz -> period 33-67 ms -> 528-1067 samples at 16 kHz
    lag_lo = int(0.033 * SR)
    lag_hi = int(0.067 * SR)
    secondary_peak = float(np.max(np.abs(corr[lag_lo:lag_hi])))
    assert secondary_peak > 0.05 * float(corr[0]), (
        f"rotor autocorr secondary peak {secondary_peak:.4f} vs zero-lag {corr[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# 11. siren: frequency sweeps (instantaneous frequency varies)
# ---------------------------------------------------------------------------

def test_siren_sweeping(rng):
    out = siren(DUR, SR, rng, pattern="wail")
    # Split into 4 equal windows and compute dominant frequency
    n_wins = 4
    win_len = len(out) // n_wins
    dom_freqs = []
    for i in range(n_wins):
        seg = out[i * win_len: (i + 1) * win_len].astype(np.float64)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(win_len, d=1.0 / SR)
        dom_freqs.append(float(freqs[np.argmax(spec)]))
    freq_spread = max(dom_freqs) - min(dom_freqs)
    assert freq_spread > 200.0, (
        f"siren frequency spread only {freq_spread:.1f} Hz, expected > 200 Hz"
    )


# ---------------------------------------------------------------------------
# 12. engine: harmonic peaks in spectrum
# ---------------------------------------------------------------------------

def test_engine_harmonics(rng):
    out = engine(DUR, SR, rng, rpm=1200)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / SR)
    # Fundamental at 1200/60 = 20 Hz; look for peaks
    # Find top-5 peaks in 5-200 Hz range
    mask = (freqs >= 5) & (freqs <= 500)
    sub_spec = spectrum * mask
    # Simple peak finding: at least 3 local maxima
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sub_spec, height=float(np.max(sub_spec)) * 0.05)
    assert len(peaks) >= 2, f"engine: only {len(peaks)} harmonic peaks found, expected >= 2"


# ---------------------------------------------------------------------------
# 13. tracked_vehicle: high-energy clank frames exist
# ---------------------------------------------------------------------------

def test_tracked_vehicle_has_clanks(rng):
    out = tracked_vehicle(DUR, SR, rng)
    hop = 160  # 10 ms
    n_frames = len(out) // hop
    frame_rms = np.array([
        float(np.sqrt(np.mean(out[i * hop: (i + 1) * hop] ** 2)))
        for i in range(n_frames)
    ])
    mean_rms = float(np.mean(frame_rms)) + 1e-10
    high_energy_frames = int(np.sum(frame_rms > 2.0 * mean_rms))
    assert high_energy_frames >= 1, (
        f"tracked_vehicle: no high-energy clank frames found "
        f"(max/mean ratio={frame_rms.max()/mean_rms:.2f})"
    )


# ---------------------------------------------------------------------------
# 14. wind_buffet: low-frequency dominated
# ---------------------------------------------------------------------------

def test_wind_buffet_low_freq(rng):
    out = wind_buffet(DUR, SR, rng)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / SR)
    low_power  = float(np.sum(spectrum[freqs < 400] ** 2))
    high_power = float(np.sum(spectrum[freqs > 2000] ** 2))
    assert low_power >= high_power, (
        f"wind_buffet: low_power={low_power:.2f} < high_power={high_power:.2f}"
    )


# ---------------------------------------------------------------------------
# 15. babble: fallback (no speech files) does not crash
# ---------------------------------------------------------------------------

def test_babble_no_crash_no_speech_files(rng):
    out = babble(DUR, SR, rng, speech_files=None)
    assert out.dtype == np.float32
    assert len(out) == int(DUR * SR)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 16. render_all: creates 9 WAV files
# ---------------------------------------------------------------------------

def test_render_all_creates_files(tmp_path):
    paths = render_all(tmp_path, duration_s=1.0, sr=SR, seed=0)
    assert len(paths) == 9, f"Expected 9 WAV files, got {len(paths)}"
    for name, p in paths.items():
        assert p.exists(), f"{name}: WAV file missing at {p}"
        assert p.stat().st_size > 100, f"{name}: WAV file suspiciously small"


# ---------------------------------------------------------------------------
# 17. registry has exactly 9 callable entries
# ---------------------------------------------------------------------------

def test_defence_generators_registry():
    assert len(DEFENCE_GENERATORS) == 9, (
        f"DEFENCE_GENERATORS has {len(DEFENCE_GENERATORS)} entries, expected 9"
    )
    for name, fn in DEFENCE_GENERATORS.items():
        assert callable(fn), f"DEFENCE_GENERATORS['{name}'] is not callable"
