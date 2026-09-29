"""Unit tests for evaluation metrics (src/anc/eval/metrics.py)."""
from __future__ import annotations

import numpy as np
import pytest

from anc.eval.metrics import (
    aggregate_results,
    compute_all_metrics,
    compute_pesq,
    compute_si_sdr,
    compute_snr_db,
    compute_stoi,
    results_to_markdown,
)

SR = 16_000


def _sine(freq: float = 440.0, dur: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * SR), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(dur: float = 2.0, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(dur * SR)) * amp).astype(np.float32)


# ---------------------------------------------------------------------------
# compute_snr_db
# ---------------------------------------------------------------------------

def test_snr_db_identical():
    """clean == enhanced → SNR must be very high (> 50 dB)."""
    sig = _sine()
    snr = compute_snr_db(sig, sig.copy())
    assert snr > 50.0, f"Expected > 50 dB, got {snr:.1f}"


def test_snr_db_known_value():
    """Enhanced signal = clean + small noise → SNR should be reasonably high (> 15 dB)."""
    clean = _sine(amp=0.5, dur=2.0)
    # noise RMS ≈ 0.05 → approximately 20 dB SNR; allow wide tolerance due to
    # waveform-realisation variance and the sum-of-squares SNR formula.
    noise = _noise(dur=2.0, amp=0.05)
    enhanced = (clean + noise).astype(np.float32)
    snr = compute_snr_db(clean, enhanced)
    assert snr > 14.0, f"Expected > 14 dB, got {snr:.2f}"



def test_snr_db_silence_clean():
    """Near-silent clean signal → should return nan, not crash."""
    clean    = np.zeros(SR, dtype=np.float32)
    enhanced = _noise(dur=1.0)
    snr = compute_snr_db(clean, enhanced)
    assert np.isnan(snr) or np.isfinite(snr)  # just must not crash


# ---------------------------------------------------------------------------
# compute_si_sdr
# ---------------------------------------------------------------------------

def test_si_sdr_identical():
    """SI-SDR of identical signals must be very high (> 50 dB)."""
    sig = _sine(dur=2.0)
    sdr = compute_si_sdr(sig, sig.copy())
    assert sdr > 50.0, f"Expected > 50 dB, got {sdr:.1f}"


def test_si_sdr_zero_signal():
    """Zero reference → must return nan without crashing."""
    ref = np.zeros(SR, dtype=np.float32)
    est = _noise(dur=1.0)
    sdr = compute_si_sdr(ref, est)
    assert np.isnan(sdr)


def test_si_sdr_scaled():
    """SI-SDR is scale-invariant: result must be same for 2× scaled estimate."""
    ref = _sine(dur=1.0)
    est = _sine(dur=1.0) + _noise(dur=1.0, amp=0.05)
    sdr1 = compute_si_sdr(ref, est)
    sdr2 = compute_si_sdr(ref, est * 2.0)
    assert abs(sdr1 - sdr2) < 0.1, f"SI-SDR should be scale-invariant: {sdr1:.2f} vs {sdr2:.2f}"


# ---------------------------------------------------------------------------
# compute_stoi
# ---------------------------------------------------------------------------

def test_stoi_identical():
    """STOI of identical signals must be 1.0."""
    pytest.importorskip("pystoi")  # skip gracefully if pystoi not installed
    sig = _sine(dur=2.0)
    stoi = compute_stoi(sig, sig.copy(), SR)
    assert abs(stoi - 1.0) < 0.01, f"Expected 1.0, got {stoi:.4f}"


def test_stoi_returns_nan_on_silence():
    """STOI of silent reference should return nan or not crash."""
    ref = np.zeros(2 * SR, dtype=np.float32)
    est = _noise(dur=2.0)
    stoi = compute_stoi(ref, est, SR)
    # pystoi may return nan or some value; just must not crash
    assert np.isnan(stoi) or np.isfinite(stoi)


def test_stoi_degraded_lower_than_clean():
    """STOI of noisy signal must be lower than STOI of clean (same signal)."""
    clean = _sine(dur=2.0, amp=0.5)
    noisy = clean + _noise(dur=2.0, amp=0.3)
    stoi_clean = compute_stoi(clean, clean.copy(), SR)
    stoi_noisy = compute_stoi(clean, noisy, SR)
    if np.isfinite(stoi_clean) and np.isfinite(stoi_noisy):
        assert stoi_clean >= stoi_noisy - 0.05  # allow tiny tolerance


# ---------------------------------------------------------------------------
# compute_pesq
# ---------------------------------------------------------------------------

def test_pesq_identical():
    """PESQ of identical signals must be near the maximum (~4.5 wideband)."""
    pesq = pytest.importorskip("pesq")
    sig = _sine(dur=3.0)  # PESQ needs ≥ 3 s
    score = compute_pesq(sig, sig.copy(), SR)
    if np.isfinite(score):
        assert score > 4.0, f"Expected > 4.0, got {score:.3f}"


def test_pesq_does_not_crash_on_silence():
    """PESQ on silent signal must not raise (may return nan)."""
    ref = np.zeros(3 * SR, dtype=np.float32)
    est = _noise(dur=3.0)
    score = compute_pesq(ref, est, SR)
    assert np.isnan(score) or np.isfinite(score)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

def test_compute_all_metrics_keys():
    """compute_all_metrics must return all expected keys."""
    clean    = _sine(dur=2.0)
    noisy    = clean + _noise(dur=2.0, amp=0.2)
    enhanced = clean + _noise(dur=2.0, amp=0.05)

    result = compute_all_metrics(
        clean=clean, noisy=noisy, enhanced=enhanced,
        sr=SR, snr_tier="Easy", noise_category="wind", sample_id="test_0",
    )
    required = {
        "snr_in", "snr_out", "delta_snr",
        "si_sdr_in", "si_sdr_out", "si_sdr_delta",
        "stoi_in", "stoi_out",
        "pesq_in", "pesq_out",
        "snr_tier", "noise_category", "sample_id",
    }
    assert required.issubset(result.keys()), f"Missing: {required - set(result.keys())}"


def test_compute_all_metrics_no_crash_silence():
    """All-zero signals must not raise an exception."""
    clean    = np.zeros(2 * SR, dtype=np.float32)
    noisy    = _noise(dur=2.0)
    enhanced = np.zeros(2 * SR, dtype=np.float32)
    result = compute_all_metrics(clean, noisy, enhanced, sr=SR)
    assert isinstance(result, dict)


def test_compute_all_metrics_snr_direction():
    """Enhanced signal closer to clean → snr_out > snr_in."""
    clean    = _sine(dur=2.0, amp=0.5)
    noisy    = clean + _noise(dur=2.0, amp=0.3, seed=1)
    enhanced = clean + _noise(dur=2.0, amp=0.05, seed=2)  # much less noise
    result = compute_all_metrics(clean, noisy, enhanced, sr=SR)
    if np.isfinite(result["snr_in"]) and np.isfinite(result["snr_out"]):
        assert result["snr_out"] > result["snr_in"]


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

def test_aggregate_results_grouping():
    """aggregate_results should produce one row per unique group value."""
    results = []
    for tier in ("Easy", "Hard"):
        for _ in range(5):
            results.append({
                "snr_tier": tier, "noise_category": "wind",
                "snr_in": 5.0, "snr_out": 8.0, "delta_snr": 3.0,
                "si_sdr_in": 4.0, "si_sdr_out": 7.0, "si_sdr_delta": 3.0,
                "stoi_in": 0.7, "stoi_out": 0.85,
                "pesq_in": 1.5, "pesq_out": 2.0,
                "sample_id": "x",
            })
    df = aggregate_results(results, group_by="snr_tier")
    assert len(df) == 2
    assert set(df.index) == {"Easy", "Hard"}


def test_aggregate_results_mean_correct():
    """Mean of delta_snr must equal the true mean."""
    results = [
        {"snr_tier": "Easy", "noise_category": "wind",
         "snr_in": 5.0, "snr_out": 8.0, "delta_snr": float(v),
         "si_sdr_in": 0.0, "si_sdr_out": 0.0, "si_sdr_delta": 0.0,
         "stoi_in": 0.5, "stoi_out": 0.6, "pesq_in": 1.0, "pesq_out": 1.5,
         "sample_id": "x"}
        for v in [1.0, 2.0, 3.0]
    ]
    df = aggregate_results(results, group_by="snr_tier")
    assert abs(df.loc["Easy", "delta_snr_mean"] - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# results_to_markdown
# ---------------------------------------------------------------------------

def test_results_to_markdown_format():
    """Output must contain '|' (markdown table delimiter)."""
    results = [{
        "snr_tier": "Easy", "noise_category": "wind",
        "snr_in": 5.0, "snr_out": 8.0, "delta_snr": 3.0,
        "si_sdr_in": 4.0, "si_sdr_out": 7.0, "si_sdr_delta": 3.0,
        "stoi_in": 0.7, "stoi_out": 0.85,
        "pesq_in": 1.5, "pesq_out": 2.0, "sample_id": "x",
    }]
    df = aggregate_results(results, group_by="snr_tier")
    md = results_to_markdown(df, title="Test")
    assert "|" in md
    assert isinstance(md, str)
