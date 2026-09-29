"""Unit tests for NLMS and spectral-subtraction baselines."""
from __future__ import annotations

import numpy as np
import pytest

from anc.baselines.nlms import NLMSFilter
from anc.baselines.spectral_subtraction import SpectralSubtractor

SR = 16_000
HOP_LEN = 160


# ---------------------------------------------------------------------------
# NLMSFilter
# ---------------------------------------------------------------------------

def test_nlms_convergence():
    """NLMS must reduce noise when reference = exact noise copy.

    Setup: primary = speech_sine + noise, reference = noise (exact).
    After adaptation the output SNR should exceed 10 dB.
    """
    rng = np.random.default_rng(0)
    n = 4 * SR
    t = np.linspace(0, 4.0, n, endpoint=False, dtype=np.float64)
    speech = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise  = rng.standard_normal(n).astype(np.float32) * 0.5

    primary   = (speech + noise)
    reference = noise.copy()

    filt = NLMSFilter(n_taps=64, mu=0.1, delta=1e-6)
    enhanced = filt.process_signal(primary, reference)

    # Compute output SNR on last half (convergence period)
    half = n // 2
    sig = speech[half:]
    err = enhanced[half:] - speech[half:]  # residual noise
    snr_out = 10 * np.log10(np.mean(sig**2) / (np.mean(err**2) + 1e-12))
    assert snr_out > 10.0, f"Expected output SNR > 10 dB, got {snr_out:.1f} dB"


def test_nlms_zero_reference():
    """Zero reference must not crash; weights stay at zero, output = input."""
    rng = np.random.default_rng(1)
    primary   = rng.standard_normal(1000).astype(np.float32)
    reference = np.zeros(1000, dtype=np.float32)

    filt = NLMSFilter(n_taps=32, mu=0.01, delta=1e-6)
    enhanced = filt.process_signal(primary, reference)

    assert np.isfinite(enhanced).all()
    np.testing.assert_allclose(enhanced, primary, atol=1e-5)


def test_nlms_state_save_restore():
    """Saved and restored state must give identical continuation."""
    rng = np.random.default_rng(2)
    primary   = rng.standard_normal(2000).astype(np.float32)
    reference = rng.standard_normal(2000).astype(np.float32)

    filt = NLMSFilter(n_taps=32, mu=0.01, delta=1e-6)
    _ = filt.process_signal(primary[:1000], reference[:1000], reset=True)
    state = filt.get_state()

    # Process the second half
    out_original = filt.process_signal(primary[1000:], reference[1000:], reset=False)

    # Restore and process again
    filt2 = NLMSFilter(n_taps=32, mu=0.01, delta=1e-6)
    filt2.set_state(state)
    out_restored = filt2.process_signal(primary[1000:], reference[1000:], reset=False)

    np.testing.assert_allclose(out_original, out_restored, atol=1e-5)


def test_nlms_frame_matches_sample():
    """Frame-by-frame and sample-by-sample must give identical output."""
    rng = np.random.default_rng(3)
    primary   = rng.standard_normal(500).astype(np.float32)
    reference = rng.standard_normal(500).astype(np.float32)

    filt_sample = NLMSFilter(n_taps=32, mu=0.01, delta=1e-6)
    out_sample  = filt_sample.process_signal(primary, reference, reset=True)

    filt_frame  = NLMSFilter(n_taps=32, mu=0.01, delta=1e-6)
    filt_frame.reset()
    out_frame_parts = []
    frame_size = 100
    for i in range(0, 500, frame_size):
        chunk = filt_frame.process_frame(
            primary[i:i+frame_size], reference[i:i+frame_size]
        )
        out_frame_parts.append(chunk)
    out_frame = np.concatenate(out_frame_parts)

    np.testing.assert_allclose(out_sample.astype(np.float64),
                                out_frame.astype(np.float64), atol=1e-5)


def test_nlms_streaming_continuity():
    """Processing in small chunks must equal processing all at once."""
    rng = np.random.default_rng(4)
    primary   = rng.standard_normal(4000).astype(np.float32)
    reference = rng.standard_normal(4000).astype(np.float32)

    filt_full = NLMSFilter(n_taps=32, mu=0.01, delta=1e-6)
    out_full  = filt_full.process_signal(primary, reference, reset=True)

    filt_stream = NLMSFilter(n_taps=32, mu=0.01, delta=1e-6)
    filt_stream.reset()
    chunks = []
    for i in range(0, 4000, 10):
        chunk = filt_stream.process_frame(primary[i:i+10], reference[i:i+10])
        chunks.append(chunk)
    out_stream = np.concatenate(chunks)

    np.testing.assert_allclose(out_full.astype(np.float64),
                                out_stream.astype(np.float64), atol=1e-5)


def test_nlms_init_validates():
    """Invalid parameters must raise ValueError."""
    with pytest.raises(ValueError):
        NLMSFilter(n_taps=0)
    with pytest.raises(ValueError):
        NLMSFilter(mu=0.0)
    with pytest.raises(ValueError):
        NLMSFilter(delta=-1.0)


# ---------------------------------------------------------------------------
# SpectralSubtractor
# ---------------------------------------------------------------------------

def _make_sub() -> SpectralSubtractor:
    return SpectralSubtractor(
        sr=SR, n_fft=512, hop_len=HOP_LEN, win_len=320,
        alpha=2.0, beta=0.01, noise_est_frames=6, smoothing=0.98,
    )


def test_spectral_sub_basic():
    """SpectralSubtractor must reduce noise power without NaN/Inf."""
    rng = np.random.default_rng(5)
    n = 4 * SR
    speech = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 4, n)).astype(np.float32)
    noise  = rng.standard_normal(n).astype(np.float32) * 0.5
    noisy  = speech + noise

    sub      = _make_sub()
    enhanced = sub.process_signal(noisy, reset=True)

    assert np.isfinite(enhanced).all()
    assert not np.all(enhanced == 0)

    noise_in  = np.mean((noisy - speech) ** 2)
    noise_out = np.mean((enhanced[:len(speech)] - speech) ** 2)
    assert noise_out < noise_in, "SpectralSub should reduce noise power"


def test_spectral_sub_same_length():
    """Output must be the same length as the input."""
    rng = np.random.default_rng(6)
    n = 4 * SR
    noisy = rng.standard_normal(n).astype(np.float32)

    sub = _make_sub()
    enhanced = sub.process_signal(noisy)
    assert len(enhanced) == n


def test_spectral_sub_silence():
    """All-zero input must not crash and output must be near-zero."""
    silence = np.zeros(4 * SR, dtype=np.float32)
    sub = _make_sub()
    out = sub.process_signal(silence)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) < 0.1


def test_spectral_sub_from_config(base_cfg):
    """from_config class method must produce a working SpectralSubtractor."""
    sub = SpectralSubtractor.from_config(base_cfg, sr=SR)
    rng = np.random.default_rng(7)
    noisy = rng.standard_normal(SR).astype(np.float32)
    out = sub.process_signal(noisy)
    assert len(out) == SR
    assert np.isfinite(out).all()


def test_nlms_from_config(base_cfg):
    """from_config class method must produce a working NLMSFilter."""
    filt = NLMSFilter.from_config(base_cfg)
    assert filt.n_taps == int(base_cfg.baselines.nlms.n_taps)
    assert filt.mu == float(base_cfg.baselines.nlms.mu)
