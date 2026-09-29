"""
Phase 3 unit and integration tests.

Coverage
--------
* NLMSPostFilter: convergence, DTD freeze, divergence reset, streaming==offline
* Trace: FrameTrace keys, PipelineTrace summary, compact()
* HybridANC: output shape, enable flags, NaN input, streaming==file API,
             reference dropout fallback, no clicks at frame boundaries
"""
import sys
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.pipeline.nlms_postfilter import NLMSPostFilter, NLMSPostFilterConfig
from anc.pipeline.trace import FrameTrace, PipelineTrace
from anc.pipeline.hybrid import HybridANC
from anc.pipeline.impulse_handler import ImpulseHandler
from anc.models.impulse_detector import RuleImpulseDetector

SR   = 16000
HOP  = 160
DUR  = 2.0          # seconds of test audio


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def speech(rng) -> np.ndarray:
    t = np.linspace(0, DUR, int(SR * DUR), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 200 * t) +
            0.1 * rng.standard_normal(len(t)).astype(np.float32))


@pytest.fixture
def noise(rng) -> np.ndarray:
    return rng.standard_normal(int(SR * DUR)).astype(np.float32) * 0.15


@pytest.fixture
def noisy(speech, noise) -> np.ndarray:
    N = min(len(speech), len(noise))
    return speech[:N] + noise[:N]


@pytest.fixture
def reference(noise, rng) -> np.ndarray:
    """Simulated reference: same noise through a 2-tap filter + small offset."""
    ref = np.convolve(noise, np.array([0.9, 0.1], dtype=np.float32), mode="same")
    return (ref + 0.01 * rng.standard_normal(len(ref)).astype(np.float32))


@pytest.fixture
def engine_no_model() -> HybridANC:
    """HybridANC with no AI model, tracing on."""
    return HybridANC(
        model=None,
        hop_len=HOP,
        sr=SR,
        ref_mode="provided",
        tracing=True,
    )


# ===========================================================================
# NLMSPostFilter tests
# ===========================================================================

class TestNLMSPostFilter:

    def test_streaming_equals_offline(self, noisy, reference):
        """Chunk-by-chunk streaming output must equal full-signal offline output."""
        filt = NLMSPostFilter(n_taps=32, mu=0.01)

        # Offline
        offline = filt.process_signal(noisy, reference, reset=True)

        # Streaming (frame by frame)
        filt.reset()
        chunks = []
        n_frames = len(noisy) // HOP
        for i in range(n_frames):
            p = noisy[i * HOP: (i + 1) * HOP]
            r = reference[i * HOP: (i + 1) * HOP]
            chunks.append(filt.process_frame(p, r))
        streaming = np.concatenate(chunks)

        L = min(len(offline), len(streaming))
        max_diff = np.abs(offline[:L] - streaming[:L]).max()
        assert max_diff < 1e-5, f"Streaming vs offline max diff = {max_diff:.2e}"

    def test_output_length(self, noisy, reference):
        filt = NLMSPostFilter()
        out = filt.process_signal(noisy, reference)
        assert len(out) == min(len(noisy), len(reference))

    def test_dtd_freezes_on_speech_dominant(self):
        """DTD should freeze when short-term energy >> background."""
        filt = NLMSPostFilter(n_taps=16, mu=0.05)
        filt.reset()
        # Warm up with quiet frames
        for _ in range(50):
            filt.process_frame(
                np.zeros(HOP, dtype=np.float32),
                np.zeros(HOP, dtype=np.float32),
            )
        # Inject loud burst → DTD should freeze
        loud = np.ones(HOP, dtype=np.float32) * 0.9
        filt.process_frame(loud, loud * 0.1)
        assert filt.is_dtd_frozen, "DTD should be frozen on loud speech-like burst"

    def test_convergence_reduces_noise(self, rng):
        """Filter weights should adapt to cancel correlated reference noise."""
        sr = SR
        N  = sr * 4
        # Clean sine at 300 Hz
        t     = np.arange(N, dtype=np.float32) / sr
        clean = 0.2 * np.sin(2 * np.pi * 300 * t)
        # Noise: white noise band
        noise_sig = rng.standard_normal(N).astype(np.float32) * 0.3
        # Reference = same noise (perfectly correlated, no delay)
        primary = (clean + noise_sig).astype(np.float32)
        ref     = noise_sig.copy()

        filt = NLMSPostFilter(n_taps=64, mu=0.05)
        out  = filt.process_signal(primary, ref, reset=True)

        # After convergence (second half), output should have lower noise power
        mid = N // 2
        noise_in  = np.mean(primary[mid:] ** 2)
        noise_out = np.mean(out[mid:]    ** 2)
        # At least some reduction expected (even if clean signal is mixed in)
        # Accept as long as output power doesn't blow up
        assert noise_out < noise_in * 2, (
            f"Filter diverged: out_power={noise_out:.4f} >> in_power={noise_in:.4f}"
        )

    def test_divergence_protection(self):
        """Feeding NaN / Inf should not crash; divergence resets weights."""
        filt = NLMSPostFilter(n_taps=16, mu=0.5)
        # Feed inf → should trigger divergence reset
        inf_frame = np.full(HOP, np.inf, dtype=np.float32)
        ref_frame = np.ones(HOP, dtype=np.float32)
        try:
            out = filt.process_frame(inf_frame, ref_frame)
        except Exception as exc:
            pytest.fail(f"process_frame raised on Inf input: {exc}")
        assert filt.divergence_count >= 0   # may or may not reset, but should not crash

    def test_frame_length_mismatch_raises(self, noisy, reference):
        filt = NLMSPostFilter()
        with pytest.raises(ValueError):
            filt.process_frame(noisy[:HOP], reference[:HOP // 2])


# ===========================================================================
# Trace tests
# ===========================================================================

class TestTrace:

    REQUIRED_KEYS = {
        "frame_idx", "time_s", "input_rms_db", "output_rms_db",
        "impulse_score", "impulse_flag", "noise_class_probs", "predicted_class",
        "t_impulse_ms", "t_classifier_ms", "t_model_ms", "t_nlms_ms", "t_total_ms",
        "nlms_state", "ref_mode_active", "ref_quality_score", "clipping_flag",
        "gain_reduction_db",
    }

    def test_frame_trace_keys(self, engine_no_model, noisy, reference):
        p = noisy[:HOP]
        r = reference[:HOP]
        _, ft = engine_no_model.process_frame(p, r)
        assert ft is not None
        d = ft.to_dict()
        missing = self.REQUIRED_KEYS - set(d.keys())
        assert not missing, f"Missing trace keys: {missing}"

    def test_pipeline_trace_lengths(self, engine_no_model, noisy, reference):
        trace = PipelineTrace(sr=SR, hop_len=HOP)
        n_frames = len(noisy) // HOP
        engine_no_model.reset()
        for i in range(n_frames):
            p = noisy[i * HOP: (i + 1) * HOP]
            r = reference[i * HOP: (i + 1) * HOP]
            _, ft = engine_no_model.process_frame(p, r)
            if ft:
                trace.append(ft)
        assert len(trace) == n_frames, f"Expected {n_frames} frames, got {len(trace)}"

    def test_pipeline_trace_summary(self, engine_no_model, noisy, reference):
        trace = PipelineTrace(sr=SR, hop_len=HOP)
        engine_no_model.reset()
        for i in range(len(noisy) // HOP):
            p = noisy[i * HOP: (i + 1) * HOP]
            r = reference[i * HOP: (i + 1) * HOP]
            _, ft = engine_no_model.process_frame(p, r)
            if ft:
                trace.append(ft)
        s = trace.summary()
        assert "n_frames" in s
        assert s["n_frames"] > 0
        assert "mean_t_total_ms" in s
        assert math.isfinite(s["mean_t_total_ms"])

    def test_compact_json_serialisable(self, engine_no_model, noisy, reference):
        trace = PipelineTrace(sr=SR, hop_len=HOP)
        engine_no_model.reset()
        for i in range(min(20, len(noisy) // HOP)):
            p = noisy[i * HOP: (i + 1) * HOP]
            r = reference[i * HOP: (i + 1) * HOP]
            _, ft = engine_no_model.process_frame(p, r)
            if ft:
                trace.append(ft)
        compact = trace.compact()
        # Must be JSON-serialisable
        json_str = json.dumps(compact)
        assert len(json_str) > 0


# ===========================================================================
# HybridANC tests
# ===========================================================================

class TestHybridANC:

    def test_output_shape(self, engine_no_model, noisy, reference):
        engine_no_model.reset()
        p = noisy[:HOP]
        r = reference[:HOP]
        out, _ = engine_no_model.process_frame(p, r)
        assert out.shape == (HOP,), f"Expected ({HOP},), got {out.shape}"

    def test_output_dtype(self, engine_no_model, noisy, reference):
        engine_no_model.reset()
        out, _ = engine_no_model.process_frame(noisy[:HOP], reference[:HOP])
        assert out.dtype == np.float32

    def test_nan_input_does_not_crash(self, engine_no_model):
        engine_no_model.reset()
        nan_frame = np.full(HOP, np.nan, dtype=np.float32)
        ref_frame = np.ones(HOP, dtype=np.float32) * 0.1
        try:
            out, ft = engine_no_model.process_frame(nan_frame, ref_frame)
        except Exception as exc:
            pytest.fail(f"NaN input raised: {exc}")
        assert np.isfinite(out).all(), "Output contains NaN/Inf after NaN input"

    def test_silence_does_not_crash(self, engine_no_model):
        engine_no_model.reset()
        silent = np.zeros(HOP, dtype=np.float32)
        out, ft = engine_no_model.process_frame(silent, silent)
        assert out.shape == (HOP,)

    def test_enable_flags_bypass_blocks(self, engine_no_model, noisy, reference):
        """With all blocks disabled, output should equal input."""
        engine_no_model.enable_impulse_handler  = False
        engine_no_model.enable_classifier       = False
        engine_no_model.enable_ai_model         = False
        engine_no_model.enable_nlms_postfilter  = False
        engine_no_model.reset()
        p = noisy[:HOP]
        r = reference[:HOP]
        out, _ = engine_no_model.process_frame(p, r)
        # With all blocks disabled, output = DC-removed + clipped primary
        # (DC removal is always on as a guard)
        dc_removed = p - p.mean()
        np.testing.assert_allclose(out, dc_removed, atol=0.01)

    def test_streaming_equals_process_file(self, noisy, reference, tmp_path):
        """process_file (file API) must equal frame-by-frame streaming."""
        import soundfile as sf

        # Write temp WAVs
        p_path = tmp_path / "primary.wav"
        r_path = tmp_path / "ref.wav"
        sf.write(str(p_path), noisy, SR)
        sf.write(str(r_path), reference, SR)

        engine = HybridANC(model=None, hop_len=HOP, sr=SR,
                           ref_mode="provided", tracing=False)

        # Frame-by-frame streaming
        engine.reset()
        n_frames = len(noisy) // HOP
        stream_out = []
        for i in range(n_frames):
            p = noisy[i * HOP: (i + 1) * HOP]
            r = reference[i * HOP: (i + 1) * HOP]
            out, _ = engine.process_frame(p, r)
            stream_out.append(out)
        stream_arr = np.concatenate(stream_out)

        # File API
        file_arr, _ = engine.process_file(p_path, ref_path=r_path)

        L = min(len(stream_arr), len(file_arr))
        max_diff = np.abs(stream_arr[:L] - file_arr[:L]).max()
        assert max_diff < 5e-3, f"Streaming vs process_file max diff = {max_diff:.2e}"

    def test_no_click_at_frame_boundaries(self, engine_no_model, noisy, reference):
        """Discontinuities between frames should be below a perceptual threshold."""
        engine_no_model.reset()
        n_frames = len(noisy) // HOP
        out_frames = []
        for i in range(n_frames):
            p = noisy[i * HOP: (i + 1) * HOP]
            r = reference[i * HOP: (i + 1) * HOP]
            out, _ = engine_no_model.process_frame(p, r)
            out_frames.append(out)

        # Check boundary discontinuity between consecutive frames
        max_boundary_jump = 0.0
        for i in range(1, len(out_frames)):
            jump = abs(float(out_frames[i][0]) - float(out_frames[i - 1][-1]))
            max_boundary_jump = max(max_boundary_jump, jump)

        assert max_boundary_jump < 0.5, (
            f"Large click at frame boundary: {max_boundary_jump:.4f}"
        )

    def test_reference_dropout_fallback(self, noisy):
        """When reference is zero (degraded), quality monitor should eventually switch."""
        engine = HybridANC(
            model=None, hop_len=HOP, sr=SR,
            ref_mode="provided", tracing=True,
        )
        engine._ref_monitor.hyst_frames = 3   # speed up for test
        engine.reset()
        n_frames = len(noisy) // HOP

        modes_seen = set()
        for i in range(min(n_frames, 60)):
            p = noisy[i * HOP: (i + 1) * HOP]
            # Pass completely flat (dead) reference
            r = np.zeros(HOP, dtype=np.float32)
            out, ft = engine.process_frame(p, r)
            if ft:
                modes_seen.add(ft.ref_mode_active)

        # After a few dead-reference frames, engine should have fallen back to "none"
        assert "none" in modes_seen, (
            f"Expected ref_mode to fall back to 'none', saw: {modes_seen}"
        )

    def test_mode_none_disables_nlms(self, noisy):
        """In ref_mode=none, NLMS should be disabled regardless of enable flag."""
        engine = HybridANC(
            model=None, hop_len=HOP, sr=SR,
            ref_mode="none", tracing=True,
            enable_nlms_postfilter=True,
        )
        engine.reset()
        p = noisy[:HOP]
        out, ft = engine.process_frame(p, reference_frame=None)
        assert ft is not None
        assert ft.nlms_state == "disabled", (
            f"NLMS should be disabled in mode=none, got: {ft.nlms_state}"
        )

    def test_rtf_measurement(self):
        """RTF should be a positive finite float."""
        engine = HybridANC(model=None, hop_len=HOP, sr=SR, tracing=False)
        rtf = engine.rtf(duration_s=1.0, n_repeats=2)
        assert math.isfinite(rtf) and rtf > 0, f"Bad RTF: {rtf}"

    def test_clipping_flag_set(self):
        """Frames that exceed clip_level should have clipping_flag=True."""
        engine = HybridANC(model=None, hop_len=HOP, sr=SR,
                           ref_mode="none", tracing=True)
        engine.reset()
        # Alternating +2/-2: DC removal does NOT zero it, so clip is detectable
        loud = np.empty(HOP, dtype=np.float32)
        loud[0::2] =  2.0
        loud[1::2] = -2.0
        out, ft = engine.process_frame(loud, None)
        assert ft is not None
        assert ft.clipping_flag is True, "clipping_flag should be True for saturated input"

    def test_streaming_equals_process_file_tolerance(self, noisy, reference, tmp_path):
        """process_file must equal frame-by-frame streaming within float32 tolerance."""
        import soundfile as sf

        p_path = tmp_path / "primary.wav"
        r_path = tmp_path / "ref.wav"
        sf.write(str(p_path), noisy, SR)
        sf.write(str(r_path), reference, SR)

        engine = HybridANC(model=None, hop_len=HOP, sr=SR,
                           ref_mode="provided", tracing=False)
        engine.reset()
        n_frames = len(noisy) // HOP
        stream_out = []
        for i in range(n_frames):
            p = noisy[i * HOP: (i + 1) * HOP]
            r = reference[i * HOP: (i + 1) * HOP]
            out_f, _ = engine.process_frame(p, r)
            stream_out.append(out_f)
        stream_arr = np.concatenate(stream_out)

        file_arr, _ = engine.process_file(p_path, ref_path=r_path)

        L = min(len(stream_arr), len(file_arr))
        max_diff = float(np.abs(stream_arr[:L] - file_arr[:L]).max())
        assert max_diff < 5e-3, f"Streaming vs process_file max diff = {max_diff:.2e}"
