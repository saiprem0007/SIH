"""
HybridANC — the complete streaming noise-cancellation engine.

Block diagram (left to right, per hop):
   primary ──► [DC/clip guard] ──► [Impulse Handler] ──► [AI Model] ──► [NLMS post-filter] ──► output
   reference ──────────────────────────────────────────────────────────► [NLMS post-filter]
                                  [Noise Classifier] (on reference)

Reference-channel modes
-----------------------
"provided"  : caller supplies a real second-mic channel (normal wearable use)
"simulated" : reference is regenerated from noise/speech components
              (only for synthetic test scenarios)
"none"      : primary-only; NLMS post-filter is disabled

Reference quality monitor
-------------------------
Short-term coherence between primary and reference is computed every hop.
If coherence drops below a threshold (broken/wind-only mic) the engine
automatically falls back to "none" mode (with hysteresis).  Every switch
is logged in the frame trace.

Enable flags (for ablation and UI toggles)
------------------------------------------
enable_impulse_handler  : bool
enable_classifier       : bool
enable_ai_model         : bool
enable_nlms_postfilter  : bool

All four can be toggled at runtime without restarting the engine.

Streaming API
-------------
    engine = HybridANC.from_config(cfg)
    engine.reset()
    for chunk_p, chunk_r in frame_iter:
        out_chunk, trace_frame = engine.process_frame(chunk_p, chunk_r)

File API (same code path)
-------------------------
    enhanced, trace = engine.process_file("noisy.wav", ref_path="ref.wav")
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from anc.pipeline.impulse_handler import ImpulseHandler, ImpulseHandlerConfig
from anc.pipeline.nlms_postfilter import NLMSPostFilter, NLMSPostFilterConfig
from anc.pipeline.trace import FrameTrace, PipelineTrace, make_timer
from anc.models.impulse_detector import RuleImpulseDetector, ImpulseDetectorConfig
from anc.models.noise_classifier import NoiseClassifier, NOISE_CLASSES

logger = logging.getLogger(__name__)

EPS = 1e-10


# ---------------------------------------------------------------------------
# Reference quality monitor
# ---------------------------------------------------------------------------

class _RefQualityMonitor:
    """Short-term energy-coherence monitor for the reference channel.

    Uses the normalised cross-correlation coefficient between primary and
    reference over a short window.  A healthy reference is partially
    correlated with the noise in primary.  If the reference is wind-only
    or dead, coherence drops toward 0.

    Hysteresis: switch to "none" below low_threshold; switch back to
    "provided" above high_threshold.
    """

    def __init__(
        self,
        window_frames: int = 20,
        low_threshold: float = 0.05,
        high_threshold: float = 0.15,
        hysteresis_frames: int = 10,
    ) -> None:
        self.window = window_frames
        self.low_thr  = low_threshold
        self.high_thr = high_threshold
        self.hyst_frames = hysteresis_frames
        self._p_buf: List[float] = []
        self._r_buf: List[float] = []
        self._active: bool = True          # True = reference is usable
        self._switch_counter: int = 0

    def reset(self) -> None:
        self._p_buf = []
        self._r_buf = []
        self._active = True
        self._switch_counter = 0

    def update(
        self, primary_frame: np.ndarray, ref_frame: np.ndarray
    ) -> Tuple[float, bool]:
        """Update monitor with one frame. Returns (quality_score, is_active)."""
        p_rms = float(np.sqrt(np.mean(primary_frame ** 2) + EPS))
        r_rms = float(np.sqrt(np.mean(ref_frame    ** 2) + EPS))
        # Use RMS as a proxy for coherence (fast, causal)
        score = float(np.clip(
            np.corrcoef(primary_frame.astype(np.float64),
                        ref_frame.astype(np.float64))[0, 1],
            0.0, 1.0,
        ))
        if not math.isfinite(score):
            score = 0.0

        self._p_buf.append(p_rms)
        self._r_buf.append(r_rms)
        if len(self._p_buf) > self.window:
            self._p_buf.pop(0)
            self._r_buf.pop(0)

        # Hysteresis logic
        if self._active and score < self.low_thr:
            self._switch_counter += 1
            if self._switch_counter >= self.hyst_frames:
                self._active = False
                self._switch_counter = 0
                logger.info("RefQuality: switching to mode=none (score=%.3f)", score)
        elif not self._active and score > self.high_thr:
            self._switch_counter += 1
            if self._switch_counter >= self.hyst_frames:
                self._active = True
                self._switch_counter = 0
                logger.info("RefQuality: switching back to ref mode (score=%.3f)", score)
        else:
            self._switch_counter = 0

        return score, self._active


# ---------------------------------------------------------------------------
# HybridANC
# ---------------------------------------------------------------------------

class HybridANC:
    """Complete streaming hybrid ANC engine.

    Args:
        model           : AI enhancement model (nn.Module or None)
        impulse_detector: RuleImpulseDetector (always used)
        impulse_handler : ImpulseHandler
        noise_classifier: NoiseClassifier (or None)
        nlms_postfilter : NLMSPostFilter
        hop_len         : hop size in samples (default 160 = 10 ms at 16 kHz)
        sr              : sample rate
        ref_mode        : "provided" | "simulated" | "none"
        enable_impulse_handler  : toggle impulse suppression
        enable_classifier       : toggle noise classifier
        enable_ai_model         : toggle AI model
        enable_nlms_postfilter  : toggle NLMS post-filter
        tracing         : emit FrameTrace records (adds ~0.1 ms per frame)
    """

    def __init__(
        self,
        model=None,
        impulse_detector: Optional[RuleImpulseDetector] = None,
        impulse_handler: Optional[ImpulseHandler] = None,
        noise_classifier: Optional[NoiseClassifier] = None,
        nlms_postfilter: Optional[NLMSPostFilter] = None,
        hop_len: int = 160,
        sr: int = 16000,
        ref_mode: str = "provided",
        enable_impulse_handler: bool = True,
        enable_classifier: bool = True,
        enable_ai_model: bool = True,
        enable_nlms_postfilter: bool = True,
        tracing: bool = True,
    ) -> None:
        # Components
        self.model             = model
        self.detector          = impulse_detector or RuleImpulseDetector()
        self.handler           = impulse_handler  or ImpulseHandler()
        self.classifier        = noise_classifier
        self.nlms              = nlms_postfilter  or NLMSPostFilter()
        # Config
        self.hop_len           = hop_len
        self.sr                = sr
        self.ref_mode          = ref_mode
        self.tracing           = tracing
        # Enable flags
        self.enable_impulse_handler  = enable_impulse_handler
        self.enable_classifier       = enable_classifier
        self.enable_ai_model         = enable_ai_model
        self.enable_nlms_postfilter  = enable_nlms_postfilter
        # Reference quality monitor
        self._ref_monitor = _RefQualityMonitor()
        self._effective_ref_mode = ref_mode
        # Frame counter
        self._frame_idx = 0
        # AI model streaming state
        self._model_state: Any = None
        # Overlap-add buffer for AI output (covers encoder look-back)
        self._ola_buf = np.zeros(hop_len * 4, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all streaming state."""
        self.detector.reset()
        self.handler.reset()
        if self.classifier is not None:
            self.classifier.reset()
        self.nlms.reset()
        self._ref_monitor.reset()
        self._effective_ref_mode = self.ref_mode
        self._frame_idx = 0
        self._model_state = None
        self._ola_buf[:] = 0.0

    # ------------------------------------------------------------------
    # Input guards
    # ------------------------------------------------------------------

    @staticmethod
    def _guard(frame: np.ndarray, clip_level: float = 0.999) -> Tuple[np.ndarray, bool]:
        """Remove DC offset, clip, NaN/Inf guard. Returns (frame, clipping_flag)."""
        f = np.array(frame, dtype=np.float32)
        # NaN / Inf
        bad = ~np.isfinite(f)
        if bad.any():
            f[bad] = 0.0
        # DC offset
        f -= float(f.mean())
        # Clipping detection + hard clip
        clipped = bool(np.any(np.abs(f) > clip_level))
        f = np.clip(f, -clip_level, clip_level)
        return f, clipped

    @staticmethod
    def _rms_db(frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + EPS))
        return float(20.0 * np.log10(rms))

    # ------------------------------------------------------------------
    # AI model forward (handles different model APIs)
    # ------------------------------------------------------------------

    def _ai_forward(
        self, primary: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Run AI model on one hop. Returns enhanced frame (numpy)."""
        if self.model is None:
            return primary.copy()
        try:
            import torch
            p = torch.from_numpy(primary).unsqueeze(0)    # (1, hop)
            r = torch.from_numpy(reference).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                # Streaming mode if available
                if hasattr(self.model, "forward_stream"):
                    if self._model_state is None:
                        self._model_state = self.model.init_stream_state(batch_size=1)
                    out, self._model_state = self.model.forward_stream(p, r, self._model_state)
                else:
                    out = self.model(p, r)
                    if isinstance(out, tuple):
                        out = out[0]
            return out.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            logger.warning("AI model forward failed: %s — passing through", exc)
            return primary.copy()

    # ------------------------------------------------------------------
    # Core per-frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        primary_frame: np.ndarray,
        reference_frame: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[FrameTrace]]:
        """Process one hop-length frame.

        Args:
            primary_frame  : (hop_len,) float32
            reference_frame: (hop_len,) float32 or None

        Returns:
            (enhanced_frame, FrameTrace or None)
        """
        t_total_start = time.perf_counter()
        hop  = self.hop_len
        fidx = self._frame_idx
        t_s  = fidx * hop / self.sr

        # ---- Input guards ----
        primary, clipped_p = self._guard(primary_frame[:hop] if len(primary_frame) >= hop
                                          else np.pad(primary_frame, (0, hop - len(primary_frame))))
        in_rms_db = self._rms_db(primary)

        # Reference handling
        if reference_frame is not None and len(reference_frame) >= hop:
            reference, _ = self._guard(reference_frame[:hop])
        else:
            reference = np.zeros(hop, dtype=np.float32)

        # Reference quality monitor
        ref_quality, ref_ok = (1.0, True)
        if self.ref_mode != "none" and reference_frame is not None:
            ref_quality, ref_ok = self._ref_monitor.update(primary, reference)
            prev_mode = self._effective_ref_mode
            self._effective_ref_mode = self.ref_mode if ref_ok else "none"
            if prev_mode != self._effective_ref_mode and self.tracing:
                logger.info("[frame %d] ref_mode switched: %s -> %s",
                            fidx, prev_mode, self._effective_ref_mode)
        elif self.ref_mode == "none":
            self._effective_ref_mode = "none"

        use_ref = (self._effective_ref_mode != "none") and (reference_frame is not None)

        # ---- Impulse detector ----
        t_imp = make_timer()
        with t_imp:
            imp_flag, imp_score = self.detector.process_frame(primary)

        # ---- Impulse handler ----
        if self.enable_impulse_handler:
            frame_after_handler, imp_flag = self.handler.process_frame(
                primary,
                reference if use_ref else None,
                imp_flag,
            )
        else:
            frame_after_handler = primary

        # ---- Noise classifier ----
        t_clf = make_timer()
        noise_probs = {c: 0.0 for c in NOISE_CLASSES}
        noise_class = "other"
        if self.enable_classifier and self.classifier is not None:
            with t_clf:
                ref_for_clf = reference if use_ref else primary
                probs_arr = self.classifier.process_frame(ref_for_clf)
                for i, c in enumerate(NOISE_CLASSES):
                    noise_probs[c] = float(probs_arr[i])
                noise_class = NOISE_CLASSES[int(np.argmax(probs_arr))]

        # ---- AI enhancement model ----
        t_model = make_timer()
        if self.enable_ai_model:
            with t_model:
                ref_for_ai = reference if use_ref else np.zeros(hop, dtype=np.float32)
                ai_out = self._ai_forward(frame_after_handler, ref_for_ai)
        else:
            ai_out = frame_after_handler

        # Ensure correct length
        if len(ai_out) > hop:
            ai_out = ai_out[:hop]
        elif len(ai_out) < hop:
            ai_out = np.pad(ai_out, (0, hop - len(ai_out)))

        # ---- NLMS post-filter ----
        t_nlms = make_timer()
        nlms_state_str = "disabled"
        if self.enable_nlms_postfilter and use_ref:
            with t_nlms:
                final_out = self.nlms.process_frame(ai_out, reference)
            nlms_state_str = "frozen" if self.nlms.is_dtd_frozen else "adapting"
        else:
            final_out = ai_out

        # ---- Output metrics ----
        out_rms_db = self._rms_db(final_out)
        gain_reduction_db = out_rms_db - in_rms_db

        t_total_ms = (time.perf_counter() - t_total_start) * 1000.0
        self._frame_idx += 1

        # ---- Trace ----
        ft: Optional[FrameTrace] = None
        if self.tracing:
            ft = FrameTrace(
                frame_idx        = fidx,
                time_s           = t_s,
                input_rms_db     = in_rms_db,
                output_rms_db    = out_rms_db,
                gain_reduction_db= gain_reduction_db,
                impulse_score    = float(imp_score),
                impulse_flag     = int(imp_flag),
                noise_class_probs= noise_probs,
                predicted_class  = noise_class,
                t_impulse_ms     = t_imp.ms,
                t_classifier_ms  = t_clf.ms,
                t_model_ms       = t_model.ms,
                t_nlms_ms        = t_nlms.ms,
                t_total_ms       = t_total_ms,
                nlms_state       = nlms_state_str,
                ref_mode_active  = self._effective_ref_mode,
                ref_quality_score= float(ref_quality),
                clipping_flag    = clipped_p,
            )

        return final_out.astype(np.float32), ft

    # ------------------------------------------------------------------
    # Full-file processing (uses same streaming path)
    # ------------------------------------------------------------------

    def process_file(
        self,
        input_path: str | Path,
        ref_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, PipelineTrace]:
        """Process an audio file end-to-end using the streaming frame loop.

        Args:
            input_path : path to the noisy primary WAV
            ref_path   : path to the reference WAV (or None for mode='none')
            output_path: if given, save the enhanced signal here
            target_sr  : if given, resample inputs to this rate first

        Returns:
            (enhanced_signal np.ndarray float32, PipelineTrace)
        """
        import soundfile as sf

        sr = target_sr or self.sr

        # Load primary
        primary, file_sr = sf.read(str(input_path), dtype="float32", always_2d=False)
        if primary.ndim > 1:
            primary = primary.mean(axis=1)
        if file_sr != sr:
            primary = _resample(primary, file_sr, sr)

        # Load reference
        reference: Optional[np.ndarray] = None
        if ref_path is not None:
            reference, ref_sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
            if reference.ndim > 1:
                reference = reference.mean(axis=1)
            if ref_sr != sr:
                reference = _resample(reference, ref_sr, sr)
            # Pad/trim to same length
            L = min(len(primary), len(reference))
            primary   = primary[:L]
            reference = reference[:L]

        self.reset()
        hop = self.hop_len
        n_frames = len(primary) // hop
        out_buf  = np.zeros(n_frames * hop, dtype=np.float32)
        trace    = PipelineTrace(sr=sr, hop_len=hop)

        for i in range(n_frames):
            p = primary[i * hop: (i + 1) * hop]
            r = reference[i * hop: (i + 1) * hop] if reference is not None else None
            out_frame, ft = self.process_frame(p, r)
            out_buf[i * hop: (i + 1) * hop] = out_frame
            if ft is not None:
                trace.append(ft)

        if output_path is not None:
            sf.write(str(output_path), out_buf, sr)
            logger.info("Enhanced audio saved to %s", output_path)

        return out_buf, trace

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, model=None, ref_mode: str = "provided") -> "HybridANC":
        """Build HybridANC from a Config object.

        Args:
            cfg      : Config with cfg.audio.*, cfg.pipeline.* (if present)
            model    : pre-built AI model (or None)
            ref_mode : "provided" | "simulated" | "none"
        """
        hop_len = int(cfg.audio.hop_len)
        sr      = int(cfg.audio.sample_rate)

        det_cfg  = ImpulseDetectorConfig(sample_rate=sr, hop_len=hop_len)
        hand_cfg = ImpulseHandlerConfig(sample_rate=sr, hop_len=hop_len)

        return cls(
            model             = model,
            impulse_detector  = RuleImpulseDetector(det_cfg),
            impulse_handler   = ImpulseHandler(hand_cfg),
            noise_classifier  = NoiseClassifier(),
            nlms_postfilter   = NLMSPostFilter(),
            hop_len           = hop_len,
            sr                = sr,
            ref_mode          = ref_mode,
            tracing           = True,
        )

    # ------------------------------------------------------------------
    def rtf(self, duration_s: float = 4.0, n_repeats: int = 5) -> float:
        """Measure real-time factor on dummy audio."""
        self.reset()
        hop = self.hop_len
        n_frames = int(duration_s * self.sr / hop)
        times = []
        for _ in range(n_repeats):
            self.reset()
            t0 = time.perf_counter()
            for _ in range(n_frames):
                p = np.random.randn(hop).astype(np.float32) * 0.1
                r = np.random.randn(hop).astype(np.float32) * 0.05
                self.process_frame(p, r)
            times.append(time.perf_counter() - t0)
        rtf = float(np.mean(times) / duration_s)
        logger.info("RTF = %.4f  (%.1f x real-time)", rtf, 1.0 / max(rtf, 1e-9))
        return rtf


# ---------------------------------------------------------------------------
# Resampling helper (scipy, no torchaudio dep at runtime)
# ---------------------------------------------------------------------------

def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal
    from math import gcd
    g = gcd(orig_sr, target_sr)
    from scipy.signal import resample_poly
    return resample_poly(signal, target_sr // g, orig_sr // g).astype(np.float32)
