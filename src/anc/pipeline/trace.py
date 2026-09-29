"""
Trace schema for ANC-SIH Pipeline Inspector.

Every hop produces one FrameTrace; the HybridANC accumulates these into a
list that can be serialised to JSON (for the UI) or summarised into a
compact PipelineTrace (downsampled time-series + aggregate stats).

Design goals
------------
* JSON-serialisable (all fields are Python scalars or plain dicts)
* Switchable: when tracing=False the overhead is ~zero (no object creation)
* Small payload for long files via compact_trace()

Fields (per hop)
----------------
See FrameTrace dataclass.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-frame trace record
# ---------------------------------------------------------------------------

@dataclass
class FrameTrace:
    """Trace data for a single processing hop.

    All timing fields are in milliseconds. RMS fields are in dB (or 0.0 if
    the signal is silent). Probability dicts sum to 1.0.
    """
    # Identity
    frame_idx: int = 0
    time_s: float = 0.0             # centre time of this frame (seconds)

    # I/O levels
    input_rms_db: float = -100.0
    output_rms_db: float = -100.0
    gain_reduction_db: float = 0.0  # negative = attenuation

    # Impulse detector
    impulse_score: float = 0.0
    impulse_flag: int = 0

    # Noise classifier
    noise_class_probs: Dict[str, float] = field(default_factory=dict)
    predicted_class: str = "other"

    # Per-stage compute (ms)
    t_impulse_ms: float = 0.0
    t_classifier_ms: float = 0.0
    t_model_ms: float = 0.0
    t_nlms_ms: float = 0.0
    t_total_ms: float = 0.0

    # NLMS state
    nlms_state: str = "disabled"    # "adapting" | "frozen" | "disabled"

    # Reference channel
    ref_mode_active: str = "none"   # "simulated" | "provided" | "none"
    ref_quality_score: float = 1.0  # 0=broken, 1=healthy

    # Flags
    clipping_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class PipelineTrace:
    """Aggregated trace for a processed file.

    Contains per-frame records (can be large) and summary statistics.
    Use compact() to produce a small payload for API responses.
    """
    sr: int = 16000
    hop_len: int = 160
    frames: List[FrameTrace] = field(default_factory=list)

    # ------------------------------------------------------------------
    def append(self, ft: FrameTrace) -> None:
        self.frames.append(ft)

    def __len__(self) -> int:
        return len(self.frames)

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all frames."""
        if not self.frames:
            return {}

        def _mean(key: str) -> float:
            vals = [getattr(f, key) for f in self.frames
                    if not math.isnan(getattr(f, key))]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        def _pct(key: str, val: Any) -> float:
            total = len(self.frames)
            return 100.0 * sum(1 for f in self.frames if getattr(f, key) == val) / max(total, 1)

        # Noise class distribution
        class_counts: Dict[str, int] = {}
        for f in self.frames:
            cls = f.predicted_class
            class_counts[cls] = class_counts.get(cls, 0) + 1
        dominant_class = max(class_counts, key=class_counts.get) if class_counts else "unknown"

        return {
            "n_frames":          len(self.frames),
            "duration_s":        self.frames[-1].time_s if self.frames else 0.0,
            "mean_input_rms_db": _mean("input_rms_db"),
            "mean_output_rms_db":_mean("output_rms_db"),
            "mean_gain_reduction_db": _mean("gain_reduction_db"),
            "impulse_frame_pct": _pct("impulse_flag", 1),
            "dominant_noise_class": dominant_class,
            "class_distribution": class_counts,
            "mean_t_total_ms":   _mean("t_total_ms"),
            "mean_ref_quality":  _mean("ref_quality_score"),
            "clipping_frame_pct":100.0 * sum(1 for f in self.frames if f.clipping_flag) / max(len(self.frames), 1),
            "ref_mode_counts":   {m: sum(1 for f in self.frames if f.ref_mode_active == m)
                                  for m in ("simulated", "provided", "none")},
        }

    # ------------------------------------------------------------------
    def compact(self, downsample: int = 10) -> Dict[str, Any]:
        """Return a compact payload for API responses.

        Args:
            downsample: keep every Nth frame for the time-series fields

        Returns:
            dict with summary + downsampled time series
        """
        ds_frames = self.frames[::downsample]
        return {
            "summary": self.summary(),
            "time_s":          [f.time_s          for f in ds_frames],
            "input_rms_db":    [f.input_rms_db    for f in ds_frames],
            "output_rms_db":   [f.output_rms_db   for f in ds_frames],
            "impulse_score":   [f.impulse_score   for f in ds_frames],
            "impulse_flag":    [f.impulse_flag    for f in ds_frames],
            "predicted_class": [f.predicted_class for f in ds_frames],
            "t_total_ms":      [f.t_total_ms      for f in ds_frames],
            "ref_quality_score":[f.ref_quality_score for f in ds_frames],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sr":      self.sr,
            "hop_len": self.hop_len,
            "frames":  [f.to_dict() for f in self.frames],
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Convenience timer context manager
# ---------------------------------------------------------------------------

class _Timer:
    """Tiny context manager that measures wall-clock time in ms."""
    def __init__(self) -> None:
        self.ms: float = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0


def make_timer() -> _Timer:
    return _Timer()
