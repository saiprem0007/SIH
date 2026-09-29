#!/usr/bin/env python
"""
ANC-SIH Phase 3 — Pipeline CLI

Usage examples
--------------
# No reference (primary-only):
python scripts/run_pipeline.py --input noisy.wav --mode none --out clean.wav --trace trace.json

# With provided reference:
python scripts/run_pipeline.py --input noisy.wav --ref ref.wav --mode provided --out clean.wav

# Measure RTF only:
python scripts/run_pipeline.py --input noisy.wav --mode none --rtf-only

# List available models:
python scripts/run_pipeline.py --list-models

Flags
-----
--input PATH        Input noisy WAV file  (required unless --rtf-only / --list-models)
--ref PATH          Reference mic WAV (required if --mode provided)
--mode MODE         "provided" | "simulated" | "none"  (default: none)
--model ID          Model id from configs/models.yaml   (default: convtasnet_base)
--out PATH          Output enhanced WAV  (optional)
--trace PATH        Output trace JSON file  (optional)
--config PATH       YAML config  (default: configs/base.yaml)
--no-trace          Disable trace collection
--rtf-only          Just measure RTF and exit
--list-models       Print available models and exit
--ablation          Run all 5 ablation variants and print a comparison table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np

from anc.utils.config import load_config
from anc.utils.seed import set_seed
from anc.utils.logging_utils import get_logger
from anc.models.registry import list_models, load_hybrid, get_model_info
from anc.pipeline.hybrid import HybridANC

logger = get_logger("run_pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_models() -> None:
    print("\nAvailable models:")
    print(f"{'ID':<30} {'Arch':<14} {'Params':>8}  {'Needs ref':>9}  Description")
    print("-" * 90)
    for m in list_models():
        p = f"{m.get('params_M','?')}M" if m.get('params_M') else "?"
        print(f"{m['id']:<30} {m['architecture']:<14} {p:>8}  "
              f"{'yes' if m.get('requires_reference') else 'no':>9}  "
              f"{str(m.get('description',''))[:50].strip()}")
    print()


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-10))
    return 20.0 * np.log10(rms)


def _delta_snr(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """Estimate delta-SNR (requires clean reference for proper SNR; approximation here)."""
    def snr(s, n):
        return 10.0 * np.log10(
            np.sum(s.astype(np.float64) ** 2) /
            (np.sum((s - n).astype(np.float64) ** 2) + 1e-12)
        )
    # Without clean we use: output_rms / noise_estimate
    return _rms_db(enhanced) - _rms_db(noisy)


def _run_one(
    engine: HybridANC,
    input_path: Path,
    ref_path: Path | None,
    out_path: Path | None,
    trace_path: Path | None,
) -> dict:
    """Run engine on one file. Return summary dict."""
    t0 = time.perf_counter()
    enhanced, trace = engine.process_file(
        input_path,
        ref_path=ref_path,
        output_path=out_path,
    )
    wall_s = time.perf_counter() - t0

    import soundfile as sf
    original, _ = sf.read(str(input_path), dtype="float32", always_2d=False)
    if original.ndim > 1:
        original = original.mean(axis=1)
    dur_s = len(original) / engine.sr
    rtf = wall_s / max(dur_s, 1e-9)

    summary = trace.summary()
    print(f"\n{'='*60}")
    print(f"Input   : {input_path.name}  ({dur_s:.1f} s)")
    print(f"Model   : {engine.model.__class__.__name__ if engine.model else 'none'}")
    print(f"Ref mode: {engine._effective_ref_mode}")
    print(f"RTF     : {rtf:.4f}  ({1/max(rtf,1e-9):.1f}x real-time)")
    print(f"Frames  : {summary.get('n_frames',0)}")
    print(f"Input RMS  : {summary.get('mean_input_rms_db',0):.1f} dB")
    print(f"Output RMS : {summary.get('mean_output_rms_db',0):.1f} dB")
    print(f"Mean gain reduction: {summary.get('mean_gain_reduction_db',0):.1f} dB")
    print(f"Impulse frames: {summary.get('impulse_frame_pct',0):.1f}%")
    print(f"Dominant noise: {summary.get('dominant_noise_class','?')}")
    print(f"Mean frame time: {summary.get('mean_t_total_ms',0):.2f} ms/frame  "
          f"(budget: {engine.hop_len/engine.sr*1000:.1f} ms)")
    print(f"{'='*60}\n")

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as f:
            json.dump(trace.compact(downsample=5), f, indent=2)
        print(f"Trace saved to {trace_path}")

    return {"rtf": rtf, **summary}


def _run_ablation(cfg, input_path: Path, ref_path: Path | None, model_id: str) -> None:
    """Run all 5 ablation configurations and print a comparison table."""
    import soundfile as sf

    original, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    if original.ndim > 1:
        original = original.mean(axis=1)

    configs = [
        ("baseline_nlms",    False, False, False, True),
        ("ai_only",          False, False, True,  False),
        ("ai_impulse",       True,  False, True,  False),
        ("ai_nlms",          False, False, True,  True),
        ("full_hybrid",      True,  True,  True,  True),
    ]

    ref_mode = "provided" if ref_path else "none"
    rows = []

    for name, imp, clf, ai, nlms in configs:
        engine = load_hybrid(model_id, ref_mode=ref_mode, cfg=cfg)
        engine.enable_impulse_handler = imp
        engine.enable_classifier      = clf
        engine.enable_ai_model        = ai
        engine.enable_nlms_postfilter = nlms and (ref_mode != "none")
        engine.tracing = False  # faster

        engine.reset()
        enhanced, _ = engine.process_file(input_path, ref_path=ref_path)

        gain_db = _rms_db(enhanced) - _rms_db(original[:len(enhanced)])
        rows.append((name, imp, clf, ai, nlms, gain_db))

    print("\nAblation results (gain = output_rms - input_rms, negative = noise reduction):")
    print(f"{'Config':<20} {'Impulse':>8} {'Classif':>8} {'AI':>4} {'NLMS':>5} {'Gain(dB)':>10}")
    print("-" * 60)
    for name, imp, clf, ai, nlms, gain in rows:
        print(f"{name:<20} {'Y' if imp else 'N':>8} {'Y' if clf else 'N':>8} "
              f"{'Y' if ai else 'N':>4} {'Y' if nlms else 'N':>5} {gain:>10.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ANC-SIH run_pipeline.py")
    p.add_argument("--input",        default=None)
    p.add_argument("--ref",          default=None)
    p.add_argument("--mode",         default="none", choices=["provided", "simulated", "none"])
    p.add_argument("--model",        default="convtasnet_base")
    p.add_argument("--out",          default=None)
    p.add_argument("--trace",        default=None)
    p.add_argument("--config",       default="configs/base.yaml")
    p.add_argument("--no-trace",     action="store_true")
    p.add_argument("--rtf-only",     action="store_true")
    p.add_argument("--list-models",  action="store_true")
    p.add_argument("--ablation",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.project.seed))

    if args.list_models:
        _print_models()
        return

    ref_mode = args.mode
    engine = load_hybrid(args.model, ref_mode=ref_mode, cfg=cfg)
    engine.tracing = not args.no_trace

    if args.rtf_only:
        rtf = engine.rtf(duration_s=4.0, n_repeats=5)
        import platform, torch
        print(f"\nMachine : {platform.node()}")
        print(f"CPU     : {platform.processor()}")
        print(f"PyTorch : {torch.__version__}")
        print(f"CUDA    : {'yes' if torch.cuda.is_available() else 'no'}")
        print(f"Model   : {args.model}")
        print(f"RTF     : {rtf:.5f}  ({1/max(rtf,1e-9):.1f}x real-time)")
        return

    if args.input is None:
        print("ERROR: --input is required (or use --list-models / --rtf-only).")
        sys.exit(1)

    input_path  = Path(args.input)
    ref_path    = Path(args.ref)   if args.ref   else None
    out_path    = Path(args.out)   if args.out   else None
    trace_path  = Path(args.trace) if args.trace else None

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)
    if ref_path and not ref_path.exists():
        print(f"ERROR: reference file not found: {ref_path}")
        sys.exit(1)

    if args.ablation:
        _run_ablation(cfg, input_path, ref_path, args.model)
        return

    _run_one(engine, input_path, ref_path, out_path, trace_path)


if __name__ == "__main__":
    main()
