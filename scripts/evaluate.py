#!/usr/bin/env python3
"""
CLI script to run baseline evaluation on the fixed ANC-SIH test set.

Usage:
    python scripts/evaluate.py --config configs/base.yaml --method nlms
    python scripts/evaluate.py --config configs/base.yaml --method all --measure-rtf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from anc.utils.config import load_config
from anc.utils.seed import set_seed
from anc.utils.logging_utils import get_logger

logger = get_logger("evaluate")

AVAILABLE_METHODS = ("nlms", "spectral", "noisy", "all")


def _get_processor(method: str, cfg, sr: int) -> Callable:
    """Return a callable (primary, reference) -> enhanced for a given method."""
    if method == "noisy":
        return lambda primary, reference: primary.copy()

    if method == "nlms":
        from anc.baselines.nlms import NLMSFilter
        filt = NLMSFilter.from_config(cfg)
        def _nlms(primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
            filt.reset()
            return filt.process_signal(primary, reference, reset=False)
        return _nlms

    if method == "spectral":
        from anc.baselines.spectral_subtraction import SpectralSubtractor
        sub = SpectralSubtractor.from_config(cfg, sr=sr)
        def _spectral(primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
            sub.reset()
            return sub.process_signal(primary, reset=False)
        return _spectral

    raise ValueError(f"Unknown method: {method}. Choose from {AVAILABLE_METHODS}.")


def _evaluate_method(
    method: str,
    cfg,
    test_dir: Path,
    output_dir: Path,
    max_samples: int,
    measure_rtf: bool,
) -> List[dict]:
    """Run evaluation for one method. Returns list of per-sample metric dicts."""
    from anc.data.dataset import FixedTestSet
    from anc.eval.metrics import (
        compute_all_metrics,
        measure_rtf as rtf_fn,
        results_to_csv,
        plot_results,
    )

    sr = int(cfg.audio.sample_rate)
    dataset = FixedTestSet(test_dir)
    processor = _get_processor(method, cfg, sr)

    results: List[dict] = []
    n = min(max_samples, len(dataset))
    logger.info("Evaluating method='%s' on %d samples …", method, n)

    from tqdm import tqdm
    for idx in tqdm(range(n), desc=method, unit="sample"):
        sample = dataset[idx]
        clean  = sample["clean"].numpy()
        noisy  = sample["primary"].numpy()
        ref    = sample["reference"].numpy()

        try:
            enhanced = processor(noisy, ref)
        except Exception as exc:
            logger.warning("Sample %d processing failed (%s): %s — skipping.", idx, method, exc)
            continue

        metrics = compute_all_metrics(
            clean=clean,
            noisy=noisy,
            enhanced=enhanced,
            sr=sr,
            snr_tier=sample["snr_tier"],
            noise_category=sample["noise_class"],
            sample_id=sample.get("sample_id", str(idx)),
        )
        metrics["method"] = method
        results.append(metrics)

    if not results:
        logger.error("No results for method '%s'.", method)
        return results

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{method}_results.csv"
    results_to_csv(results, csv_path)
    logger.info("Results saved: %s", csv_path)

    # Save plots
    plot_dir = output_dir / f"{method}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_results(results, plot_dir, method_name=method)
    except Exception as exc:
        logger.warning("Could not generate plots: %s", exc)

    # RTF measurement
    if measure_rtf:
        try:
            if method == "nlms":
                from anc.baselines.nlms import NLMSFilter
                filt = NLMSFilter.from_config(cfg)
                dummy_ref = np.zeros(4 * sr, dtype=np.float32)
                rtf = rtf_fn(
                    lambda s: filt.process_signal(s, dummy_ref, reset=True),
                    audio_dur_s=4.0, sr=sr,
                    n_repeats=int(cfg.eval.rtf_repeats),
                )
            elif method == "spectral":
                from anc.baselines.spectral_subtraction import SpectralSubtractor
                sub = SpectralSubtractor.from_config(cfg, sr=sr)
                rtf = rtf_fn(
                    lambda s: sub.process_signal(s, reset=True),
                    audio_dur_s=4.0, sr=sr,
                    n_repeats=int(cfg.eval.rtf_repeats),
                )
            else:
                rtf = rtf_fn(lambda s: s, audio_dur_s=4.0, sr=sr, n_repeats=3)
            pass_str = "PASS" if rtf < 1.0 else "FAIL"
            print(f"\n  RTF [{method}]: {rtf:.4f}  {pass_str} (<1.0 required)")
        except Exception as exc:
            logger.warning("RTF measurement failed: %s", exc)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ANC-SIH baselines.")
    parser.add_argument("--config",       default="configs/base.yaml")
    parser.add_argument(
        "--method", default="all", choices=list(AVAILABLE_METHODS),
        help="Method to evaluate.",
    )
    parser.add_argument("--test-dir",     default=None, help="Override test-set dir.")
    parser.add_argument("--output-dir",   default=None, help="Override output dir.")
    parser.add_argument("--max-samples",  type=int, default=10**9)
    parser.add_argument("--measure-rtf",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.project.seed))

    test_dir   = Path(args.test_dir   or cfg.data.testset_dir)
    output_dir = Path(args.output_dir or cfg.eval.output_dir)
    sr = int(cfg.audio.sample_rate)

    if not test_dir.exists():
        logger.error(
            "Test set not found at %s. Run: python scripts/make_testset.py", test_dir
        )
        sys.exit(1)

    methods = ("nlms", "spectral", "noisy") if args.method == "all" else (args.method,)

    all_results: dict = {}
    for method in methods:
        results = _evaluate_method(
            method=method,
            cfg=cfg,
            test_dir=test_dir,
            output_dir=output_dir,
            max_samples=args.max_samples,
            measure_rtf=args.measure_rtf,
        )
        all_results[method] = results

    # --- Print summary tables ---
    import pandas as pd
    from anc.eval.metrics import aggregate_results, results_to_markdown

    print("\n" + "="*70)
    print("  RESULTS BY SNR TIER")
    print("="*70)
    for method, results in all_results.items():
        if not results:
            continue
        df_tier = aggregate_results(results, group_by="snr_tier")
        md = results_to_markdown(df_tier, title=f"Method: {method}")
        print(f"\n{md}")

    print("\n" + "="*70)
    print("  RESULTS BY NOISE CATEGORY")
    print("="*70)
    for method, results in all_results.items():
        if not results:
            continue
        df_cat = aggregate_results(results, group_by="noise_category")
        md = results_to_markdown(df_cat, title=f"Method: {method}")
        print(f"\n{md}")

    # Comparison table across methods (if multiple)
    if len(all_results) > 1:
        rows = []
        for method, results in all_results.items():
            if not results:
                continue
            import numpy as _np
            dsnr = _np.nanmean([r.get("delta_snr", float("nan")) for r in results])
            stoi = _np.nanmean([r.get("stoi_out", float("nan")) for r in results])
            pesq = _np.nanmean([r.get("pesq_out", float("nan")) for r in results])
            rows.append({"method": method, "Δ-SNR": round(dsnr, 2),
                         "STOI_out": round(stoi, 3), "PESQ_out": round(pesq, 3)})
        cmp_df = pd.DataFrame(rows).set_index("method")
        print("\n" + "="*70)
        print("  COMPARISON SUMMARY")
        print("="*70)
        # Print comparison as a simple ASCII table (no tabulate dependency)
        print(cmp_df.to_string())
        print()

    print(f"Results written to: {output_dir}\n")


if __name__ == "__main__":
    main()
