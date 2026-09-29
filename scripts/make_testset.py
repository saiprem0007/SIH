#!/usr/bin/env python3
"""
CLI script to generate the fixed, reproducible evaluation test set.

Usage:
    python scripts/make_testset.py --config configs/base.yaml
    python scripts/make_testset.py --config configs/base.yaml --n-per-tier 5 --n-per-category 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from anc.utils.config import load_config
from anc.utils.seed import set_seed
from anc.utils.logging_utils import get_logger

logger = get_logger("make_testset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed ANC-SIH test set.")
    parser.add_argument("--config",           default="configs/base.yaml")
    parser.add_argument("--output-dir",       default=None,  help="Override output dir.")
    parser.add_argument("--n-per-tier",       type=int, default=20)
    parser.add_argument("--n-per-category",   type=int, default=10)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--force",            action="store_true", help="Overwrite existing test set.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    output_dir = Path(args.output_dir or cfg.data.testset_dir)

    # Safety: don't overwrite unless forced
    manifest_csv = output_dir / "manifest.csv"
    if manifest_csv.exists() and not args.force:
        logger.info("Test set already exists at %s (use --force to regenerate).", output_dir)
    else:
        import pandas as pd
        from anc.data.manifest import load_manifest
        from anc.data.dataset import generate_fixed_test_set

        manifest_dir = Path(cfg.data.manifest_dir)
        speech_test = load_manifest(manifest_dir / "speech_test.csv")
        noise_test  = load_manifest(manifest_dir / "noise_test.csv")

        # Auto-create demo dataset if manifests are missing
        if speech_test.empty or noise_test.empty:
            logger.warning(
                "Manifests not found — auto-generating demo dataset first."
            )
            import subprocess, sys
            subprocess.run(
                [sys.executable, "scripts/download_data.py", "--config", args.config, "--demo"],
                check=True,
            )
            speech_test = load_manifest(manifest_dir / "speech_test.csv")
            noise_test  = load_manifest(manifest_dir / "noise_test.csv")

        generate_fixed_test_set(
            cfg=cfg,
            speech_manifest=speech_test,
            noise_manifest=noise_test,
            output_dir=output_dir,
            n_per_tier=args.n_per_tier,
            n_per_category=args.n_per_category,
            seed=args.seed,
        )

    # --- Verification / summary ---
    import soundfile as sf
    import random

    manifest_csv = output_dir / "manifest.csv"
    if not manifest_csv.exists():
        logger.error("manifest.csv not found — generation may have failed.")
        return

    import pandas as pd
    df = pd.read_csv(manifest_csv)
    print(f"\n{'='*50}")
    print(f"  Test set: {output_dir}")
    print(f"  Total samples : {len(df)}")
    print(f"\n  By tier:")
    print(df["tier"].value_counts().to_string())
    print(f"\n  By category:")
    print(df["category"].value_counts().to_string())

    # Sanity check: load 3 random samples
    print(f"\n  --- Sanity check (3 random samples) ---")
    sample_dirs = sorted(p.parent for p in output_dir.rglob("primary.wav"))
    picks = random.sample(sample_dirs, min(3, len(sample_dirs)))
    for sd in picks:
        import json
        with open(sd / "meta.json") as f:
            meta = json.load(f)
        p_info = sf.info(str(sd / "primary.wav"))
        print(
            f"  {sd.relative_to(output_dir)} | "
            f"tier={meta['snr_tier']} snr={meta['snr_actual']:.1f}dB "
            f"cat={meta['noise_class']} dur={p_info.duration:.1f}s"
        )
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
