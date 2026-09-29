#!/usr/bin/env python3
"""
CLI script to download and prepare datasets for ANC-SIH.

Usage:
    python scripts/download_data.py --config configs/base.yaml --demo
    python scripts/download_data.py --config configs/base.yaml --librispeech
    python scripts/download_data.py --config configs/base.yaml --build-manifest
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo src/ is on the path when called as a script
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from anc.utils.config import load_config, Config
from anc.utils.seed import set_seed
from anc.utils.logging_utils import get_logger

logger = get_logger("download_data")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _download_librispeech(cfg: Config) -> None:
    from anc.data.download import download_librispeech
    raw = Path(cfg.data.raw_dir)
    download_librispeech(raw, subsets=["train-clean-100", "dev-clean", "test-clean"])


def _download_esc50(cfg: Config) -> None:
    from anc.data.download import download_esc50
    download_esc50(Path(cfg.data.raw_dir))


def _download_openslr_rir(cfg: Config) -> None:
    from anc.data.download import download_openslr_rir
    download_openslr_rir(Path(cfg.data.raw_dir))


def _show_demand_instructions(cfg: Config) -> None:
    from anc.data.download import download_demand
    download_demand(Path(cfg.data.raw_dir))


def _index_manual(folder: str, cfg: Config) -> None:
    from anc.data.download import index_manual_folder
    manifest_dir = Path(cfg.data.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_csv = manifest_dir / f"manual_{Path(folder).stem}.csv"
    index_manual_folder(folder, out_csv, source_name="manual", license="CC0")
    logger.info("Manual index saved: %s", out_csv)


def _build_manifests(cfg: Config) -> None:
    """Build speech and noise manifests from all configured dataset paths."""
    import pandas as pd
    from anc.data.manifest import (
        build_noise_manifest,
        build_speech_manifest,
        build_synthetic_manifest,
        merge_manifests,
        save_manifest,
    )

    split_ratios = {
        "train": float(cfg.data.splits.train),
        "val":   float(cfg.data.splits.val),
        "test":  float(cfg.data.splits.test),
    }
    manifest_dir = Path(cfg.data.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # --- Speech ---
    speech_dfs: list = []
    for ds in cfg.data.speech_datasets:
        root = Path(ds.path)
        if root.exists():
            df = build_speech_manifest(root, ds.source, split_ratios)
            if not df.empty:
                speech_dfs.append(df)
                logger.info("Speech %s: %d files", ds.name, len(df))
        else:
            logger.warning("Speech dataset not found: %s (skip)", root)

    if not speech_dfs:
        logger.warning("No speech data found! Run with --demo to generate a tiny dataset.")

    speech_all = merge_manifests(*speech_dfs) if speech_dfs else pd.DataFrame()
    for split in ("train", "val", "test"):
        sub = speech_all[speech_all["split"] == split] if not speech_all.empty else pd.DataFrame()
        save_manifest(sub, manifest_dir / f"speech_{split}.csv")
        logger.info("speech_%s.csv: %d files", split, len(sub))

    # --- Noise ---
    noise_dfs: list = []
    for ds in cfg.data.noise_datasets:
        root = Path(ds.path)
        if root.exists():
            df = build_noise_manifest(root, ds.source, split_ratios)
            if not df.empty:
                noise_dfs.append(df)
                logger.info("Noise %s: %d files", ds.name, len(df))
        else:
            logger.info("Noise dataset not found: %s — adding synthetic substitutes.", root)

    # Always add synthetic noise to fill gaps
    for split in ("train", "val", "test"):
        synth = build_synthetic_manifest(split=split, n_per_category=100)
        noise_dfs.append(synth)

    noise_all = merge_manifests(*noise_dfs) if noise_dfs else pd.DataFrame()
    for split in ("train", "val", "test"):
        sub = noise_all[noise_all["split"] == split] if not noise_all.empty else pd.DataFrame()
        save_manifest(sub, manifest_dir / f"noise_{split}.csv")
        cat_dist = sub["category"].value_counts().to_dict() if not sub.empty else {}
        logger.info("noise_%s.csv: %d files — %s", split, len(sub), cat_dist)

    # Print summary
    print("\n===== Manifest Summary =====")
    for split in ("train", "val", "test"):
        s_path = manifest_dir / f"speech_{split}.csv"
        n_path = manifest_dir / f"noise_{split}.csv"
        s_count = len(pd.read_csv(s_path)) if s_path.exists() else 0
        n_count = len(pd.read_csv(n_path)) if n_path.exists() else 0
        print(f"  {split:8s}: {s_count:5d} speech, {n_count:5d} noise")
    print()


def _demo(cfg: Config) -> None:
    """Create a tiny demo dataset for quick testing."""
    import pandas as pd
    from anc.data.download import generate_demo_dataset
    from anc.data.manifest import (
        build_noise_manifest,
        build_speech_manifest,
        merge_manifests,
        save_manifest,
        build_synthetic_manifest,
    )

    demo_cfg = cfg.demo
    raw_dir = Path(cfg.data.raw_dir)
    manifest_dir = Path(cfg.data.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    speech_dir = raw_dir / "demo_speech"
    noise_dir  = raw_dir / "demo_noise"

    speech_files, noise_files = generate_demo_dataset(
        speech_dir,
        noise_dir,
        n_speech=int(demo_cfg.n_speech),
        n_noise=int(demo_cfg.n_noise),
        sr=int(cfg.audio.sample_rate),
        seed=int(cfg.project.seed),
    )

    split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    s_df = build_speech_manifest(speech_dir, "demo", split_ratios)
    n_df = build_noise_manifest(noise_dir, "demo", split_ratios)

    for split in ("train", "val", "test"):
        s_sub = s_df[s_df["split"] == split] if not s_df.empty else pd.DataFrame()
        n_sub = n_df[n_df["split"] == split] if not n_df.empty else pd.DataFrame()
        # Add synthetic noise to fill categories
        n_synth = build_synthetic_manifest(split=split, n_per_category=20)
        n_combined = merge_manifests(n_sub, n_synth)
        save_manifest(s_sub, manifest_dir / f"speech_{split}.csv")
        save_manifest(n_combined, manifest_dir / f"noise_{split}.csv")

    logger.info("Demo manifests written to %s", manifest_dir)
    print(
        f"\nDemo dataset ready!\n"
        f"   Speech clips : {len(speech_files)}\n"
        f"   Noise clips  : {len(noise_files)}\n"
        f"   Manifests    : {manifest_dir}\n"
        f"\nNext: python scripts/make_testset.py --config configs/base.yaml\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for ANC-SIH.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/base.yaml", help="Path to YAML config."
    )
    parser.add_argument("--librispeech",   action="store_true", help="Download LibriSpeech.")
    parser.add_argument("--esc50",         action="store_true", help="Download ESC-50.")
    parser.add_argument("--openslr-rir",   action="store_true", help="Download OpenSLR RIR (SLR28).")
    parser.add_argument("--demand",        action="store_true", help="Print DEMAND download instructions.")
    parser.add_argument("--index-manual",  metavar="PATH",      help="Index a folder of wav files as noise.")
    parser.add_argument("--build-manifest",action="store_true", help="Build all manifests from downloaded data.")
    parser.add_argument("--demo",          action="store_true", help="Create tiny demo dataset for quick testing.")
    parser.add_argument("--all",           action="store_true", help="Download everything auto-downloadable.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.project.seed))

    if args.all:
        args.librispeech = True
        args.esc50 = True
        args.openslr_rir = True
        args.build_manifest = True

    any_action = False

    if args.demo:
        any_action = True
        _demo(cfg)

    if args.librispeech:
        any_action = True
        try:
            _download_librispeech(cfg)
        except Exception as e:
            logger.error("LibriSpeech download failed: %s", e)

    if args.esc50:
        any_action = True
        try:
            _download_esc50(cfg)
        except Exception as e:
            logger.error("ESC-50 download failed: %s", e)

    if args.openslr_rir:
        any_action = True
        try:
            _download_openslr_rir(cfg)
        except Exception as e:
            logger.error("OpenSLR RIR download failed: %s", e)

    if args.demand:
        any_action = True
        _show_demand_instructions(cfg)

    if args.index_manual:
        any_action = True
        _index_manual(args.index_manual, cfg)

    if args.build_manifest:
        any_action = True
        _build_manifests(cfg)

    if not any_action:
        parser.print_help()


if __name__ == "__main__":
    main()
