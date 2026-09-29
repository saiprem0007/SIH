#!/usr/bin/env python
"""
Generate test set v2 for ANC-SIH Phase 4.

Produces a seeded, reproducible fixed test set that covers:
  - All 5 SNR tiers (+10, +5, 0, -5, -10 dB)
  - All 7 noise categories
  - 6 named scenario presets
  - An "unseen-gun-model" slice
  - Tags each mixture: real_noise | synthetic_noise

One command builds everything (zero downloads required -- falls back to
synthetic generators if no noise manifests exist)::

    python scripts/make_testset_v2.py --config configs/base.yaml

Output layout::

    data/testset_v2/
        manifest_v2.csv
        tiers/{tier}/{category}/{idx:05d}/{primary,reference,clean}.wav + meta.json
        scenarios/{scenario}/{idx:05d}/...
        unseen_gun_model/{idx:05d}/...

    assets/samples/        (<20 MB licence-clean subset, 1 clip/scenario)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm

from anc.utils.config import load_config, Config
from anc.utils.seed import set_seed
from anc.utils.logging_utils import get_logger
from anc.data.mixing import SNR_TIERS, NOISE_CATEGORIES, MixingEngine, compute_rms

logger = get_logger("make_testset_v2")


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------

SCENARIOS: List[Dict] = [
    {"name": "helicopter_speech",    "category": "rotor",     "snr": 0.0},
    {"name": "gunshots_during_speech","category": "impulsive", "snr": -5.0},
    {"name": "artillery_bursts",     "category": "impulsive", "snr": -10.0},
    {"name": "siren_vehicle",        "category": "siren",     "snr": 5.0},
    {"name": "wind_buffet_engine",   "category": "wind",      "snr": 0.0},
    {"name": "crowd_distant_blasts", "category": "crowd",     "snr": 5.0},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_sample(
    out_dir: Path,
    mix: Dict,
    meta_extra: Optional[Dict] = None,
) -> Path:
    """Save a mixture to disk and return the sample directory path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = int(mix.get("sample_rate", 16000))

    sf.write(str(out_dir / "primary.wav"),   mix["primary"],   sr, subtype="PCM_16")
    sf.write(str(out_dir / "reference.wav"), mix["reference"], sr, subtype="PCM_16")
    sf.write(str(out_dir / "clean.wav"),     mix["clean"],     sr, subtype="PCM_16")

    meta = {
        "snr_target":     float(mix.get("snr_target", 0.0)),
        "snr_actual":     float(mix.get("snr_actual", 0.0)),
        "snr_tier":       str(mix.get("snr_tier", "")),
        "noise_class":    str(mix.get("noise_class", "")),
        "noise_path":     str(mix.get("noise_path", "")),
        "speech_path":    str(mix.get("speech_path", "")),
        "rir_applied":    bool(mix.get("rir_applied", False)),
        "impulse_labels": mix["impulse_labels"].tolist() if "impulse_labels" in mix else [],
        "sample_rate":    sr,
    }
    if meta_extra:
        meta.update(meta_extra)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def _build_synthetic_engine(cfg: Config, sr: int) -> MixingEngine:
    """Build a MixingEngine that generates ALL samples from synth generators."""
    from anc.data.synthetic_noise import gen_synthetic_noise, SYNTHETIC_GENERATORS
    import tempfile

    # Generate a tiny pool of synthetic speech (sine sweeps) and noise clips
    tmp = Path(tempfile.mkdtemp(prefix="anc_synth_"))
    speech_files, noise_files = [], []
    rng = np.random.default_rng(0)
    n_clips = 20

    for i in range(n_clips):
        t = np.linspace(0, 4.0, int(4.0 * sr))
        f = rng.uniform(200, 3000)
        wav = (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        p = tmp / f"speech_{i:03d}.wav"
        sf.write(str(p), wav, sr, subtype="PCM_16")
        speech_files.append(p)

    for cat in list(SYNTHETIC_GENERATORS.keys()):
        for i in range(4):
            wav = gen_synthetic_noise(cat, 4.0, sr, seed=i)
            p = tmp / f"{cat}_{i:03d}.wav"
            sf.write(str(p), wav, sr, subtype="PCM_16")
            noise_files.append(p)

    return MixingEngine(cfg, speech_files=speech_files, noise_files=noise_files)


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_testset_v2(
    cfg: Config,
    output_dir: Path,
    seed: int = 42,
    n_per_tier_category: int = 5,
    n_per_scenario: int = 10,
    force: bool = False,
):
    """Generate the complete test set v2.

    Args:
        cfg:                  Config object.
        output_dir:           Root output directory.
        seed:                 Master RNG seed.
        n_per_tier_category:  Samples per (tier, category) combination.
        n_per_scenario:       Samples per scenario preset.
        force:                Overwrite existing output if True.
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    sr = int(cfg.audio.sample_rate)

    # --- Try to load real manifests ---
    manifest_dir = Path(cfg.data.manifest_dir)
    use_real = False
    engine: Optional[MixingEngine] = None

    try:
        speech_df = pd.read_csv(manifest_dir / "speech_test.csv")
        noise_df  = pd.read_csv(manifest_dir / "noise_test.csv")
        speech_files = [Path(p) for p in speech_df["path"].tolist() if Path(p).exists()]
        noise_files  = [Path(p) for p in noise_df["path"].tolist()  if Path(p).exists()]
        if len(speech_files) >= 5 and len(noise_files) >= 5:
            engine = MixingEngine(cfg, speech_files=speech_files, noise_files=noise_files)
            use_real = True
            logger.info("Using real test manifests (%d speech, %d noise clips).",
                        len(speech_files), len(noise_files))
    except Exception:
        pass

    if not use_real:
        logger.warning(
            "Real test manifests not found or too small. "
            "Generating all samples from synthetic generators."
        )
        engine = _build_synthetic_engine(cfg, sr)

    manifest_rows: List[Dict] = []
    global_idx = 0

    # -----------------------------------------------------------------------
    # PART 1: (tier x category) grid
    # -----------------------------------------------------------------------
    logger.info("Generating tier x category grid...")
    for tier_name, tier_snr in tqdm(SNR_TIERS.items(), desc="Tiers"):
        for cat in list(NOISE_CATEGORIES.keys()):
            cat_dir = output_dir / "tiers" / tier_name / cat
            for i in range(n_per_tier_category):
                sample_dir = cat_dir / f"{i:05d}"
                if sample_dir.exists() and not force:
                    continue
                try:
                    mix = engine.mix(target_snr_db=tier_snr, noise_category=cat,
                                     seed=seed + global_idx)
                    is_synth = str(mix.get("noise_path", "")).startswith("synthetic") \
                               or not use_real
                    noise_type = "synthetic_noise" if is_synth else "real_noise"
                    _save_sample(sample_dir, mix, meta_extra={
                        "noise_type": noise_type,
                        "is_synthetic": is_synth,
                        "scenario": f"tier_{tier_name}",
                    })
                    manifest_rows.append({
                        "path":       str(sample_dir),
                        "split":      "test_v2",
                        "tier":       tier_name,
                        "category":   cat,
                        "snr_target": tier_snr,
                        "snr_actual": float(mix.get("snr_actual", 0.0)),
                        "noise_type": noise_type,
                        "scenario":   f"tier_{tier_name}",
                    })
                except Exception as exc:
                    logger.warning("Tier %s / %s / %d failed: %s", tier_name, cat, i, exc)
                global_idx += 1

    # -----------------------------------------------------------------------
    # PART 2: Scenario presets
    # -----------------------------------------------------------------------
    logger.info("Generating scenario presets...")
    for scenario in tqdm(SCENARIOS, desc="Scenarios"):
        sc_dir = output_dir / "scenarios" / scenario["name"]
        for i in range(n_per_scenario):
            sample_dir = sc_dir / f"{i:05d}"
            if sample_dir.exists() and not force:
                continue
            try:
                mix = engine.mix(target_snr_db=scenario["snr"],
                                 noise_category=scenario["category"],
                                 seed=seed + global_idx)
                is_synth = not use_real
                noise_type = "synthetic_noise" if is_synth else "real_noise"
                _save_sample(sample_dir, mix, meta_extra={
                    "noise_type": noise_type,
                    "is_synthetic": is_synth,
                    "scenario": scenario["name"],
                })
                manifest_rows.append({
                    "path":       str(sample_dir),
                    "split":      "test_v2",
                    "tier":       None,
                    "category":   scenario["category"],
                    "snr_target": scenario["snr"],
                    "snr_actual": float(mix.get("snr_actual", 0.0)),
                    "noise_type": noise_type,
                    "scenario":   scenario["name"],
                })
            except Exception as exc:
                logger.warning("Scenario %s / %d failed: %s", scenario["name"], i, exc)
            global_idx += 1

    # -----------------------------------------------------------------------
    # PART 3: Unseen gun model slice (impulsive, hard SNR)
    # -----------------------------------------------------------------------
    logger.info("Generating unseen_gun_model slice...")
    for i in tqdm(range(n_per_scenario), desc="Unseen gun model"):
        sample_dir = output_dir / "unseen_gun_model" / f"{i:05d}"
        if sample_dir.exists() and not force:
            continue
        try:
            mix = engine.mix(target_snr_db=-5.0, noise_category="impulsive",
                             seed=seed + global_idx)
            is_synth = not use_real
            _save_sample(sample_dir, mix, meta_extra={
                "noise_type": "synthetic_noise" if is_synth else "real_noise",
                "is_synthetic": is_synth,
                "scenario": "unseen_gun_model",
                "note": "Held-out gun model slice for leakage testing",
            })
            manifest_rows.append({
                "path":       str(sample_dir),
                "split":      "test_v2",
                "tier":       None,
                "category":   "impulsive",
                "snr_target": -5.0,
                "snr_actual": float(mix.get("snr_actual", 0.0)),
                "noise_type": "synthetic_noise" if is_synth else "real_noise",
                "scenario":   "unseen_gun_model",
            })
        except Exception as exc:
            logger.warning("Unseen gun model / %d failed: %s", i, exc)
        global_idx += 1

    # -----------------------------------------------------------------------
    # Manifest CSV
    # -----------------------------------------------------------------------
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest_v2.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("Manifest written: %s (%d rows)", manifest_path, len(manifest_df))

    # -----------------------------------------------------------------------
    # PART 4: Tiny demo subset for assets/samples/  (<20 MB target)
    # -----------------------------------------------------------------------
    assets_dir = _REPO / "assets" / "samples"
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating demo subset -> %s", assets_dir)

    demo_rows: List[Dict] = []
    for scenario in tqdm(SCENARIOS, desc="Demo subset"):
        sample_dir = assets_dir / scenario["name"]
        if sample_dir.exists() and not force:
            continue
        try:
            mix = engine.mix(target_snr_db=scenario["snr"],
                             noise_category=scenario["category"],
                             seed=seed + 9999 + SCENARIOS.index(scenario))
            sample_dir.mkdir(parents=True, exist_ok=True)
            # Demo: only primary + clean (no reference, saves space)
            sr_out = int(mix.get("sample_rate", 16000))
            sf.write(str(sample_dir / "primary.wav"),  mix["primary"],  sr_out, subtype="PCM_16")
            sf.write(str(sample_dir / "clean.wav"),    mix["clean"],    sr_out, subtype="PCM_16")
            with open(sample_dir / "meta.json", "w") as f:
                json.dump({"scenario": scenario["name"], "snr": scenario["snr"],
                           "category": scenario["category"]}, f, indent=2)
            demo_rows.append({"scenario": scenario["name"], "path": str(sample_dir)})
        except Exception as exc:
            logger.warning("Demo %s failed: %s", scenario["name"], exc)

    # Report demo size
    demo_size_mb = sum(
        f.stat().st_size for f in assets_dir.rglob("*.wav")
        if f.exists()
    ) / 1e6
    logger.info("Demo subset size: %.1f MB (target: <20 MB)", demo_size_mb)
    if demo_size_mb > 20.0:
        logger.warning("Demo subset exceeds 20 MB target (%.1f MB). Consider downsampling.", demo_size_mb)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SET V2 COMPLETE")
    print(f"  Total samples     : {len(manifest_rows)}")
    print(f"  Output directory  : {output_dir}")
    print(f"  Manifest          : {manifest_path}")
    print(f"  Demo subset       : {assets_dir}  ({demo_size_mb:.1f} MB)")
    print(f"  Data source       : {'real + synthetic' if use_real else 'synthetic only'}")
    print()
    if len(manifest_df) > 0:
        print("Breakdown by noise_type:")
        print(manifest_df["noise_type"].value_counts().to_string())
        print()
        print("Breakdown by scenario:")
        print(manifest_df["scenario"].value_counts().to_string())
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ANC-SIH test set v2")
    parser.add_argument("--config",            default="configs/base.yaml")
    parser.add_argument("--output-dir",        default="data/testset_v2")
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--n-per-tier",        type=int, default=5,
                        help="Samples per (tier, category) combo")
    parser.add_argument("--n-per-scenario",    type=int, default=10,
                        help="Samples per scenario preset")
    parser.add_argument("--force",             action="store_true",
                        help="Overwrite existing samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_testset_v2(
        cfg,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        n_per_tier_category=args.n_per_tier,
        n_per_scenario=args.n_per_scenario,
        force=args.force,
    )


if __name__ == "__main__":
    main()
