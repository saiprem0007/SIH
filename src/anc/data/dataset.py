"""
PyTorch Dataset classes for the ANC-SIH speech enhancement pipeline.

Two modes:
  - SpeechEnhancementDataset: on-the-fly mixing (training / unlimited variety).
  - FixedTestSet: loads a pre-generated, reproducible test set from disk.

Also provides ``generate_fixed_test_set`` to pre-bake evaluation data.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from anc.data.manifest import load_manifest
from anc.data.mixing import MixingEngine, NOISE_CATEGORIES, SNR_TIERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# On-the-fly dataset
# ---------------------------------------------------------------------------

class SpeechEnhancementDataset(Dataset):
    """On-the-fly mixing dataset for training.

    Each call to ``__getitem__`` generates a fresh mixture using
    :class:`~anc.data.mixing.MixingEngine`.  The dataset is effectively
    infinite (length = *virtual_size*); use *virtual_size* to control the
    number of steps per epoch.

    Args:
        cfg:             Config object with audio/data/mixing sections.
        split:           Which manifest split to use ('train', 'val', 'test').
        speech_manifest: Pre-loaded speech DataFrame; loaded from cfg if None.
        noise_manifest:  Pre-loaded noise DataFrame; loaded from cfg if None.
        virtual_size:    Reported dataset length (steps per epoch).
    """

    def __init__(
        self,
        cfg,
        split: str = "train",
        speech_manifest: Optional[pd.DataFrame] = None,
        noise_manifest: Optional[pd.DataFrame] = None,
        virtual_size: int = 10_000,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.virtual_size = virtual_size
        self.sr: int = int(cfg.audio.sample_rate)

        # Load manifests from config paths if not supplied
        manifest_dir = Path(cfg.data.manifest_dir)
        if speech_manifest is None:
            speech_manifest = load_manifest(manifest_dir / f"speech_{split}.csv")
        if noise_manifest is None:
            noise_manifest = load_manifest(manifest_dir / f"noise_{split}.csv")

        if speech_manifest.empty:
            raise RuntimeError(
                f"Speech manifest for split='{split}' is empty. "
                "Run: python scripts/download_data.py --build-manifest"
            )
        if noise_manifest.empty:
            raise RuntimeError(
                f"Noise manifest for split='{split}' is empty. "
                "Run: python scripts/download_data.py --build-manifest"
            )

        # Filter manifests to the requested split
        speech_rows = speech_manifest[speech_manifest["split"] == split]
        noise_rows  = noise_manifest[noise_manifest["split"] == split]

        speech_files = [
            Path(p) for p in speech_rows["path"].tolist()
            if not str(p).startswith("synthetic::")
        ]
        noise_files = [
            Path(p) for p in noise_rows["path"].tolist()
            if not str(p).startswith("synthetic::")
        ]

        # Extract impulse files from noise manifest
        impulse_mask = noise_rows["category"] == "impulsive"
        impulse_files = [
            Path(p) for p in noise_rows[impulse_mask]["path"].tolist()
            if not str(p).startswith("synthetic::")
        ]

        rir_dir = Path(cfg.data.rir_dir) if hasattr(cfg.data, "rir_dir") else None
        rir_files: List[Path] = []
        if rir_dir and rir_dir.exists():
            rir_files = list(rir_dir.rglob("*.wav"))

        self.engine = MixingEngine(
            cfg=cfg,
            speech_files=speech_files,
            noise_files=noise_files,
            rir_files=rir_files or None,
            impulse_files=impulse_files or None,
        )

        logger.info(
            "SpeechEnhancementDataset [%s]: %d speech, %d noise, virtual_size=%d",
            split, len(speech_files), len(noise_files), virtual_size,
        )

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate one mixture.

        Uses *idx* as the random seed for within-epoch reproducibility.

        Args:
            idx: Dataset index (used as RNG seed).

        Returns:
            Dict with keys: primary, reference, clean, snr_target, snr_actual,
            snr_tier, noise_class, impulse_labels.
            Waveform tensors are float32, shape (N,).
        """
        result = self.engine.mix(seed=idx)

        return {
            "primary":       torch.from_numpy(result["primary"]),
            "reference":     torch.from_numpy(result["reference"]),
            "clean":         torch.from_numpy(result["clean"]),
            "snr_target":    float(result["snr_target"]),
            "snr_actual":    float(result["snr_actual"]),
            "snr_tier":      result["snr_tier"],
            "noise_class":   result["noise_class"],
            "impulse_labels":torch.from_numpy(result["impulse_labels"]),
        }


# ---------------------------------------------------------------------------
# Fixed test-set loader
# ---------------------------------------------------------------------------

class FixedTestSet(Dataset):
    """Loads a pre-generated, reproducible fixed test set from disk.

    Each sample is stored as a subdirectory containing:
        primary.wav   — noisy primary mic signal
        reference.wav — reference mic signal
        clean.wav     — clean speech target
        meta.json     — metadata (snr, noise_class, snr_tier, impulse_labels …)

    Args:
        test_dir: Root directory of the pre-generated test set.
    """

    def __init__(self, test_dir: str | Path) -> None:
        self.test_dir = Path(test_dir)
        if not self.test_dir.exists():
            raise FileNotFoundError(
                f"Test-set directory not found: {self.test_dir}. "
                "Run: python scripts/make_testset.py"
            )

        # Collect all sample directories (containing primary.wav)
        self.sample_dirs: List[Path] = sorted(
            p.parent for p in self.test_dir.rglob("primary.wav")
        )

        if not self.sample_dirs:
            raise RuntimeError(f"No samples found under {self.test_dir}.")

        logger.info("FixedTestSet: %d samples at %s", len(self.sample_dirs), self.test_dir)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load one test sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with primary, reference, clean (float32 tensors) and metadata.
        """
        sample_dir = self.sample_dirs[idx]

        primary,  sr = sf.read(str(sample_dir / "primary.wav"),  dtype="float32")
        reference, _ = sf.read(str(sample_dir / "reference.wav"), dtype="float32")
        clean,     _ = sf.read(str(sample_dir / "clean.wav"),     dtype="float32")

        with open(sample_dir / "meta.json", "r") as f:
            meta: Dict[str, Any] = json.load(f)

        impulse_labels = np.asarray(meta.get("impulse_labels", []), dtype=np.float32)

        return {
            "primary":       torch.from_numpy(primary),
            "reference":     torch.from_numpy(reference),
            "clean":         torch.from_numpy(clean),
            "snr_target":    float(meta.get("snr_target", 0.0)),
            "snr_actual":    float(meta.get("snr_actual", 0.0)),
            "snr_tier":      str(meta.get("snr_tier", "")),
            "noise_class":   str(meta.get("noise_class", "")),
            "impulse_labels":torch.from_numpy(impulse_labels),
            "sample_id":     str(meta.get("sample_id", sample_dir.name)),
        }


# ---------------------------------------------------------------------------
# Test-set generator
# ---------------------------------------------------------------------------

def generate_fixed_test_set(
    cfg,
    speech_manifest: pd.DataFrame,
    noise_manifest: pd.DataFrame,
    output_dir: str | Path,
    n_per_tier: int = 20,
    n_per_category: int = 10,
    seed: int = 42,
) -> None:
    """Pre-generate a reproducible fixed test set and write it to disk.

    Generates samples for each (SNR tier, noise category) combination
    and an additional set covering all tiers uniformly.

    Directory structure::

        output_dir/
          {tier}/{category}/{id:05d}/
            primary.wav
            reference.wav
            clean.wav
            meta.json
          manifest.csv

    Args:
        cfg:              Config object.
        speech_manifest:  Speech manifest DataFrame (test split).
        noise_manifest:   Noise manifest DataFrame (test split).
        output_dir:       Root directory for the generated test set.
        n_per_tier:       Samples per SNR tier (across all categories).
        n_per_category:   Samples per (tier × category) pair.
        seed:             Global random seed for full reproducibility.
    """
    import random as _random

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr: int = int(cfg.audio.sample_rate)

    # Extract real file paths
    speech_files = [
        Path(p) for p in speech_manifest["path"].tolist()
        if not str(p).startswith("synthetic::")
    ]
    noise_files = [
        Path(p) for p in noise_manifest["path"].tolist()
        if not str(p).startswith("synthetic::")
    ]

    rir_dir = Path(cfg.data.rir_dir) if hasattr(cfg.data, "rir_dir") else None
    rir_files: List[Path] = list(rir_dir.rglob("*.wav")) if (rir_dir and rir_dir.exists()) else []

    impulse_files: List[Path] = [
        Path(p) for p in noise_manifest[noise_manifest["category"] == "impulsive"]["path"].tolist()
        if not str(p).startswith("synthetic::")
    ]

    if not speech_files:
        logger.warning("No speech files in test manifest — using demo generation.")
        from anc.data.download import generate_demo_dataset
        sp_dir = output_dir / "_demo_speech"
        ns_dir = output_dir / "_demo_noise"
        speech_files, noise_files = generate_demo_dataset(
            sp_dir, ns_dir, n_speech=50, n_noise=50, sr=sr, seed=seed
        )

    engine = MixingEngine(
        cfg=cfg,
        speech_files=speech_files,
        noise_files=noise_files,
        rir_files=rir_files or None,
        impulse_files=impulse_files or None,
    )

    manifest_rows: List[Dict[str, Any]] = []
    global_seed = seed
    sample_idx = 0

    # Generate tier × category grid
    tiers = list(SNR_TIERS.keys())
    categories = list(NOISE_CATEGORIES.keys())

    total = len(tiers) * len(categories) * n_per_category + len(tiers) * n_per_tier
    pbar = tqdm(total=total, desc="Generating test set")

    for tier_name, tier_snr in SNR_TIERS.items():
        for cat in categories:
            for i in range(n_per_category):
                mix_seed = global_seed + sample_idx
                result = engine.mix(
                    target_snr_db=tier_snr,
                    noise_category=cat,
                    seed=mix_seed,
                )
                _save_sample(output_dir, tier_name, cat, sample_idx, result, sr)
                manifest_rows.append({
                    "sample_id":   f"{sample_idx:06d}",
                    "tier":        tier_name,
                    "category":    cat,
                    "snr_target":  result["snr_target"],
                    "snr_actual":  result["snr_actual"],
                    "speech_path": result["speech_path"],
                    "noise_path":  result["noise_path"],
                })
                sample_idx += 1
                pbar.update(1)

        # Additional per-tier samples (any category)
        for i in range(n_per_tier):
            mix_seed = global_seed + sample_idx
            result = engine.mix(target_snr_db=tier_snr, seed=mix_seed)
            _save_sample(output_dir, tier_name, "any", sample_idx, result, sr)
            manifest_rows.append({
                "sample_id":   f"{sample_idx:06d}",
                "tier":        tier_name,
                "category":    result["noise_class"],
                "snr_target":  result["snr_target"],
                "snr_actual":  result["snr_actual"],
                "speech_path": result["speech_path"],
                "noise_path":  result["noise_path"],
            })
            sample_idx += 1
            pbar.update(1)

    pbar.close()

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    logger.info(
        "Test set generated: %d samples at %s", sample_idx, output_dir
    )
    logger.info("Category distribution:\n%s", manifest_df["category"].value_counts())
    logger.info("Tier distribution:\n%s", manifest_df["tier"].value_counts())


def _save_sample(
    output_dir: Path,
    tier: str,
    category: str,
    idx: int,
    result: Dict[str, Any],
    sr: int,
) -> None:
    """Write one sample to disk."""
    sample_dir = output_dir / tier / category / f"{idx:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sf.write(str(sample_dir / "primary.wav"),   result["primary"],   sr, subtype="PCM_16")
    sf.write(str(sample_dir / "reference.wav"), result["reference"], sr, subtype="PCM_16")
    sf.write(str(sample_dir / "clean.wav"),     result["clean"],     sr, subtype="PCM_16")

    meta = {
        "sample_id":      f"{idx:06d}",
        "snr_target":     float(result["snr_target"]),
        "snr_actual":     float(result["snr_actual"]),
        "snr_tier":       result["snr_tier"],
        "noise_class":    result["noise_class"],
        "rir_applied":    result["rir_applied"],
        "speech_path":    result["speech_path"],
        "noise_path":     result["noise_path"],
        "impulse_labels": result["impulse_labels"].tolist(),
        "sample_rate":    sr,
    }
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DataLoader collate function for speech enhancement samples.

    Stacks tensor values; collects strings and scalars as lists.

    Args:
        batch: List of sample dicts from ``__getitem__``.

    Returns:
        Batched dict with tensors stacked along dim 0.
    """
    out: Dict[str, Any] = {}
    for key in batch[0]:
        vals = [item[key] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out
