"""
Manifest building and loading for the ANC-SIH data pipeline.

Manifests are CSV files describing available speech and noise files,
with columns: path, source, type, category, speaker_id, split, duration.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_EXTENSIONS: set = {".wav", ".flac", ".mp3", ".ogg"}

NOISE_KEYWORDS: Dict[str, List[str]] = {
    "impulsive": ["gunshot", "gun_shot", "explosion", "blast", "artillery", "shot", "firearm"],
    "rotor":     ["helicopter", "drone", "rotor", "propeller"],
    "vehicle":   ["engine", "vehicle", "car", "truck", "idling", "engine_idling"],
    "siren":     ["siren", "alarm", "emergency", "horn"],
    "wind":      ["wind", "air", "breeze"],
    "crowd":     ["crowd", "urban", "street", "jackhammer", "drilling", "construction"],
    "misc":      [],
}

# UrbanSound8K class name → our category
_US8K_CLASS_MAP: Dict[str, str] = {
    "gun_shot":     "impulsive",
    "siren":        "siren",
    "engine_idling":"vehicle",
    "jackhammer":   "crowd",
    "drilling":     "crowd",
    "street_music": "crowd",
    "children_playing": "crowd",
    "dog_bark":     "misc",
    "car_horn":     "siren",
    "air_conditioner": "vehicle",
}

# ESC-50 categories → our categories (partial mapping)
_ESC50_MAP: Dict[str, str] = {
    "fireworks":    "impulsive",
    "gunshot":      "impulsive",
    "thunderstorm": "impulsive",
    "helicopter":   "rotor",
    "engine":       "vehicle",
    "train":        "vehicle",
    "car_horn":     "siren",
    "siren":        "siren",
    "wind":         "wind",
    "rain":         "wind",
    "crickets":     "misc",
    "crow":         "misc",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def categorize_noise(path: str | Path, source: str = "") -> str:
    """Determine noise category from file path and source name.

    Checks path components (folder names, filename stem) against
    ``NOISE_KEYWORDS``.  Source-specific mappings are applied first.

    Args:
        path:   Absolute or relative path to the audio file.
        source: Dataset source name (e.g. 'urbansound8k', 'esc50').

    Returns:
        Category string, one of the keys in ``NOISE_KEYWORDS``.
    """
    p = Path(path)
    p_str = str(p).lower()
    stem = p.stem.lower()

    # UrbanSound8K: class name is embedded in the path (foldN/ subdirs)
    # US8K filenames end with -<classID>.wav; folder names are the class names
    if source == "urbansound8k":
        for us8k_cls, cat in _US8K_CLASS_MAP.items():
            if us8k_cls in p_str:
                return cat

    # ESC-50: category embedded in filename suffix
    if source == "esc50":
        for esc_cls, cat in _ESC50_MAP.items():
            if esc_cls in stem:
                return cat

    # General keyword matching
    for cat, keywords in NOISE_KEYWORDS.items():
        if cat == "misc":
            continue
        for kw in keywords:
            if kw in p_str:
                return cat

    return "misc"


def get_audio_duration(path: str | Path, sr: int = 16000) -> float:
    """Return duration of an audio file in seconds without loading samples.

    Args:
        path: Path to the audio file.
        sr:   Not used directly; present for API consistency.

    Returns:
        Duration in seconds, or 0.0 if the file cannot be read.
    """
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception as exc:
        logger.warning("Could not read duration of %s: %s", path, exc)
        return 0.0


def _scan_audio_files(root: Path) -> List[Path]:
    """Recursively collect all audio files under *root*."""
    files: List[Path] = []
    for ext in VALID_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def _speaker_from_path(path: Path, source: str) -> str:
    """Extract a speaker identifier from a file path.

    Args:
        path:   Path to the audio file.
        source: Dataset source name.

    Returns:
        String speaker identifier.
    """
    if source == "librispeech":
        # LibriSpeech structure: <root>/<split>/<speaker_id>/<chapter_id>/<file>
        parts = path.parts
        # Walk from root: find the numeric speaker folder
        for part in parts:
            if part.isdigit():
                return part
        return path.parent.parent.name
    if source == "vctk":
        # VCTK filenames: p<NUM>_<utt>.wav
        stem = path.stem
        if stem.startswith("p") and "_" in stem:
            return stem.split("_")[0]
        return path.parent.name
    # Default: use parent folder name as speaker
    return path.parent.name


def _assign_splits(
    items: List,
    split_ratios: Dict[str, float],
    seed: int = 42,
) -> List[str]:
    """Randomly assign split labels to a list of items.

    Args:
        items:        List of items (speakers or file paths).
        split_ratios: Dict with keys 'train', 'val', 'test' summing to ~1.0.
        seed:         Random seed for reproducibility.

    Returns:
        List of split labels, same length as *items*.
    """
    rng = random.Random(seed)
    items_shuffled = list(items)
    rng.shuffle(items_shuffled)

    n = len(items_shuffled)
    n_train = int(split_ratios.get("train", 0.8) * n)
    n_val   = int(split_ratios.get("val",   0.1) * n)

    splits: Dict = {}
    for i, item in enumerate(items_shuffled):
        if i < n_train:
            splits[item] = "train"
        elif i < n_train + n_val:
            splits[item] = "val"
        else:
            splits[item] = "test"

    return [splits.get(item, "test") for item in items]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_speech_manifest(
    root_dir: str | Path,
    source: str,
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a manifest DataFrame for a speech dataset.

    Splits files **by speaker** to prevent data leakage between splits.

    Args:
        root_dir:     Root directory of the speech dataset.
        source:       Dataset identifier (e.g. 'librispeech', 'vctk').
        split_ratios: Dict with 'train'/'val'/'test' fractions.
        seed:         Random seed for reproducible splits.

    Returns:
        DataFrame with columns:
            path, source, type, category, speaker_id, split, duration.
    """
    split_ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    root_dir = Path(root_dir)
    if not root_dir.exists():
        logger.warning("Speech dir does not exist: %s", root_dir)
        return pd.DataFrame()

    files = _scan_audio_files(root_dir)
    if not files:
        logger.warning("No audio files found in %s", root_dir)
        return pd.DataFrame()

    # Group by speaker
    speaker_to_files: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        spk = _speaker_from_path(f, source)
        speaker_to_files[spk].append(f)

    speakers = sorted(speaker_to_files.keys())
    speaker_splits = _assign_splits(speakers, split_ratios, seed=seed)
    spk_split_map = dict(zip(speakers, speaker_splits))

    rows = []
    for spk, spk_files in speaker_to_files.items():
        for f in spk_files:
            rows.append({
                "path":       str(f),
                "source":     source,
                "type":       "speech",
                "category":   "speech",
                "speaker_id": spk,
                "split":      spk_split_map[spk],
                "duration":   get_audio_duration(f),
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Speech manifest: %d files, %d speakers — %s",
        len(df), len(speakers), root_dir,
    )
    return df


def build_noise_manifest(
    root_dir: str | Path,
    source: str,
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a manifest DataFrame for a noise dataset.

    Splits files **by file** (no speaker grouping).

    Args:
        root_dir:     Root directory of the noise dataset.
        source:       Dataset identifier.
        split_ratios: Dict with 'train'/'val'/'test' fractions.
        seed:         Random seed.

    Returns:
        DataFrame with columns:
            path, source, type, category, split, duration, license.
    """
    split_ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    root_dir = Path(root_dir)

    if not root_dir.exists():
        logger.warning("Noise dir does not exist: %s", root_dir)
        return pd.DataFrame()

    # Special handling for US8K metadata
    us8k_meta: Optional[pd.DataFrame] = None
    if source == "urbansound8k":
        meta_path = root_dir / "metadata" / "UrbanSound8K.csv"
        if meta_path.exists():
            us8k_meta = pd.read_csv(meta_path)
            us8k_meta["slice_file_name"] = us8k_meta["slice_file_name"].astype(str)

    # Special handling for ESC-50 metadata
    esc50_meta: Optional[pd.DataFrame] = None
    if source == "esc50":
        meta_path = root_dir / "meta" / "esc50.csv"
        if meta_path.exists():
            esc50_meta = pd.read_csv(meta_path)

    files = _scan_audio_files(root_dir)
    if not files:
        logger.warning("No audio files found in %s", root_dir)
        return pd.DataFrame()

    rows = []
    for f in files:
        cat = "misc"
        lic = "unknown"

        if us8k_meta is not None:
            row = us8k_meta[us8k_meta["slice_file_name"] == f.name]
            if not row.empty:
                class_name = row.iloc[0].get("class", "")
                cat = _US8K_CLASS_MAP.get(str(class_name).lower(), "misc")
        elif esc50_meta is not None:
            row = esc50_meta[esc50_meta["filename"] == f.name]
            if not row.empty:
                class_name = str(row.iloc[0].get("category", "")).lower()
                cat = _ESC50_MAP.get(class_name, categorize_noise(f, source))
                lic = "CC BY-NC"
        else:
            cat = categorize_noise(f, source)

        rows.append({
            "path":     str(f),
            "source":   source,
            "type":     "noise",
            "category": cat,
            "split":    "train",   # placeholder; assigned below
            "duration": get_audio_duration(f),
            "license":  lic,
        })

    df = pd.DataFrame(rows)
    # Assign splits by file
    file_list = df["path"].tolist()
    splits = _assign_splits(file_list, split_ratios, seed=seed)
    df["split"] = splits

    logger.info(
        "Noise manifest: %d files — %s — categories: %s",
        len(df), root_dir,
        df["category"].value_counts().to_dict(),
    )
    return df


def build_synthetic_manifest(
    split: str = "train",
    n_per_category: int = 100,
    sr: int = 16000,
) -> pd.DataFrame:
    """Build a manifest entry for synthetic noise (no real files).

    Args:
        split:          Which split this synthetic data belongs to.
        n_per_category: Number of virtual clips per category.
        sr:             Sample rate (for documentation only).

    Returns:
        DataFrame with path='synthetic::<category>::<idx>', license='synthetic-cc0'.
    """
    from anc.data.synthetic_noise import SYNTHETIC_GENERATORS

    rows = []
    for cat in SYNTHETIC_GENERATORS:
        for i in range(n_per_category):
            rows.append({
                "path":     f"synthetic::{cat}::{i}",
                "source":   "synthetic",
                "type":     "noise",
                "category": cat,
                "split":    split,
                "duration": 4.0,
                "license":  "synthetic-cc0",
            })
    return pd.DataFrame(rows)


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Save a manifest DataFrame to CSV.

    Args:
        df:   DataFrame to save.
        path: Destination CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Manifest saved: %s (%d rows)", path, len(df))


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a manifest CSV into a DataFrame.

    Args:
        path: CSV path to load.

    Returns:
        DataFrame, or empty DataFrame if file does not exist.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Manifest not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info("Manifest loaded: %s (%d rows)", path, len(df))
    return df


def merge_manifests(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Concatenate multiple manifests into one, resetting the index.

    Args:
        *dfs: Variable number of DataFrames to merge.

    Returns:
        Combined DataFrame with reset integer index.
    """
    non_empty = [df for df in dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)
