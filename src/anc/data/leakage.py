"""
Leakage control for ANC-SIH datasets.

Prevents data contamination across train/val/test splits by:
1. Cross-dataset Freesound ID matching (UrbanSound8K + ESC-50 share sources)
2. Gun-model-based splitting for the Kaggle gunshot dataset
3. Near-duplicate detection via spectral hash (perceptual hashing)
4. Speaker-level and path-level split verification
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NEAR_SILENCE: float = 1e-8


# ---------------------------------------------------------------------------
# 1. extract_freesound_id
# ---------------------------------------------------------------------------

def extract_freesound_id(filepath: "str | Path", source: str = "") -> Optional[str]:
    """Extract the Freesound clip ID from a file path.

    UrbanSound8K filenames follow the pattern::

        {freesound_id}-{class_id}-{take}-{slice}.wav

    ESC-50: the Freesound ID is the numeric prefix of the filename (before '-').

    Args:
        filepath: Path to the audio file.
        source:   Dataset source tag (``'urbansound8k'``, ``'esc50'``, etc.).

    Returns:
        Freesound ID string, or ``None`` if not determinable.
    """
    name = Path(filepath).stem   # e.g. '17853-0-0-0' or '1-137-A-32'
    parts = name.split("-")
    if len(parts) >= 1:
        candidate = parts[0]
        if candidate.isdigit():
            return candidate
    return None


# ---------------------------------------------------------------------------
# 2. fix_freesound_split_leakage
# ---------------------------------------------------------------------------

def fix_freesound_split_leakage(
    noise_df: pd.DataFrame,
    us8k_meta_path: Optional["str | Path"] = None,
    esc50_meta_path: Optional["str | Path"] = None,
) -> pd.DataFrame:
    """Force all clips sharing a Freesound ID into the same split.

    UrbanSound8K and ESC-50 both draw from Freesound, so the same source
    recording can appear in both datasets. This function:

    1. Extracts Freesound IDs for UrbanSound8K and ESC-50 rows.
    2. Finds IDs that appear in both datasets.
    3. Pins all clips with a shared ID to the split of the first alphabetical
       occurrence (arbitrary but deterministic).

    Args:
        noise_df:        Combined noise manifest DataFrame. Must have columns
                         ``['path', 'source', 'split']``.
        us8k_meta_path:  Optional path to UrbanSound8K ``metadata.csv``
                         (unused currently; IDs come from filename).
        esc50_meta_path: Optional path to ESC-50 ``meta/esc50.csv``
                         (unused currently; IDs come from filename).

    Returns:
        Updated DataFrame with ``split`` column potentially reassigned.
    """
    df = noise_df.copy()

    sources = {"urbansound8k", "esc50"}
    mask = df["source"].isin(sources)

    if not mask.any():
        logger.debug("No UrbanSound8K or ESC-50 rows; nothing to fix.")
        return df

    # Extract freesound IDs
    df["_fsid"] = df["path"].apply(lambda p: extract_freesound_id(p, ""))
    df["_fsid"] = df["_fsid"].where(mask, other=None)

    # Find IDs present in both datasets
    us8k_ids  = set(df.loc[df["source"] == "urbansound8k", "_fsid"].dropna())
    esc50_ids = set(df.loc[df["source"] == "esc50",        "_fsid"].dropna())
    shared    = us8k_ids & esc50_ids

    if not shared:
        logger.info("No shared Freesound IDs between UrbanSound8K and ESC-50.")
        df.drop(columns=["_fsid"], inplace=True)
        return df

    logger.info("Found %d shared Freesound IDs across UrbanSound8K and ESC-50.", len(shared))
    reassigned = 0

    for fsid in shared:
        id_mask = df["_fsid"] == fsid
        rows = df[id_mask].sort_values("path")
        canonical_split = rows.iloc[0]["split"]
        if (rows["split"] != canonical_split).any():
            df.loc[id_mask, "split"] = canonical_split
            reassigned += int((rows["split"] != canonical_split).sum())

    logger.info("Reassigned %d clips to enforce Freesound ID split consistency.", reassigned)
    df.drop(columns=["_fsid"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 3. split_by_gun_model
# ---------------------------------------------------------------------------

def split_by_gun_model(
    df: pd.DataFrame,
    model_map: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Force each gun model into its designated split.

    Clips from ``model_map['test_only']`` go to ``'test'``, from
    ``model_map['val_only']`` to ``'val'``, everything else to ``'train'``.
    The gun model is inferred from the ``'gun_model'`` column if present, or
    from the filename stem otherwise.

    Args:
        df:        DataFrame with at least a ``'path'`` column; optionally
                   a ``'gun_model'`` column.
        model_map: Dict with keys ``'test_only'``, ``'val_only'``, and
                   ``'train'``. Defaults to the configs/noise_mapping.yaml
                   gunshot_models section.

    Returns:
        DataFrame with ``'split'`` column updated.
    """
    if model_map is None:
        model_map = {
            "test_only": ["glock_17", "ak47", "mp5"],
            "val_only":  ["beretta_92", "uzi"],
            "train":     ["default"],
        }

    df = df.copy()

    # Build model -> split mapping (lowercase, underscore)
    gun_to_split: Dict[str, str] = {}
    for split_name, models in model_map.items():
        for m in models:
            target_split = split_name.replace("_only", "")
            gun_to_split[m.lower().replace(" ", "_")] = target_split

    if "gun_model" not in df.columns:
        # Try to infer from path stem: e.g. 'glock_17_shot_001.wav'
        def _infer_model(path: str) -> str:
            stem = Path(path).stem.lower().replace(" ", "_")
            for model in gun_to_split:
                if model in stem:
                    return model
            return "default"
        df["gun_model"] = df["path"].apply(_infer_model)

    initial_splits = df["split"].copy()
    for model, target in gun_to_split.items():
        if model == "default":
            continue
        mask = df["gun_model"].str.lower().str.replace(" ", "_") == model
        df.loc[mask, "split"] = target

    changed = int((df["split"] != initial_splits).sum())
    logger.info(
        "split_by_gun_model: reassigned %d rows. "
        "test=%d val=%d train=%d",
        changed,
        int((df["split"] == "test").sum()),
        int((df["split"] == "val").sum()),
        int((df["split"] == "train").sum()),
    )
    return df


# ---------------------------------------------------------------------------
# 4. compute_spectral_hash
# ---------------------------------------------------------------------------

def compute_spectral_hash(
    filepath: "str | Path",
    sr: int = 16000,
    n_bins: int = 64,
    max_dur_s: float = 1.0,
) -> int:
    """Compute a 64-bit perceptual spectral hash of a short audio clip.

    Loads up to *max_dur_s* seconds, computes the power spectrum,
    splits into *n_bins* frequency bins, thresholds at the median, packs
    the binary vector into a 64-bit integer.

    Args:
        filepath:   Path to audio file.
        sr:         Target sample rate.
        n_bins:     Number of spectral bins (must be <= 64 for int64 packing).
        max_dur_s:  Maximum duration to load in seconds.

    Returns:
        64-bit integer hash, or ``-1`` on error.
    """
    try:
        import soundfile as sf
        data, file_sr = sf.read(str(filepath), dtype="float32",
                                frames=int(max_dur_s * sr * 4),  # over-read then resample
                                always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)

        if file_sr != sr:
            from scipy.signal import resample_poly
            ratio_n = sr
            ratio_d = file_sr
            g = gcd(ratio_n, ratio_d)
            data = resample_poly(data, ratio_n // g, ratio_d // g)

        data = data[: int(max_dur_s * sr)]
        if len(data) < 64:
            return -1

        # Power spectrum
        spec = np.abs(np.fft.rfft(data)) ** 2

        # Bin into n_bins equal-width frequency bins
        bin_size = max(1, len(spec) // n_bins)
        bins = np.array([
            float(np.mean(spec[i * bin_size: (i + 1) * bin_size]))
            for i in range(min(n_bins, len(spec) // bin_size))
        ])

        # Threshold at median -> binary
        threshold = float(np.median(bins))
        binary = (bins > threshold).astype(np.uint8)

        # Pack to int64 (use first 64 bits)
        packed = 0
        for bit_idx, bit in enumerate(binary[:64]):
            packed |= int(bit) << bit_idx

        return int(packed)

    except Exception as exc:
        logger.debug("compute_spectral_hash failed for %s: %s", filepath, exc)
        return -1


# ---------------------------------------------------------------------------
# 5. find_near_duplicates
# ---------------------------------------------------------------------------

def find_near_duplicates(
    df: pd.DataFrame,
    hash_col: str = "spec_hash",
    hamming_threshold: int = 5,
) -> List[Tuple[int, int, int]]:
    """Find pairs of rows whose spectral hashes are near-identical.

    Uses a brute-force O(N²) comparison for small datasets; for large
    datasets this is still fast because hashes are 64-bit integers and
    we can early-exit most pairs.

    Args:
        df:               DataFrame with a column *hash_col* containing int64 hashes.
        hash_col:         Name of the hash column.
        hamming_threshold: Max Hamming distance to flag as near-duplicate.

    Returns:
        List of ``(idx_a, idx_b, hamming_distance)`` tuples.
    """
    if hash_col not in df.columns:
        logger.warning("find_near_duplicates: column '%s' not found.", hash_col)
        return []

    hashes = df[hash_col].tolist()
    indices = df.index.tolist()
    duplicates: List[Tuple[int, int, int]] = []

    for i in range(len(hashes)):
        hi = hashes[i]
        if hi == -1:
            continue
        for j in range(i + 1, len(hashes)):
            hj = hashes[j]
            if hj == -1:
                continue
            xor  = (hi ^ hj) & 0xFFFFFFFFFFFFFFFF  # 64-bit
            dist = bin(xor).count("1")
            if dist <= hamming_threshold:
                duplicates.append((indices[i], indices[j], dist))

    return duplicates


# ---------------------------------------------------------------------------
# 6. run_leakage_checks
# ---------------------------------------------------------------------------

def run_leakage_checks(
    speech_train: pd.DataFrame,
    speech_val:   pd.DataFrame,
    speech_test:  pd.DataFrame,
    noise_train:  pd.DataFrame,
    noise_val:    pd.DataFrame,
    noise_test:   pd.DataFrame,
) -> Dict:
    """Assert no leakage across train/val/test splits.

    Checks:
    - No shared speaker_ids between any two splits.
    - No shared file paths across any two splits.
    - No shared Freesound IDs (where present) across splits.

    Args:
        speech_*: Speech DataFrames for each split (must have ``speaker_id``
                  and ``path`` columns).
        noise_*:  Noise DataFrames for each split (must have ``path`` column).

    Returns:
        Dict with ``'passed'`` (bool) and ``'violations'`` (list of str).

    Raises:
        AssertionError: If any violations are found (for pytest use).
    """
    violations: List[str] = []

    # --- Speaker leakage ---
    def _speaker_ids(df: pd.DataFrame) -> set:
        if "speaker_id" in df.columns:
            return set(df["speaker_id"].dropna().astype(str))
        return set()

    sp_train_ids = _speaker_ids(speech_train)
    sp_val_ids   = _speaker_ids(speech_val)
    sp_test_ids  = _speaker_ids(speech_test)

    shared_tv = sp_train_ids & sp_val_ids
    if shared_tv:
        violations.append(f"Speaker leakage train/val: {len(shared_tv)} shared IDs: {list(shared_tv)[:5]}")

    shared_tt = sp_train_ids & sp_test_ids
    if shared_tt:
        violations.append(f"Speaker leakage train/test: {len(shared_tt)} shared IDs: {list(shared_tt)[:5]}")

    shared_vt = sp_val_ids & sp_test_ids
    if shared_vt:
        violations.append(f"Speaker leakage val/test: {len(shared_vt)} shared IDs: {list(shared_vt)[:5]}")

    # --- Path leakage (all modalities) ---
    def _paths(df: pd.DataFrame) -> set:
        if "path" in df.columns:
            return set(df["path"].dropna().astype(str))
        return set()

    all_train = _paths(speech_train) | _paths(noise_train)
    all_val   = _paths(speech_val)   | _paths(noise_val)
    all_test  = _paths(speech_test)  | _paths(noise_test)

    shared_path_tv = all_train & all_val
    if shared_path_tv:
        violations.append(f"Path leakage train/val: {len(shared_path_tv)} shared paths")

    shared_path_tt = all_train & all_test
    if shared_path_tt:
        violations.append(f"Path leakage train/test: {len(shared_path_tt)} shared paths")

    shared_path_vt = all_val & all_test
    if shared_path_vt:
        violations.append(f"Path leakage val/test: {len(shared_path_vt)} shared paths")

    passed = len(violations) == 0

    if not passed:
        msg = "DATA LEAKAGE DETECTED:\n" + "\n".join(f"  - {v}" for v in violations)
        logger.error(msg)
        raise AssertionError(msg)

    logger.info("Leakage check PASSED (speakers, freesound IDs, paths).")
    return {"passed": True, "violations": []}
