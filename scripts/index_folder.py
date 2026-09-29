#!/usr/bin/env python3
"""
CLI script to scan a folder of audio files and produce a manifest CSV.

Usage
-----
    python scripts/index_folder.py \\
        --folder /path/to/audio \\
        --category impulsive \\
        --source kaggle_gunshot \\
        --license CC0 \\
        --out manifests/gunshot_index.csv

Near-duplicate detection
------------------------
For each file, a 64-bit spectral hash is computed using:
  1. Load 1 second of audio (or full file if shorter)
  2. Compute power spectrum via numpy FFT (no librosa)
  3. Split into n_bins equally-spaced frequency bins
  4. Threshold at median -> binary vector
  5. Pack bits to a 64-bit integer

Pairs whose Hamming distance is below --hamming-threshold are reported
as potential near-duplicates.
"""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from anc.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("index_folder")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_EXTENSIONS: Tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")
_N_HASH_BINS: int = 64  # must be <= 64 for int packing
_LOAD_SECONDS: float = 1.0  # seconds to read for hashing

# ---------------------------------------------------------------------------
# Category inference helpers
# ---------------------------------------------------------------------------

_KEYWORD_MAP = {
    "impulsive": ["gunshot", "gun_shot", "explosion", "blast", "artillery", "shot", "firearm"],
    "rotor":     ["helicopter", "drone", "rotor", "propeller"],
    "vehicle":   ["engine", "vehicle", "car", "truck", "idling"],
    "siren":     ["siren", "alarm", "emergency", "horn"],
    "wind":      ["wind", "air", "breeze"],
    "crowd":     ["crowd", "urban", "street", "jackhammer", "drilling", "construction"],
}


def _infer_category_from_path(path: Path) -> str:
    """Infer noise category from path components using keyword matching.

    Args:
        path: Path to the audio file.

    Returns:
        Category string (e.g. 'impulsive', 'misc').
    """
    p_str = str(path).lower()
    for cat, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in p_str:
                return cat
    return "misc"


def _load_noise_mapping_yaml() -> dict:
    """Try to load configs/noise_mapping.yaml for category inference.

    Returns:
        Dict with category keyword lists, or empty dict on failure.
    """
    try:
        import yaml
        yaml_path = _REPO / "configs" / "noise_mapping.yaml"
        if yaml_path.exists():
            with yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return data
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load noise_mapping.yaml: %s", exc)
    return {}


# ---------------------------------------------------------------------------
# Spectral hash (pure numpy, no librosa)
# ---------------------------------------------------------------------------

def _build_mel_filterbank(n_fft: int, sr: int, n_bins: int) -> np.ndarray:
    """Build a simple triangular mel filterbank matrix.

    Args:
        n_fft: Number of FFT points (i.e. n_fft // 2 + 1 freq bins).
        sr:    Sample rate in Hz.
        n_bins: Number of mel filter bins.

    Returns:
        Float64 matrix of shape (n_bins, n_fft // 2 + 1).
    """
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_lo = hz_to_mel(0.0)
    mel_hi = hz_to_mel(sr / 2.0)
    mel_points = np.linspace(mel_lo, mel_hi, n_bins + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor(hz_points / (sr / n_fft)).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)

    fb = np.zeros((n_bins, n_freqs), dtype=np.float64)
    for m in range(1, n_bins + 1):
        f_m_minus = bin_points[m - 1]
        f_m       = bin_points[m]
        f_m_plus  = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            denom = f_m - f_m_minus
            if denom > 0:
                fb[m - 1, k] = (k - f_m_minus) / denom

        for k in range(f_m, f_m_plus + 1):
            denom = f_m_plus - f_m
            if denom > 0:
                fb[m - 1, k] = (f_m_plus - k) / denom

    return fb


def compute_spectral_hash_from_array(
    audio: np.ndarray,
    sr: int,
    n_bins: int = _N_HASH_BINS,
) -> int:
    """Compute a 64-bit spectral hash from a 1-D audio array.

    Steps:
      1. Compute power spectrum via numpy FFT.
      2. Apply triangular mel filterbank.
      3. Threshold at median -> binary vector of length n_bins.
      4. Pack bits into a 64-bit integer.

    Args:
        audio:  1-D float audio array (mono).
        sr:     Sample rate in Hz.
        n_bins: Number of frequency bins (<=64).

    Returns:
        64-bit integer hash, or -1 if an error occurred.
    """
    assert n_bins <= 64, "n_bins must be <= 64 for int packing"
    try:
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float64)

        n_fft = 512
        # Pad or truncate to n_fft for a single frame
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        else:
            audio = audio[:n_fft]

        spectrum = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2  # power spectrum

        fb = _build_mel_filterbank(n_fft, sr, n_bins)
        mel_energy = fb @ spectrum  # shape (n_bins,)

        median_val = np.median(mel_energy)
        bits = (mel_energy > median_val).astype(np.uint64)

        # Pack bits into a single 64-bit integer
        hash_val = np.uint64(0)
        for i, bit in enumerate(bits[:64]):
            hash_val |= np.uint64(int(bit)) << np.uint64(i)

        return int(hash_val)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Spectral hash computation failed: %s", exc)
        return -1


def _load_one_second(filepath: Path, sr_target: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    """Load at most _LOAD_SECONDS of audio from a file.

    Args:
        filepath:  Path to the audio file.
        sr_target: Target sample rate (for approximate frame count).

    Returns:
        Tuple (audio_array, actual_sr).  audio_array is None on error.
    """
    try:
        info = sf.info(str(filepath))
        actual_sr = info.samplerate
        max_frames = int(min(_LOAD_SECONDS, float(info.frames) / actual_sr) * actual_sr)
        audio, actual_sr = sf.read(
            str(filepath),
            frames=max(1, max_frames),
            dtype="float32",
            always_2d=False,
        )
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float64), actual_sr
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load %s: %s", filepath, exc)
        return None, 0


# ---------------------------------------------------------------------------
# Hamming distance helpers
# ---------------------------------------------------------------------------

def _hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two 64-bit integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Number of differing bits (0-64).
    """
    xor = (a ^ b) & 0xFFFFFFFFFFFFFFFF
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def _find_near_duplicates(
    hashes: List[int],
    threshold: int = 5,
) -> List[Tuple[int, int, int]]:
    """Find pairs with Hamming distance below threshold.

    Uses a bucket grouping strategy: exact matches are found first, then
    near matches.  For datasets larger than ~10k files this is approximate.

    Args:
        hashes:    List of 64-bit integer hashes.
        threshold: Maximum Hamming distance to report.

    Returns:
        List of (idx_a, idx_b, hamming_distance) tuples.
    """
    n = len(hashes)
    pairs: List[Tuple[int, int, int]] = []

    # Exact-match bucket first (distance == 0)
    from collections import defaultdict
    bucket: dict = defaultdict(list)
    for i, h in enumerate(hashes):
        if h != -1:
            bucket[h].append(i)

    for h_val, indices in bucket.items():
        if len(indices) > 1:
            for ia in range(len(indices)):
                for ib in range(ia + 1, len(indices)):
                    pairs.append((indices[ia], indices[ib], 0))

    # Near-match: only compare within the valid hashes (O(n^2) but threshold is tiny)
    valid_idx = [i for i, h in enumerate(hashes) if h != -1]
    if n > 5000:
        logger.warning(
            "Near-duplicate check over %d files is O(n^2); this may be slow.", n
        )

    for ii in range(len(valid_idx)):
        ia = valid_idx[ii]
        for jj in range(ii + 1, len(valid_idx)):
            ib = valid_idx[jj]
            dist = _hamming_distance(hashes[ia], hashes[ib])
            if 0 < dist < threshold:
                pairs.append((ia, ib, dist))

    return pairs


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_folder(
    folder: Path,
    category: Optional[str],
    source: str,
    license_str: str,
) -> pd.DataFrame:
    """Recursively scan a folder and build a manifest DataFrame.

    Args:
        folder:      Root directory to scan.
        category:    Forced noise category string, or None to infer per file.
        source:      Source label (e.g. 'kaggle_gunshot').
        license_str: License string (e.g. 'CC0', 'CC-BY').

    Returns:
        DataFrame with columns:
            path, category, source, original_class, duration, license, is_synthetic.
    """
    rows = []
    skipped = 0

    files: List[Path] = []
    for ext in VALID_EXTENSIONS:
        files.extend(folder.rglob(f"*{ext}"))
    files = sorted(files)

    logger.info("Found %d audio files under %s", len(files), folder)

    hashes: List[int] = []

    for filepath in files:
        try:
            info = sf.info(str(filepath))
            duration = float(info.frames) / float(info.samplerate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SKIP (corrupt/unreadable) %s: %s", filepath.name, exc)
            skipped += 1
            hashes.append(-1)
            continue

        cat = category if category else _infer_category_from_path(filepath)

        # Original class: leaf folder name (reasonable default)
        original_class = filepath.parent.name

        rows.append({
            "path":           str(filepath),
            "category":       cat,
            "source":         source,
            "original_class": original_class,
            "duration":       round(duration, 4),
            "license":        license_str,
            "is_synthetic":   False,
        })

        # Compute hash
        audio, sr = _load_one_second(filepath)
        if audio is not None:
            hashes.append(compute_spectral_hash_from_array(audio, sr))
        else:
            hashes.append(-1)

    df = pd.DataFrame(rows)
    logger.info("Scanned %d valid files, skipped %d corrupted.", len(df), skipped)
    return df, hashes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the index_folder CLI."""
    parser = argparse.ArgumentParser(
        description="Index a folder of audio files into a manifest CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--folder", required=True,
        help="Root folder to scan recursively for audio files.",
    )
    parser.add_argument(
        "--category", default=None,
        help="Noise category (e.g. impulsive, rotor). "
             "If omitted, inferred from filename/path.",
    )
    parser.add_argument(
        "--source", default="manual",
        help="Source label to embed in manifest (e.g. 'kaggle_gunshot').",
    )
    parser.add_argument(
        "--license", dest="license_str", default="unknown",
        help="License string (e.g. CC0, CC-BY, proprietary).",
    )
    parser.add_argument(
        "--out", default="manifest.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--hamming-threshold", type=int, default=5,
        help="Maximum Hamming distance to flag as near-duplicate (default 5).",
    )
    parser.add_argument(
        "--no-hash", action="store_true",
        help="Skip spectral hashing (faster for very large collections).",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"[ERROR] Folder does not exist: {folder}")
        sys.exit(1)
    if not folder.is_dir():
        print(f"[ERROR] Path is not a directory: {folder}")
        sys.exit(1)

    print(f"\n[index_folder] Scanning: {folder}")
    print(f"  category        : {args.category or '(inferred per file)'}")
    print(f"  source          : {args.source}")
    print(f"  license         : {args.license_str}")
    print(f"  output          : {args.out}")
    print(f"  hamming thresh  : {args.hamming_threshold}")
    print()

    df, hashes = scan_folder(folder, args.category, args.source, args.license_str)

    if df.empty:
        print("[WARNING] No valid audio files found. Manifest not saved.")
        sys.exit(0)

    # --- Summary table ---
    print("\n===== Manifest Summary =====")
    print(f"  Total files     : {len(df)}")
    print(f"  Total duration  : {df['duration'].sum():.1f} s  "
          f"({df['duration'].sum() / 3600:.2f} h)")
    print(f"\n  By category:")
    for cat, count in df["category"].value_counts().items():
        dur = df.loc[df["category"] == cat, "duration"].sum()
        print(f"    {cat:<20s}: {count:5d} files,  {dur:8.1f} s")
    print()

    # --- Near-duplicate detection ---
    if not args.no_hash and hashes:
        print("[index_folder] Running near-duplicate detection ...")
        near_dups = _find_near_duplicates(hashes, threshold=args.hamming_threshold)

        if near_dups:
            print(f"\n[WARNING] Found {len(near_dups)} potential near-duplicate pair(s):")
            for ia, ib, dist in near_dups[:20]:
                path_a = df.iloc[ia]["path"] if ia < len(df) else "N/A"
                path_b = df.iloc[ib]["path"] if ib < len(df) else "N/A"
                print(f"  hamming={dist:2d}  {Path(path_a).name}  <->  {Path(path_b).name}")
            if len(near_dups) > 20:
                print(f"  ... and {len(near_dups) - 20} more pairs (see CSV for full list).")
        else:
            print("  No near-duplicates found (hamming threshold=%d)." % args.hamming_threshold)

        # Embed hash into df for downstream use
        hash_col: List[int] = []
        h_iter = iter(hashes)
        for _ in range(len(df)):
            hash_col.append(next(h_iter, -1))
        df["spec_hash"] = hash_col
    else:
        df["spec_hash"] = -1

    # --- Save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[index_folder] Manifest saved: {out_path}  ({len(df)} rows)\n")


if __name__ == "__main__":
    main()
