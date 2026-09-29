#!/usr/bin/env python3
"""
Extended download script for ANC-SIH (Phase 4).

Adds the following new data sources vs Phase 0 (download_data.py):
  --kaggle-urbansound8k  : chrisfilo/urbansound8k via Kaggle API
  --kaggle-gunshot       : emrahaydemr/gunshot-audio-dataset via Kaggle API
  --belize               : Belize gunshot dataset (Zenodo record 4993368)
  --musan                : MUSAN corpus (OpenSLR-17, auto-download)
  --voice-bank           : VoiceBank-DEMAND (Edinburgh DataShare, manual)
  --build-manifest       : Rebuild all manifests from downloaded data

Usage
-----
    python scripts/download_data_v2.py --config configs/base.yaml --musan
    python scripts/download_data_v2.py --config configs/base.yaml --all
    python scripts/download_data_v2.py --config configs/base.yaml --build-manifest

Kaggle API setup (3 steps)
---------------------------
1. Go to https://www.kaggle.com/settings and create an API token (Download JSON)
2. Place kaggle.json at ~/.kaggle/kaggle.json (Linux/Mac) or
   C:/Users/USER/.kaggle/kaggle.json (Windows)
3. Install kagglehub: pip install kagglehub

For Google Colab:
1. Run: from google.colab import files; files.upload()  (upload kaggle.json)
2. Run: import os; os.makedirs('/root/.kaggle', exist_ok=True); !cp kaggle.json /root/.kaggle/
3. Run: pip install kagglehub
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from anc.utils.config import load_config, Config  # noqa: E402
from anc.utils.seed import set_seed               # noqa: E402
from anc.utils.logging_utils import get_logger    # noqa: E402

logger = get_logger("download_data_v2")

# ---------------------------------------------------------------------------
# Kaggle setup instructions (ASCII-only)
# ---------------------------------------------------------------------------

KAGGLE_SETUP_INSTRUCTIONS: str = """
Kaggle API setup (3 steps):
1. Go to https://www.kaggle.com/settings and create an API token (Download JSON)
2. Place kaggle.json at ~/.kaggle/kaggle.json (Linux/Mac) or
   C:/Users/USER/.kaggle/kaggle.json (Windows)
3. Install kagglehub: pip install kagglehub

For Google Colab:
1. Run: from google.colab import files; files.upload()  (upload kaggle.json)
2. Run: import os; os.makedirs('/root/.kaggle', exist_ok=True); !cp kaggle.json /root/.kaggle/
3. Run: pip install kagglehub
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MUSAN_URL: str = "http://www.openslr.org/resources/17/musan.tar.gz"
_MUSAN_TARBALL: str = "musan.tar.gz"
_MUSAN_EXTRACTED: str = "musan"

_ZENODO_BELIZE_URL: str = "https://zenodo.org/record/4993368"

_VOICEBANK_DEMAND_URL: str = "https://datashare.ed.ac.uk/handle/10283/2791"

_TARGET_SR: int = 16_000


# ---------------------------------------------------------------------------
# Audio preprocessing helper
# ---------------------------------------------------------------------------

def _resample_and_normalize_folder(
    folder: Path,
    target_sr: int = _TARGET_SR,
    peak: float = 0.95,
) -> int:
    """Resample all audio in *folder* to *target_sr* Hz and peak-normalise.

    Requires soundfile + scipy.  Files are resampled in-place only if
    the sample rate differs.  Skips files that cannot be read.

    Args:
        folder:    Root directory of audio files.
        target_sr: Target sample rate in Hz.
        peak:      Target peak amplitude (0.0-1.0).

    Returns:
        Number of files successfully processed.
    """
    try:
        import numpy as np
        import soundfile as sf
        from scipy import signal as scipy_signal
    except ImportError:
        logger.warning(
            "soundfile/scipy not installed; skipping resample_and_normalize_folder."
        )
        return 0

    exts = {".wav", ".flac", ".mp3", ".ogg"}
    files = [p for ext in exts for p in folder.rglob(f"*{ext}")]
    processed = 0

    for f in files:
        try:
            data, orig_sr = sf.read(str(f), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data[:, 0]

            if orig_sr != target_sr:
                n_out = int(len(data) * target_sr / orig_sr)
                data = scipy_signal.resample(data, n_out).astype("float32")

            pk = float(np.max(np.abs(data)))
            if pk > 1e-8:
                data = (data * (peak / pk)).astype("float32")

            sf.write(str(f), data, target_sr, subtype="FLOAT")
            processed += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip %s during resample: %s", f.name, exc)

    logger.info("resample_and_normalize_folder: processed %d files in %s", processed, folder)
    return processed


# ---------------------------------------------------------------------------
# Manifest builder (thin wrapper around existing build_noise_manifest)
# ---------------------------------------------------------------------------

def _build_and_save_noise_manifest(
    folder: Path,
    source: str,
    manifest_dir: Path,
    cfg: Config,
) -> None:
    """Build a noise manifest for *folder* and save it to *manifest_dir*.

    Args:
        folder:       Root directory of audio files.
        source:       Source label (e.g. 'urbansound8k').
        manifest_dir: Directory to write the CSV manifest file.
        cfg:          ANC-SIH Config object (for split ratios).
    """
    try:
        from anc.data.manifest import build_noise_manifest, save_manifest

        split_ratios = {
            "train": float(cfg.data.splits.train) if "data" in cfg and "splits" in cfg.data else 0.8,
            "val":   float(cfg.data.splits.val)   if "data" in cfg and "splits" in cfg.data else 0.1,
            "test":  float(cfg.data.splits.test)  if "data" in cfg and "splits" in cfg.data else 0.1,
        }
        df = build_noise_manifest(folder, source, split_ratios)
        if df.empty:
            logger.warning("Manifest for %s is empty after scanning %s", source, folder)
            return

        manifest_dir.mkdir(parents=True, exist_ok=True)
        out_path = manifest_dir / f"noise_{source}.csv"
        save_manifest(df, out_path)
        print(f"  Manifest saved: {out_path}  ({len(df)} rows)")
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not build manifest for %s: %s", source, exc)


# ---------------------------------------------------------------------------
# Download actions
# ---------------------------------------------------------------------------

def _try_kaggle_download(
    dataset_slug: str,
    dest_dir: Path,
) -> bool:
    """Attempt to download a Kaggle dataset using kagglehub, then CLI fallback.

    Tries ``kagglehub`` first; if not installed, tries the Kaggle CLI.  Prints
    clear setup instructions if both fail.

    Args:
        dataset_slug: Kaggle dataset identifier (e.g. 'chrisfilo/urbansound8k').
        dest_dir:     Destination directory for the downloaded files.

    Returns:
        True if download succeeded, False otherwise.
    """
    # Attempt 1: kagglehub
    try:
        import kagglehub  # type: ignore[import]

        print(f"  [kagglehub] Downloading {dataset_slug} ...")
        path = kagglehub.dataset_download(dataset_slug)
        src = Path(path)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy downloaded files to dest_dir
        if src != dest_dir:
            for item in src.iterdir():
                target = dest_dir / item.name
                if not target.exists():
                    if item.is_dir():
                        shutil.copytree(str(item), str(target))
                    else:
                        shutil.copy2(str(item), str(target))
        print(f"  [kagglehub] Downloaded to: {dest_dir}")
        return True
    except ImportError:
        logger.debug("kagglehub not installed; trying Kaggle CLI.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("kagglehub download failed for %s: %s", dataset_slug, exc)

    # Attempt 2: Kaggle CLI
    try:
        import subprocess

        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", dataset_slug,
               "--unzip", "-p", str(dest_dir)]
        print(f"  [kaggle CLI] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"  [kaggle CLI] Downloaded to: {dest_dir}")
            return True
        else:
            logger.warning("Kaggle CLI error: %s", result.stderr[:500])
    except FileNotFoundError:
        logger.debug("Kaggle CLI not installed.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kaggle CLI download failed for %s: %s", dataset_slug, exc)

    # Both failed: print instructions
    print(f"\n  [ERROR] Could not download {dataset_slug} automatically.")
    print("  Please set up the Kaggle API manually:")
    print(KAGGLE_SETUP_INSTRUCTIONS)
    print(f"  Then manually download the dataset at:")
    print(f"    https://www.kaggle.com/datasets/{dataset_slug}")
    print(f"  and extract it to: {dest_dir}\n")
    return False


def _download_kaggle_urbansound8k(cfg: Config) -> None:
    """Download UrbanSound8K via Kaggle.

    Dataset: chrisfilo/urbansound8k
    Target : <raw_dir>/urbansound8k/

    Args:
        cfg: ANC-SIH Config object.
    """
    raw_dir = Path(cfg.data.raw_dir) if "data" in cfg and "raw_dir" in cfg.data \
        else _REPO / "data" / "raw"
    dest = raw_dir / "urbansound8k"

    if dest.exists() and any(dest.rglob("*.wav")):
        print(f"  UrbanSound8K already exists at {dest} -- skipping download.")
        return

    print("\n[download_data_v2] Downloading UrbanSound8K ...")
    ok = _try_kaggle_download("chrisfilo/urbansound8k", dest)

    if ok:
        print("  Resampling UrbanSound8K to 16 kHz ...")
        n = _resample_and_normalize_folder(dest)
        print(f"  Processed {n} files.")
        manifest_dir = Path(cfg.data.manifest_dir) if "data" in cfg and "manifest_dir" in cfg.data \
            else _REPO / "data" / "manifests"
        _build_and_save_noise_manifest(dest, "urbansound8k", manifest_dir, cfg)


def _download_kaggle_gunshot(cfg: Config) -> None:
    """Download the Kaggle gunshot audio dataset.

    Dataset: emrahaydemr/gunshot-audio-dataset
    Target : <raw_dir>/gunshot_kaggle/

    Args:
        cfg: ANC-SIH Config object.
    """
    raw_dir = Path(cfg.data.raw_dir) if "data" in cfg and "raw_dir" in cfg.data \
        else _REPO / "data" / "raw"
    dest = raw_dir / "gunshot_kaggle"

    if dest.exists() and any(dest.rglob("*.wav")):
        print(f"  Gunshot dataset already exists at {dest} -- skipping download.")
        return

    print("\n[download_data_v2] Downloading Kaggle gunshot dataset ...")
    ok = _try_kaggle_download("emrahaydemr/gunshot-audio-dataset", dest)

    if ok:
        print("  Resampling gunshot dataset to 16 kHz ...")
        n = _resample_and_normalize_folder(dest)
        print(f"  Processed {n} files.")
        manifest_dir = Path(cfg.data.manifest_dir) if "data" in cfg and "manifest_dir" in cfg.data \
            else _REPO / "data" / "manifests"
        _build_and_save_noise_manifest(dest, "kaggle_gunshot", manifest_dir, cfg)


def _show_belize_instructions(cfg: Config) -> None:
    """Print instructions for the Belize gunshot dataset (Zenodo 4993368).

    Zenodo datasets require acceptance of dataset-specific terms; automatic
    download is therefore not attempted.

    Args:
        cfg: ANC-SIH Config object (unused but kept for consistent signature).
    """
    raw_dir = Path(cfg.data.raw_dir) if "data" in cfg and "raw_dir" in cfg.data \
        else _REPO / "data" / "raw"
    dest = raw_dir / "belize_gunshots"

    print("\n[download_data_v2] Belize Gunshot Dataset (Zenodo record 4993368)")
    print("  This dataset requires accepting terms on Zenodo before downloading.")
    print("  Manual steps:")
    print(f"    1. Visit: {_ZENODO_BELIZE_URL}")
    print("    2. Accept the dataset terms and click 'Download'.")
    print(f"    3. Extract the ZIP / TAR to: {dest}")
    print(f"    4. Re-run with --build-manifest to update manifests.\n")


def _download_musan(cfg: Config) -> None:
    """Download and extract the MUSAN corpus from OpenSLR-17.

    URL: http://www.openslr.org/17/
    Target : <raw_dir>/musan/

    Args:
        cfg: ANC-SIH Config object.
    """
    raw_dir = Path(cfg.data.raw_dir) if "data" in cfg and "raw_dir" in cfg.data \
        else _REPO / "data" / "raw"
    dest = raw_dir / "musan"

    if dest.exists() and any(dest.rglob("*.wav")):
        print(f"  MUSAN already exists at {dest} -- skipping download.")
        return

    print(f"\n[download_data_v2] Downloading MUSAN from OpenSLR ...")
    print(f"  URL: {_MUSAN_URL}")
    print("  This may take a while (~10 GB).")

    tarball_path = raw_dir / _MUSAN_TARBALL
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Stream download with progress reporting
        def _report_hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100.0, 100.0 * downloaded / total_size)
                print(f"\r  Downloading ... {pct:.1f}%", end="", flush=True)

        print(f"  Saving to: {tarball_path}")
        urllib.request.urlretrieve(_MUSAN_URL, str(tarball_path), reporthook=_report_hook)
        print()  # newline after progress
        print(f"  Download complete. Extracting ...")

        with tarfile.open(str(tarball_path), "r:gz") as tf:
            tf.extractall(str(raw_dir))

        print(f"  Extracted to: {raw_dir / _MUSAN_EXTRACTED}")

        # Rename if needed
        extracted_path = raw_dir / _MUSAN_EXTRACTED
        if extracted_path.exists() and extracted_path != dest:
            extracted_path.rename(dest)

        # Clean up tarball
        if tarball_path.exists():
            tarball_path.unlink()
            print(f"  Tarball removed.")

        print("  Resampling MUSAN to 16 kHz ...")
        n = _resample_and_normalize_folder(dest)
        print(f"  Processed {n} files.")

        manifest_dir = Path(cfg.data.manifest_dir) if "data" in cfg and "manifest_dir" in cfg.data \
            else _REPO / "data" / "manifests"
        _build_and_save_noise_manifest(dest, "musan", manifest_dir, cfg)

    except Exception as exc:  # noqa: BLE001
        logger.error("MUSAN download/extraction failed: %s", exc)
        print(f"\n  [ERROR] MUSAN download failed: {exc}")
        print("  Manual steps:")
        print(f"    1. Visit: http://www.openslr.org/17/")
        print("    2. Download musan.tar.gz manually.")
        print(f"    3. Extract it so that the 'musan/' folder is inside: {raw_dir}")
        print(f"    4. Re-run with --build-manifest.\n")


def _show_voicebank_instructions(cfg: Config) -> None:
    """Print instructions for VoiceBank-DEMAND (Edinburgh DataShare).

    Registration is required before downloading; automatic download is not
    attempted.

    Args:
        cfg: ANC-SIH Config object (unused but kept for consistent signature).
    """
    raw_dir = Path(cfg.data.raw_dir) if "data" in cfg and "raw_dir" in cfg.data \
        else _REPO / "data" / "raw"
    dest = raw_dir / "voice_bank_demand"

    print("\n[download_data_v2] VoiceBank-DEMAND (Edinburgh DataShare)")
    print("  This dataset requires free registration at Edinburgh DataShare.")
    print("  Manual steps:")
    print(f"    1. Visit: {_VOICEBANK_DEMAND_URL}")
    print("    2. Create a free account and log in.")
    print("    3. Download both 'clean_trainset_28spk_wav.zip' and")
    print("       'noisy_trainset_28spk_wav.zip' (and test equivalents).")
    print(f"    4. Extract all ZIPs to: {dest}")
    print(f"    5. Re-run with --build-manifest.\n")


def _build_all_manifests(cfg: Config) -> None:
    """Rebuild all noise and speech manifests from available data.

    Args:
        cfg: ANC-SIH Config object.
    """
    import pandas as pd
    from anc.data.manifest import (
        build_noise_manifest,
        build_speech_manifest,
        build_synthetic_manifest,
        merge_manifests,
        save_manifest,
    )

    raw_dir = Path(cfg.data.raw_dir) if "data" in cfg and "raw_dir" in cfg.data \
        else _REPO / "data" / "raw"
    manifest_dir = Path(cfg.data.manifest_dir) if "data" in cfg and "manifest_dir" in cfg.data \
        else _REPO / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    split_ratios: dict = {
        "train": float(cfg.data.splits.train) if "data" in cfg and "splits" in cfg.data else 0.8,
        "val":   float(cfg.data.splits.val)   if "data" in cfg and "splits" in cfg.data else 0.1,
        "test":  float(cfg.data.splits.test)  if "data" in cfg and "splits" in cfg.data else 0.1,
    }

    # Known noise dataset folders and their source labels
    noise_sources = [
        ("urbansound8k",   raw_dir / "urbansound8k"),
        ("kaggle_gunshot", raw_dir / "gunshot_kaggle"),
        ("belize",         raw_dir / "belize_gunshots"),
        ("musan",          raw_dir / "musan"),
        ("voice_bank",     raw_dir / "voice_bank_demand"),
        ("esc50",          raw_dir / "esc50"),
        ("demand",         raw_dir / "demand"),
        ("openslr_rir",    raw_dir / "openslr_rir"),
    ]

    # Also include datasets configured in base.yaml
    if "data" in cfg and "noise_datasets" in cfg.data:
        for ds in cfg.data.noise_datasets:
            try:
                label = str(ds.source)
                path  = Path(str(ds.path))
                if (label, path) not in [(s, p) for s, p in noise_sources]:
                    noise_sources.append((label, path))
            except AttributeError:
                pass

    noise_dfs = []
    for source_name, folder in noise_sources:
        if folder.exists():
            df = build_noise_manifest(folder, source_name, split_ratios)
            if not df.empty:
                noise_dfs.append(df)
                logger.info("Noise %s: %d files", source_name, len(df))
        else:
            logger.debug("Noise folder not found: %s", folder)

    # Always supplement with synthetic noise
    for split in ("train", "val", "test"):
        synth = build_synthetic_manifest(split=split, n_per_category=100)
        noise_dfs.append(synth)

    noise_all = merge_manifests(*noise_dfs) if noise_dfs else pd.DataFrame()
    for split in ("train", "val", "test"):
        sub = noise_all[noise_all["split"] == split] if not noise_all.empty else pd.DataFrame()
        save_manifest(sub, manifest_dir / f"noise_{split}.csv")
        logger.info("noise_%s.csv: %d files", split, len(sub))

    # Speech datasets
    speech_sources = []
    if "data" in cfg and "speech_datasets" in cfg.data:
        for ds in cfg.data.speech_datasets:
            try:
                speech_sources.append((str(ds.source), Path(str(ds.path))))
            except AttributeError:
                pass

    # Fall back to known locations
    for label, folder in [
        ("librispeech",    raw_dir / "LibriSpeech"),
        ("voice_bank",     raw_dir / "voice_bank_demand"),
    ]:
        if (label, folder) not in speech_sources:
            speech_sources.append((label, folder))

    speech_dfs = []
    for source_name, folder in speech_sources:
        if folder.exists():
            df = build_speech_manifest(folder, source_name, split_ratios)
            if not df.empty:
                speech_dfs.append(df)

    speech_all = merge_manifests(*speech_dfs) if speech_dfs else pd.DataFrame()
    for split in ("train", "val", "test"):
        sub = speech_all[speech_all["split"] == split] if not speech_all.empty else pd.DataFrame()
        save_manifest(sub, manifest_dir / f"speech_{split}.csv")

    # Summary
    print("\n===== Manifest Summary (v2) =====")
    for split in ("train", "val", "test"):
        s_path = manifest_dir / f"speech_{split}.csv"
        n_path = manifest_dir / f"noise_{split}.csv"
        s_count = len(pd.read_csv(s_path)) if s_path.exists() else 0
        n_count = len(pd.read_csv(n_path)) if n_path.exists() else 0
        print(f"  {split:8s}: {s_count:5d} speech, {n_count:5d} noise")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for download_data_v2."""
    parser = argparse.ArgumentParser(
        description="Download and prepare Phase 4 datasets for ANC-SIH.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="configs/base.yaml",
        help="Path to YAML config (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--kaggle-urbansound8k", action="store_true",
        help="Download UrbanSound8K via Kaggle (chrisfilo/urbansound8k).",
    )
    parser.add_argument(
        "--kaggle-gunshot", action="store_true",
        help="Download gunshot dataset via Kaggle (emrahaydemr/gunshot-audio-dataset).",
    )
    parser.add_argument(
        "--belize", action="store_true",
        help="Print Belize gunshot dataset download instructions (Zenodo 4993368).",
    )
    parser.add_argument(
        "--musan", action="store_true",
        help="Download MUSAN corpus from OpenSLR (auto-download, ~10 GB).",
    )
    parser.add_argument(
        "--voice-bank", action="store_true",
        help="Print VoiceBank-DEMAND download instructions (Edinburgh DataShare).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all auto-downloadable sources (Kaggle US8K, Kaggle gunshot, MUSAN).",
    )
    parser.add_argument(
        "--build-manifest", action="store_true",
        help="Rebuild all manifests from available data after downloading.",
    )
    args = parser.parse_args()

    # Load config gracefully
    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning(
            "Config not found at %s -- using minimal built-in defaults.", config_path
        )
        from anc.utils.config import Config
        cfg = Config({
            "audio":   {"sample_rate": 16000},
            "data":    {"raw_dir": "data/raw", "manifest_dir": "data/manifests",
                        "splits": {"train": 0.8, "val": 0.1, "test": 0.1}},
            "project": {"seed": 42},
        })
    else:
        cfg = load_config(config_path)

    seed = int(cfg.project.seed) if "project" in cfg and "seed" in cfg.project else 42
    set_seed(seed)

    if args.all:
        args.kaggle_urbansound8k = True
        args.kaggle_gunshot      = True
        args.musan               = True
        args.build_manifest      = True

    any_action = False

    if args.kaggle_urbansound8k:
        any_action = True
        try:
            _download_kaggle_urbansound8k(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("UrbanSound8K download failed: %s", exc)
            print(f"\n  [ERROR] UrbanSound8K download encountered an error: {exc}")
            print(KAGGLE_SETUP_INSTRUCTIONS)

    if args.kaggle_gunshot:
        any_action = True
        try:
            _download_kaggle_gunshot(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Kaggle gunshot download failed: %s", exc)
            print(f"\n  [ERROR] Kaggle gunshot download encountered an error: {exc}")
            print(KAGGLE_SETUP_INSTRUCTIONS)

    if args.belize:
        any_action = True
        _show_belize_instructions(cfg)

    if args.musan:
        any_action = True
        try:
            _download_musan(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("MUSAN download failed: %s", exc)
            print(f"\n  [ERROR] MUSAN download encountered an error: {exc}")
            print("  Please download manually from http://www.openslr.org/17/")

    if args.voice_bank:
        any_action = True
        _show_voicebank_instructions(cfg)

    if args.build_manifest:
        any_action = True
        try:
            _build_all_manifests(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Manifest build failed: %s", exc)
            print(f"\n  [ERROR] Manifest build encountered an error: {exc}\n")

    if not any_action:
        parser.print_help()
        print("\nHint: use --all to download all auto-downloadable datasets.")
        print("      use --belize or --voice-bank to see manual download instructions.")


if __name__ == "__main__":
    main()
