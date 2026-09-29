"""
Download and preparation helpers for the ANC-SIH data pipeline.

Provides auto-download for LibriSpeech, ESC-50, and OpenSLR RIRs,
manual folder indexing for Freesound/AudioSet clips, resampling utilities,
and a tiny demo dataset generator for rapid end-to-end testing.
"""
from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from math import gcd
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progress hook for urllib
# ---------------------------------------------------------------------------

class _ProgressBar:
    """urllib reporthook wrapper that shows a tqdm progress bar."""

    def __init__(self, desc: str = "Downloading") -> None:
        self._pbar: Optional[tqdm] = None
        self._desc = desc

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if self._pbar is None:
            self._pbar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=self._desc,
                leave=False,
            )
        downloaded = block_num * block_size
        if downloaded < total_size:
            self._pbar.update(min(block_size, total_size - self._pbar.n))
        else:
            self._pbar.close()
            self._pbar = None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_librispeech(
    root: str | Path,
    subsets: List[str] = ("train-clean-100", "dev-clean", "test-clean"),
) -> None:
    """Download LibriSpeech subsets using torchaudio.

    Args:
        root:    Root directory where data will be stored.
        subsets: List of subset names to download.
    """
    try:
        import torchaudio  # type: ignore
    except ImportError:
        logger.error("torchaudio not installed. Run: pip install torchaudio")
        return

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for subset in subsets:
        logger.info("Downloading LibriSpeech subset: %s …", subset)
        try:
            torchaudio.datasets.LIBRISPEECH(str(root), url=subset, download=True)
            logger.info("LibriSpeech %s: done.", subset)
        except Exception as exc:
            logger.error("Failed to download LibriSpeech %s: %s", subset, exc)


def download_esc50(root: str | Path) -> None:
    """Download ESC-50 from GitHub and extract it.

    Target: <root>/ESC-50-master/

    Args:
        root: Root directory for raw data.
    """
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    root = Path(root)
    dst_dir = root / "ESC-50-master"
    zip_path = root / "ESC-50-master.zip"

    if dst_dir.exists():
        logger.info("ESC-50 already exists at %s — skipping.", dst_dir)
        return

    root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ESC-50 from %s …", url)
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_ProgressBar("ESC-50"))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        zip_path.unlink(missing_ok=True)
        logger.info("ESC-50 extracted to %s", dst_dir)
    except Exception as exc:
        logger.error("ESC-50 download failed: %s", exc)


def download_demand(root: str | Path) -> None:
    """Print instructions for downloading the DEMAND dataset.

    DEMAND cannot be auto-downloaded without authentication.

    Args:
        root: Root directory where DEMAND should be placed.
    """
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║              DEMAND Dataset — Manual Download                ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║ 1. Go to: https://zenodo.org/record/1227121                  ║\n"
        "║ 2. Download the .zip files (16 noise environments).          ║\n"
        "║ 3. Extract them so the structure is:                         ║\n"
        f"║    {str(root)}/DEMAND/<ENV_NAME>/<CH01.wav …>               ║\n"
        "║ 4. Re-run: python scripts/download_data.py --build-manifest  ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )


def download_openslr_rir(root: str | Path) -> None:
    """Download OpenSLR RIR and noise data (SLR28).

    Target: <root>/RIRS_NOISES/

    Args:
        root: Root directory for raw data.
    """
    url = "https://www.openslr.org/resources/28/rirs_noises.zip"
    root = Path(root)
    dst_dir = root / "RIRS_NOISES"
    zip_path = root / "rirs_noises.zip"

    if dst_dir.exists():
        logger.info("RIRS_NOISES already exists at %s — skipping.", dst_dir)
        return

    root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading OpenSLR RIRs (SLR28) from %s …", url)
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_ProgressBar("RIRS_NOISES"))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        zip_path.unlink(missing_ok=True)
        logger.info("RIRS_NOISES extracted to %s", dst_dir)
    except Exception as exc:
        logger.error("OpenSLR RIR download failed: %s", exc)
        logger.info("You can manually download from %s and extract to %s", url, root)


def index_manual_folder(
    folder: str | Path,
    manifest_path: str | Path,
    source_name: str = "manual",
    license: str = "CC0",
) -> "pd.DataFrame":
    """Index a folder of manually downloaded wav files as a noise manifest.

    Args:
        folder:        Directory containing audio files (searched recursively).
        manifest_path: Where to save the resulting CSV manifest.
        source_name:   Source label to record in the manifest.
        license:       License string to record (e.g. 'CC0', 'CC-BY').

    Returns:
        DataFrame manifest.
    """
    import pandas as pd
    from anc.data.manifest import (
        VALID_EXTENSIONS,
        categorize_noise,
        get_audio_duration,
        save_manifest,
    )

    folder = Path(folder)
    if not folder.exists():
        logger.warning("Manual folder not found: %s", folder)
        return pd.DataFrame()

    files: List[Path] = []
    for ext in VALID_EXTENSIONS:
        files.extend(folder.rglob(f"*{ext}"))
    files = sorted(files)

    if not files:
        logger.warning("No audio files found in %s", folder)
        return pd.DataFrame()

    rows = []
    for f in tqdm(files, desc=f"Indexing {folder.name}"):
        rows.append({
            "path":     str(f),
            "source":   source_name,
            "type":     "noise",
            "category": categorize_noise(f, source_name),
            "split":    "train",
            "duration": get_audio_duration(f),
            "license":  license,
        })

    df = pd.DataFrame(rows)
    save_manifest(df, manifest_path)
    logger.info(
        "Indexed %d files from %s. Categories: %s",
        len(df), folder, df["category"].value_counts().to_dict(),
    )
    return df


def resample_audio(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a signal from orig_sr to target_sr using scipy.

    Args:
        signal:    Input 1-D float array.
        orig_sr:   Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled float32 array.
    """
    if orig_sr == target_sr:
        return signal.astype(np.float32)
    from scipy.signal import resample_poly
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = resample_poly(signal.astype(np.float64), up, down)
    return resampled.astype(np.float32)


def resample_and_normalize_folder(
    src_dir: str | Path,
    dst_dir: str | Path,
    target_sr: int = 16000,
    overwrite: bool = False,
) -> None:
    """Resample and peak-normalize all audio files in a folder tree.

    Args:
        src_dir:   Source directory (searched recursively for audio files).
        dst_dir:   Destination directory (relative structure preserved).
        target_sr: Target sample rate.
        overwrite: If False, skip files that already exist in dst_dir.
    """
    from anc.data.manifest import VALID_EXTENSIONS

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for ext in VALID_EXTENSIONS:
        files.extend(src_dir.rglob(f"*{ext}"))

    for src_file in tqdm(files, desc="Resampling"):
        rel = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel.with_suffix(".wav")
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists() and not overwrite:
            continue

        try:
            data, orig_sr = sf.read(str(src_file), always_2d=False)
            # Mono conversion
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            # Resample
            if orig_sr != target_sr:
                data = resample_audio(data, orig_sr, target_sr)
            # Peak normalize
            peak = np.max(np.abs(data))
            if peak > 1e-8:
                data = data * (0.95 / peak)
            sf.write(str(dst_file), data, target_sr, subtype="PCM_16")
        except Exception as exc:
            logger.warning("Could not process %s: %s", src_file, exc)


# ---------------------------------------------------------------------------
# Demo dataset generator
# ---------------------------------------------------------------------------

def _gen_speech_like(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic speech-like signal (bandpass noise)."""
    from scipy.signal import butter, lfilter
    white = rng.standard_normal(n_samples).astype(np.float64)
    nyq = sr / 2.0
    b, a = butter(4, [300 / nyq, 3400 / nyq], btype="band")
    filtered = lfilter(b, a, white).astype(np.float32)
    peak = np.max(np.abs(filtered))
    if peak > 1e-8:
        filtered = filtered * (0.9 / peak)
    return filtered


def generate_demo_dataset(
    speech_dir: str | Path,
    noise_dir: str | Path,
    n_speech: int = 50,
    n_noise: int = 50,
    sr: int = 16000,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """Create a tiny demo dataset for rapid end-to-end testing.

    If LibriSpeech already exists in speech_dir, samples from it.
    Otherwise generates synthetic speech-like signals.
    For noise, generates synthetic clips across all categories.

    Args:
        speech_dir: Directory to write speech files.
        noise_dir:  Directory to write noise files.
        n_speech:   Number of speech clips to generate/sample.
        n_noise:    Number of noise clips to generate/sample.
        sr:         Sample rate.
        seed:       Random seed.

    Returns:
        (speech_files, noise_files) as lists of Path objects.
    """
    from anc.data.manifest import VALID_EXTENSIONS
    from anc.data.synthetic_noise import SYNTHETIC_GENERATORS, gen_synthetic_noise

    rng = np.random.default_rng(seed)
    speech_dir = Path(speech_dir)
    noise_dir = Path(noise_dir)
    speech_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    seg_len = 4 * sr  # 4-second clips

    # --- Speech ---
    speech_files: List[Path] = []
    # Try sampling from real LibriSpeech first
    existing_speech: List[Path] = []
    for ext in VALID_EXTENSIONS:
        existing_speech.extend(speech_dir.rglob(f"*{ext}"))

    if len(existing_speech) >= n_speech:
        idxs = rng.choice(len(existing_speech), size=n_speech, replace=False)
        speech_files = [existing_speech[i] for i in idxs]
        logger.info("Demo: sampled %d speech files from existing data.", n_speech)
    else:
        logger.info("Demo: generating %d synthetic speech-like clips …", n_speech)
        for i in tqdm(range(n_speech), desc="Synthetic speech"):
            sig = _gen_speech_like(seg_len, sr, rng)
            out = speech_dir / "demo" / f"speech_{i:04d}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), sig, sr, subtype="PCM_16")
            speech_files.append(out)

    # --- Noise ---
    noise_files: List[Path] = []
    categories = list(SYNTHETIC_GENERATORS.keys())
    logger.info("Demo: generating %d synthetic noise clips …", n_noise)
    for i in tqdm(range(n_noise), desc="Synthetic noise"):
        cat = categories[i % len(categories)]
        sig = gen_synthetic_noise(cat, 4.0, sr, seed=int(rng.integers(0, 2**31)))
        out = noise_dir / "demo" / cat / f"noise_{i:04d}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), sig, sr, subtype="PCM_16")
        noise_files.append(out)

    logger.info(
        "Demo dataset ready: %d speech, %d noise files.",
        len(speech_files), len(noise_files),
    )
    return speech_files, noise_files
