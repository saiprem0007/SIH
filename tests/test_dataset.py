"""Unit tests for dataset classes and data pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from anc.data.synthetic_noise import SYNTHETIC_GENERATORS, gen_synthetic_noise
from anc.data.manifest import build_noise_manifest, build_speech_manifest

SR = 16_000
SEG_LEN = 4 * SR


# ---------------------------------------------------------------------------
# Synthetic noise generators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", list(SYNTHETIC_GENERATORS.keys()))
def test_synthetic_noise_shape_and_range(category: str):
    """Each generator must return float32 array of correct length in [-1, 1]."""
    sig = gen_synthetic_noise(category, duration_s=2.0, sr=SR, seed=42)
    assert sig.shape == (2 * SR,), f"Shape mismatch for category={category}: {sig.shape}"
    assert sig.dtype == np.float32, f"dtype mismatch: {sig.dtype}"
    assert np.max(np.abs(sig)) > 0.01, "Signal is near-silent"
    assert np.max(np.abs(sig)) <= 1.0, f"Signal exceeds 1.0: {np.max(np.abs(sig))}"


def test_synthetic_noise_unknown_category():
    """Unknown category must fall back gracefully (no crash, returns array)."""
    sig = gen_synthetic_noise("totally_unknown_category_xyz", 1.0, SR, seed=0)
    assert isinstance(sig, np.ndarray)
    assert len(sig) == SR


def test_synthetic_noise_reproducible():
    """Same seed must give identical output."""
    a = gen_synthetic_noise("wind", 2.0, SR, seed=99)
    b = gen_synthetic_noise("wind", 2.0, SR, seed=99)
    np.testing.assert_array_equal(a, b)


def test_synthetic_noise_different_seeds():
    """Different seeds must give different output."""
    a = gen_synthetic_noise("siren", 2.0, SR, seed=1)
    b = gen_synthetic_noise("siren", 2.0, SR, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def test_build_noise_manifest(tmp_audio_dir: Path):
    """Noise manifest must have required columns and valid split values."""
    noise_dir = tmp_audio_dir / "noise"
    df = build_noise_manifest(noise_dir, source="manual")

    if df.empty:
        pytest.skip("No noise files found in fixture (may be platform-specific).")

    required_cols = {"path", "source", "type", "category", "split"}
    assert required_cols.issubset(df.columns), f"Missing cols: {required_cols - set(df.columns)}"

    valid_splits = {"train", "val", "test"}
    assert set(df["split"].unique()).issubset(valid_splits), (
        f"Invalid split values: {set(df['split'].unique()) - valid_splits}"
    )

    # All paths must exist on disk
    for p in df["path"]:
        assert Path(p).exists(), f"Path does not exist: {p}"


def test_manifest_no_split_leakage(tmp_audio_dir: Path):
    """No file may appear in both train and test splits."""
    noise_dir = tmp_audio_dir / "noise"
    df = build_noise_manifest(noise_dir, source="manual")
    if df.empty:
        pytest.skip("Empty manifest.")

    train_paths = set(df[df["split"] == "train"]["path"])
    test_paths  = set(df[df["split"] == "test"]["path"])
    overlap = train_paths & test_paths
    assert len(overlap) == 0, f"Data leakage! {overlap}"


def test_speech_manifest_speaker_split(tmp_audio_dir: Path):
    """All files from one speaker must be in the same split (no speaker leakage)."""
    speech_dir = tmp_audio_dir / "speech"
    df = build_speech_manifest(speech_dir, source="manual")
    if df.empty:
        pytest.skip("Empty manifest.")

    for spk in df["speaker_id"].unique():
        splits = df[df["speaker_id"] == spk]["split"].unique()
        assert len(splits) == 1, (
            f"Speaker {spk} appears in multiple splits: {splits}"
        )


# ---------------------------------------------------------------------------
# FixedTestSet
# ---------------------------------------------------------------------------

def test_fixed_testset_loading(tmp_path: Path, base_cfg, tmp_audio_dir: Path):
    """FixedTestSet must load samples with correct structure."""
    from anc.data.dataset import generate_fixed_test_set, FixedTestSet
    from anc.data.manifest import build_speech_manifest, build_noise_manifest

    speech_dir = tmp_audio_dir / "speech"
    noise_dir  = tmp_audio_dir / "noise"

    # Use test split for both (with small manifests, split may coincide)
    s_df = build_speech_manifest(speech_dir, "manual", {"train": 0.5, "val": 0.0, "test": 0.5})
    n_df = build_noise_manifest(noise_dir,  "manual", {"train": 0.5, "val": 0.0, "test": 0.5})

    out_dir = tmp_path / "testset"

    # Tiny test set: 1 sample per tier, 1 per category
    generate_fixed_test_set(
        cfg=base_cfg,
        speech_manifest=s_df,
        noise_manifest=n_df,
        output_dir=out_dir,
        n_per_tier=1,
        n_per_category=1,
        seed=42,
    )

    ds = FixedTestSet(out_dir)
    assert len(ds) > 0

    sample = ds[0]
    for key in ("primary", "reference", "clean"):
        assert key in sample
        arr = sample[key].numpy()
        assert arr.ndim == 1
        assert len(arr) > 0

    # All waveforms same length
    assert len(sample["primary"]) == len(sample["reference"]) == len(sample["clean"])


def test_fixed_testset_reproducibility(tmp_path: Path, base_cfg, tmp_audio_dir: Path):
    """Generating with the same seed twice must give identical sample 0."""
    from anc.data.dataset import generate_fixed_test_set, FixedTestSet
    from anc.data.manifest import build_speech_manifest, build_noise_manifest

    speech_dir = tmp_audio_dir / "speech"
    noise_dir  = tmp_audio_dir / "noise"
    s_df = build_speech_manifest(speech_dir, "manual", {"train": 0.5, "val": 0.0, "test": 0.5})
    n_df = build_noise_manifest(noise_dir,  "manual", {"train": 0.5, "val": 0.0, "test": 0.5})

    out1 = tmp_path / "ts1"
    out2 = tmp_path / "ts2"

    generate_fixed_test_set(base_cfg, s_df, n_df, out1, n_per_tier=1, n_per_category=1, seed=42)
    generate_fixed_test_set(base_cfg, s_df, n_df, out2, n_per_tier=1, n_per_category=1, seed=42)

    ds1 = FixedTestSet(out1)
    ds2 = FixedTestSet(out2)

    assert len(ds1) == len(ds2)
    np.testing.assert_array_equal(ds1[0]["primary"].numpy(), ds2[0]["primary"].numpy())
