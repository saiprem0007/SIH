"""
Phase 4 pytest tests for the leakage control module.
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.data.leakage import (
    extract_freesound_id,
    fix_freesound_split_leakage,
    split_by_gun_model,
    compute_spectral_hash,
    find_near_duplicates,
    run_leakage_checks,
)


# ---------------------------------------------------------------------------
# 1. extract_freesound_id -- UrbanSound8K format
# ---------------------------------------------------------------------------

def test_extract_freesound_id_urbansound():
    path = "data/raw/UrbanSound8K/audio/fold1/17853-0-0-0.wav"
    result = extract_freesound_id(path, source="urbansound8k")
    assert result == "17853", f"Expected '17853', got {result!r}"


def test_extract_freesound_id_esc50():
    # ESC-50 filename: '1-137-A-32.wav'
    path = "data/raw/ESC-50-master/audio/1-137-A-32.wav"
    result = extract_freesound_id(path, source="esc50")
    assert result == "1", f"Expected '1', got {result!r}"


# ---------------------------------------------------------------------------
# 2. extract_freesound_id -- unknown source returns None
# ---------------------------------------------------------------------------

def test_extract_freesound_id_unknown_returns_none():
    path = "some_random_file_name.wav"
    # Stem 'some_random_file_name' has no numeric prefix
    result = extract_freesound_id(path, source="unknown")
    assert result is None, f"Expected None, got {result!r}"


# ---------------------------------------------------------------------------
# 3. fix_freesound_split_leakage -- no shared IDs, no changes
# ---------------------------------------------------------------------------

def test_fix_freesound_no_shared_ids():
    df = pd.DataFrame({
        "path": [
            "11111-0-0-0.wav",
            "22222-0-0-0.wav",
            "33333-A-32.wav",
            "44444-A-10.wav",
        ],
        "source": ["urbansound8k", "urbansound8k", "esc50", "esc50"],
        "split":  ["train", "val", "test", "train"],
    })
    result = fix_freesound_split_leakage(df)
    # No shared IDs -> splits unchanged
    assert list(result["split"]) == ["train", "val", "test", "train"]


def test_fix_freesound_shared_id_reassigned():
    """Clip with same Freesound ID in both datasets goes to same split."""
    df = pd.DataFrame({
        "path": [
            "12345-0-0-0.wav",   # UrbanSound8K, Freesound ID 12345
            "12345-A-32.wav",    # ESC-50, same Freesound ID 12345
        ],
        "source": ["urbansound8k", "esc50"],
        "split":  ["train", "test"],   # CONFLICT
    })
    result = fix_freesound_split_leakage(df)
    # Both should end up in the same split (the first alphabetically)
    assert result["split"].nunique() == 1, (
        f"Expected all splits equal, got: {list(result['split'])}"
    )


# ---------------------------------------------------------------------------
# 4. split_by_gun_model
# ---------------------------------------------------------------------------

def test_split_by_gun_model():
    model_map = {
        "test_only": ["glock_17", "ak47"],
        "val_only":  ["beretta_92"],
        "train":     ["default"],
    }
    df = pd.DataFrame({
        "path": [
            "glock_17_shot_001.wav",
            "ak47_burst_002.wav",
            "beretta_92_single.wav",
            "usp_shot.wav",
            "usp_shot2.wav",
        ],
        "split": ["train"] * 5,
    })
    result = split_by_gun_model(df, model_map=model_map)
    assert result.loc[0, "split"] == "test",  "glock_17 should be test"
    assert result.loc[1, "split"] == "test",  "ak47 should be test"
    assert result.loc[2, "split"] == "val",   "beretta_92 should be val"
    assert result.loc[3, "split"] == "train", "usp should remain train"
    assert result.loc[4, "split"] == "train", "usp should remain train"


# ---------------------------------------------------------------------------
# 5. compute_spectral_hash -- deterministic
# ---------------------------------------------------------------------------

def test_spectral_hash_deterministic(tmp_path):
    """Same WAV produces the same hash on two calls."""
    import soundfile as sf
    sr = 16000
    wav = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    fpath = tmp_path / "test_sine.wav"
    sf.write(str(fpath), wav, sr)

    h1 = compute_spectral_hash(fpath, sr=sr)
    h2 = compute_spectral_hash(fpath, sr=sr)
    assert h1 == h2, f"Hash not deterministic: {h1} vs {h2}"
    assert h1 != -1, "Hash returned error code -1"


def test_spectral_hash_different_for_different_files(tmp_path):
    """Sine at 440 Hz and noise should have different hashes (usually)."""
    import soundfile as sf
    sr = 16000
    rng = np.random.default_rng(0)
    sine  = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    noise = rng.standard_normal(sr).astype(np.float32)

    f1 = tmp_path / "sine.wav"
    f2 = tmp_path / "noise.wav"
    sf.write(str(f1), sine,  sr)
    sf.write(str(f2), noise, sr)

    h1 = compute_spectral_hash(f1, sr=sr)
    h2 = compute_spectral_hash(f2, sr=sr)
    # They SHOULD differ; if same it's a very rare collision
    xor  = (h1 ^ h2) & 0xFFFFFFFFFFFFFFFF
    hamming = bin(xor).count("1")
    assert hamming > 0, "Sine and noise produced identical spectral hash"


# ---------------------------------------------------------------------------
# 6. find_near_duplicates -- identical hashes detected
# ---------------------------------------------------------------------------

def test_find_near_duplicates_identical():
    df = pd.DataFrame({
        "spec_hash": [12345, 12345, 99999],
    })
    dups = find_near_duplicates(df, hamming_threshold=0)
    assert len(dups) >= 1, "Expected at least one duplicate pair"
    # The pair (0, 1) should be found
    pairs = [(a, b) for a, b, _ in dups]
    assert (0, 1) in pairs, f"Pair (0,1) not in {pairs}"


# ---------------------------------------------------------------------------
# 7. find_near_duplicates -- very different hashes not flagged
# ---------------------------------------------------------------------------

def test_find_near_duplicates_different():
    # Create hashes where every pair differs in > 5 bits
    hashes = [0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA]
    df = pd.DataFrame({"spec_hash": hashes})
    dups = find_near_duplicates(df, hamming_threshold=5)
    assert len(dups) == 0, f"Unexpected duplicates found: {dups}"


# ---------------------------------------------------------------------------
# 8. run_leakage_checks -- no violation
# ---------------------------------------------------------------------------

def _make_speech_df(speakers, paths, split):
    return pd.DataFrame({
        "speaker_id": speakers,
        "path":       paths,
        "split":      split,
    })


def _make_noise_df(paths, split):
    return pd.DataFrame({
        "path":  paths,
        "split": split,
    })


def test_run_leakage_checks_no_violation():
    s_train = _make_speech_df(["s1", "s2"], ["a.wav", "b.wav"], ["train", "train"])
    s_val   = _make_speech_df(["s3"],        ["c.wav"],          ["val"])
    s_test  = _make_speech_df(["s4"],        ["d.wav"],          ["test"])
    n_train = _make_noise_df(["n1.wav", "n2.wav"], ["train", "train"])
    n_val   = _make_noise_df(["n3.wav"],            ["val"])
    n_test  = _make_noise_df(["n4.wav"],            ["test"])

    result = run_leakage_checks(s_train, s_val, s_test, n_train, n_val, n_test)
    assert result["passed"] is True
    assert result["violations"] == []


# ---------------------------------------------------------------------------
# 9. run_leakage_checks -- speaker violation raises AssertionError
# ---------------------------------------------------------------------------

def test_run_leakage_checks_speaker_violation():
    # s1 appears in both train and val -> violation
    s_train = _make_speech_df(["s1", "s2"], ["a.wav", "b.wav"], ["train", "train"])
    s_val   = _make_speech_df(["s1", "s3"], ["c.wav", "d.wav"], ["val", "val"])
    s_test  = _make_speech_df(["s4"],       ["e.wav"],           ["test"])
    n_train = _make_noise_df(["n1.wav"], ["train"])
    n_val   = _make_noise_df(["n2.wav"], ["val"])
    n_test  = _make_noise_df(["n3.wav"], ["test"])

    with pytest.raises(AssertionError, match="Speaker leakage"):
        run_leakage_checks(s_train, s_val, s_test, n_train, n_val, n_test)


# ---------------------------------------------------------------------------
# 10. run_leakage_checks -- path violation raises AssertionError
# ---------------------------------------------------------------------------

def test_run_leakage_checks_path_violation():
    s_train = _make_speech_df(["s1"], ["a.wav"], ["train"])
    s_val   = _make_speech_df(["s2"], ["b.wav"], ["val"])
    s_test  = _make_speech_df(["s3"], ["c.wav"], ["test"])
    # same noise file in train AND test
    n_train = _make_noise_df(["shared.wav"], ["train"])
    n_val   = _make_noise_df(["n2.wav"],     ["val"])
    n_test  = _make_noise_df(["shared.wav"], ["test"])  # LEAK

    with pytest.raises(AssertionError, match="Path leakage"):
        run_leakage_checks(s_train, s_val, s_test, n_train, n_val, n_test)
