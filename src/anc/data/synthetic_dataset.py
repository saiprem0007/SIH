"""PureSyntheticDataset -- file-free dataset for debug/CI training."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
from anc.data.synthetic_noise import gen_synthetic_noise, SYNTHETIC_GENERATORS
from anc.data.mixing import SNR_TIERS

# SNR_TIERS is {tier_name: center_snr_db}
_TIER_NAMES = list(SNR_TIERS.keys())
_NOISE_CATS = list(SYNTHETIC_GENERATORS.keys())
_SNR_HALF_RANGE = 2.5  # +/- around tier centre


def _gen_speech_like(duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(duration * sr)
    raw = rng.standard_normal(n).astype(np.float32)
    rate = float(rng.uniform(3.0, 7.0))
    t = np.arange(n, dtype=np.float32) / sr
    raw *= (0.5 + 0.5 * np.sin(2 * np.pi * rate * t)).astype(np.float32)
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, [300.0 / (sr / 2), 3400.0 / (sr / 2)], btype='bandpass', output='sos')
        raw = sosfilt(sos, raw).astype(np.float32)
    except Exception:
        pass
    pk = float(np.abs(raw).max())
    if pk > 1e-6:
        raw = (raw / pk) * float(rng.uniform(0.3, 0.7))
    return raw


class PureSyntheticDataset(Dataset):
    """Fully synthetic speech-enhancement dataset (no audio files needed).

    Args:
        cfg          : config object (needs cfg.audio.sample_rate, cfg.data.segment_dur)
        split        : train | val | test  (affects seed offset only)
        virtual_size : number of samples per epoch
        seed         : base RNG seed
    """
    _SPLIT_OFFSET: dict = {'train': 0, 'val': 100000, 'test': 200000}

    def __init__(
        self,
        cfg,
        split: str = 'train',
        virtual_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.sr = int(cfg.audio.sample_rate)
        self.seg_len = int(float(cfg.data.segment_dur) * self.sr)
        self.virtual_size = virtual_size
        self.seed_base = seed + self._SPLIT_OFFSET.get(split, 0)

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed_base + idx)
        seg_dur = self.seg_len / self.sr

        # Clean speech (synthetic voiced-band noise)
        clean = _gen_speech_like(seg_dur, self.sr, rng)[: self.seg_len]
        if len(clean) < self.seg_len:
            clean = np.pad(clean, (0, self.seg_len - len(clean)))

        # Pick SNR tier and noise category
        tier_name = _TIER_NAMES[int(rng.integers(len(_TIER_NAMES)))]
        centre_snr = SNR_TIERS[tier_name]
        snr_db = float(rng.uniform(centre_snr - _SNR_HALF_RANGE, centre_snr + _SNR_HALF_RANGE))
        cat = _NOISE_CATS[int(rng.integers(len(_NOISE_CATS)))]
        noise = gen_synthetic_noise(cat, seg_dur, self.sr, seed=int(rng.integers(int(1e6))))[: self.seg_len]
        if len(noise) < self.seg_len:
            noise = np.pad(noise, (0, self.seg_len - len(noise)))

        # Mix
        crms = float(np.sqrt(np.mean(clean ** 2)) + 1e-9)
        nrms = float(np.sqrt(np.mean(noise ** 2)) + 1e-9)
        sc = crms / (nrms * (10 ** (snr_db / 20)))
        primary = np.clip(clean + noise * sc, -1.0, 1.0).astype(np.float32)

        # Reference (noise-dominant, attenuated speech leakage)
        ref_snr = float(rng.uniform(-30.0, -15.0))
        rsc = crms / (nrms * (10 ** (ref_snr / 20)))
        reference = np.clip(noise * rsc + clean * float(rng.uniform(0.01, 0.05)), -1.0, 1.0).astype(np.float32)
        clean = np.clip(clean, -1.0, 1.0).astype(np.float32)

        return {
            'primary':     torch.from_numpy(primary),
            'reference':   torch.from_numpy(reference),
            'clean':       torch.from_numpy(clean),
            'snr_tier':    tier_name,
            'noise_class': cat,
        }
