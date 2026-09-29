"""Data sub-package for ANC-SIH.

Exports the mixing engine, dataset classes, and manifest utilities.
"""

from anc.data.mixing import (
    MixingEngine,
    compute_rms,
    compute_snr_db,
    mix_at_snr,
    NOISE_CATEGORIES,
    SNR_TIERS,
)
from anc.data.dataset import (
    SpeechEnhancementDataset,
    FixedTestSet,
    generate_fixed_test_set,
)
from anc.data.manifest import (
    build_noise_manifest,
    build_speech_manifest,
    save_manifest,
    load_manifest,
)

__all__ = [
    # mixing
    "MixingEngine",
    "compute_rms",
    "compute_snr_db",
    "mix_at_snr",
    "NOISE_CATEGORIES",
    "SNR_TIERS",
    # dataset
    "SpeechEnhancementDataset",
    "FixedTestSet",
    "generate_fixed_test_set",
    # manifest
    "build_noise_manifest",
    "build_speech_manifest",
    "save_manifest",
    "load_manifest",
]
